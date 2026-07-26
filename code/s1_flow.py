#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser
import copy
# import torch_npu
# from torch_npu.contrib import transfer_to_npu 

# from test import *
import test as test
# from dataset import CompositionDataset
from dataset import CompositionDataset,CompositionDataset_u
from utils import *

from config import DefaultConfig, set_seed
from fm import DeepFlowMatchingNet

# import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(model, optimizer,cfm_models ,optimizer_cfm, config,config_cfm, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    best_auc = 0
    model.train()
    for m in cfm_models:
        if m is not None:
            m.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    
    num_training_steps,warmup_steps = config_cfm.epochs*len(train_dataloader), config_cfm.warmup_epochs*len(train_dataloader)
    scheduler_cfm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cfm, T_max=num_training_steps-warmup_steps, eta_min=0)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []
    cur_step = 0
    for epoch in range(config.epoch_start, config.epochs):
        if epoch >= config.epoch_pt_end:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
    
        if epoch < config.epoch_fm_start:
            for m in cfm_models:
                if m is not None:
                    m.eval()
        else:
            for m in cfm_models:
                if m is not None:
                    m.train()
            
        # evaluate(model,cfm_models, test_dataset, config,compose=True)
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (epoch + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            loss = 0
            fm_loss = 0
            predict = list()
            image,att_label,obj_label ,pair = batch
            pair,att_label,obj_label  = pair.cuda(), att_label.cuda(), obj_label.cuda()
            image = image.cuda()
            labels_list = [pair.cuda(), att_label.cuda(), obj_label.cuda()]
            
            # image_features = model.encode_image(image)
            # image_features = image_features / image_features.norm(
            #         dim=-1, keepdim=True
            #     )
            if epoch < config.epoch_pt_end:
                text_embeddings,normalized_img_features = model(batch, train_pairs)
            else:
                with torch.no_grad():
                    text_embeddings, normalized_img_features = model(batch, train_pairs)

            pair_text_embeddings, attr_text_embeddings, obj_text_embeddings = text_embeddings
  
            b = batch[0].size()[0]
            norm_pair_text_features = pair_text_embeddings[torch.arange(b), pair, :].cuda()
            norm_attr_text_features = attr_text_embeddings[torch.arange(b), att_label, :].cuda()
            norm_obj_text_features = obj_text_embeddings[torch.arange(b), obj_label, :].cuda()
            
            norm_text_features = [norm_pair_text_features, norm_attr_text_features, norm_obj_text_features]
            if epoch < config.epoch_pt_end:
                if "0dis" in config.model_name:
                    reg_loss =  torch.sum( (norm_attr_text_features+norm_obj_text_features+normalized_img_features[0]-norm_pair_text_features)**2,dim=1 ).mean()
                    loss += config.vis_reg_weight*reg_loss
                else:
                    vis_reg_loss = torch.sum( (normalized_img_features[1]+normalized_img_features[2] -normalized_img_features[0])**2,dim=1 ).mean()
                    text_reg_loss = torch.sum( (norm_attr_text_features+norm_obj_text_features -norm_pair_text_features)**2,dim=1 ).mean()
                    loss += config.vis_reg_weight*vis_reg_loss
                    loss += config.text_reg_weight*text_reg_loss
                for i in range(3):
                    predict.append(
                        torch.einsum(
                        "bd, bkd->bk", 
                        normalized_img_features[i], 
                        text_embeddings[i] * model.clip.logit_scale.exp())
                    )
                loss += model.loss_calu(predict, batch)
                
                

                # normalize loss to account for batch accumulation
                loss = loss / config.gradient_accumulation_steps
            
            t_shared = torch.rand(image.size(0), 1, device='cuda')
            t_list = [t_shared, t_shared, t_shared]
            
            velocity_list = []
            if epoch >= config.epoch_fm_start:
                for i in range(3):
                    if config.path_2 and i == 0:
                        continue
                    gt_v = norm_text_features[i] - normalized_img_features[i]
                    interpolated_features = (1 - t_list[i]) * normalized_img_features[i].detach() + t_list[i] * norm_text_features[i].detach()
                    pred_vel = cfm_models[i](interpolated_features, t_list[i]) 
                    # velocity_list.append(pred_vel)
                    fm_loss += config_cfm.loss_weight*torch.sum((gt_v.detach()-pred_vel)**2,dim=1).mean()
                    
                    transfer_features = interpolated_features + pred_vel* (1-t_list[i])
                    # logits_v = transfer_features @ text_embeddings[i].detach().t() * model.clip.logit_scale.exp()
                    logits_v = torch.einsum(
                        "bd, bkd->bk", 
                        transfer_features, 
                        text_embeddings[i].detach() * model.clip.logit_scale.exp()
                        )
                    ce_flow = CrossEntropyLoss()(logits_v, labels_list[i])
                    fm_loss += config_cfm.logits_weights* ce_flow
                    
                    if config.vel_consistency_weight > 0:
                        pred_vel = torch.nn.functional.normalize(pred_vel, dim=1)
                        v = pred_vel                          # [B, D]
                        labels = labels_list[i]   
                        
                        sim = (v @ v.t()) / 0.07
                        B = sim.size(0)
                        
                        sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=sim.device), float("-inf"))
                        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
                        pos_mask.fill_diagonal_(False)
                        
                        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [B, B]
                        
                        pos_counts = pos_mask.sum(dim=1)                
                        valid = pos_counts > 0
                        pos_counts = pos_counts.clamp_min(1)
                        loss_per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_counts
                        
                        if valid.any():
                            vel_contrastive_loss = loss_per_anchor[valid].mean()
                            fm_loss += config.vel_consistency_weight * vel_contrastive_loss

                    
             
                    if config.leak_augmentation:
                        for j in range(1,3):
                            gt_v = norm_text_features[i] - normalized_img_features[i-j]
                            interpolated_features = (1 - t_list[i]) * normalized_img_features[i-j].detach() + t_list[i] * norm_text_features[i].detach()
                            pred_vel = cfm_models[i](interpolated_features, t_list[i]) 
                            # velocity_list.append(pred_vel)
                            fm_loss += config_cfm.loss_weight*torch.sum((gt_v.detach()-pred_vel)**2,dim=1).mean()

                            transfer_features = interpolated_features + pred_vel* (1-t_list[i])
                            # logits_v = transfer_features @ text_embeddings[i].detach().t() * model.clip.logit_scale.exp()
                            logits_v = torch.einsum(
                                "bd, bkd->bk", 
                                transfer_features, 
                                text_embeddings[i].detach() * model.clip.logit_scale.exp()
                                )
                            ce_flow = CrossEntropyLoss()(logits_v, labels_list[i])
                            fm_loss += config_cfm.logits_weights* ce_flow

            # weights update
            if epoch < config.epoch_pt_end:
                # backward pass
                loss.backward()
                if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))
                epoch_train_losses.append(loss.item())
            
            if epoch >= config.epoch_fm_start:
                optimizer_cfm.zero_grad()
                fm_loss.backward()
                optimizer_cfm.step()
                
                epoch_train_losses.append(fm_loss.item())
                
                if cur_step < warmup_steps:
                    lr_scale = float(cur_step) / float(warmup_steps)
                    for param_group in optimizer_cfm.param_groups:
                        param_group['lr'] = config_cfm.lr * lr_scale
                else:
                    scheduler_cfm.step()
                cur_step += 1

            
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(f"epoch {epoch} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        # if (i + 1) % config.save_every_n == 0:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating test dataset:")
        if epoch <= config.epoch_pt_end-1: 
            auc = evaluate(model,cfm_models, test_dataset, config,steps=[1])
        else:
            auc = evaluate(model,cfm_models, test_dataset, config,steps=[1,8])
        if auc > best_auc:
            best_auc = auc
            
            print(f"AUC: {best_auc} saving in {config.save_path}")
        torch.save(model.state_dict(), os.path.join(config.save_path, f'troika_{epoch}.pt'))
        torch.save(cfm_models[1].state_dict(), os.path.join(config.save_path, f'cfm_attribute_{epoch}.pt'))
        torch.save(cfm_models[2].state_dict(), os.path.join(config.save_path, f'cfm_obj_{epoch}.pt'))
    
        torch.cuda.empty_cache()



def evaluate(model, cfm_models, dataset, config, compose=False, steps=[1]):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    for step in steps:
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, composed_logits = test.predict_logits_fm_joint_troika_v2_3path(
            model, cfm_models, dataset, step, path_2=config.path_2, compose=compose
        )
        test_stats = test.test(
            dataset, evaluator, composed_logits, all_attr_gt, all_obj_gt, all_pair_gt, config
        )
        key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
        result = "".join(f"{key}  {test_stats[key]:.4f}| " for key in key_set)
        print(f"Evaluating with step {step}")
        print(result)
        best_auc = round(test_stats["AUC"], 4)
        torch.cuda.empty_cache()
    return best_auc



if __name__ == "__main__":
    
    config_cfm = DefaultConfig()
    
    config = parser.parse_args()
    model_name = copy.deepcopy(config.model_name)
    if config.yml_path:
        load_args(config.yml_path, config)
        
    if config.lr_troika != 0:
        config.lr = config.lr_troika
    print(config)
    # set the seed value
    set_seed(config.seed)
    config.model_name = model_name
    config_cfm.loss_weight =config.velocity_loss_weight
    config_cfm.logit_weight =config.logit_weight
    config_cfm.lr = config.lr_fm
    dataset_path = config.dataset_path


    if 'mit-states' in config.dataset_path:
        train_dataset = CompositionDataset(dataset_path,
                                        phase='train',
                                        split='compositional-split-natural',
                                        same_prim_sample=config.same_prim_sample)

        val_dataset = CompositionDataset(dataset_path,
                                        phase='val',
                                        split='compositional-split-natural')

        test_dataset = CompositionDataset(dataset_path,
                                        phase='test',
                                        split='compositional-split-natural')
    else:
        train_dataset = CompositionDataset_u(dataset_path,
                                        phase='train',
                                        split='compositional-split-natural',
                                        same_prim_sample=config.same_prim_sample)

        val_dataset = CompositionDataset_u(dataset_path,
                                        phase='val',
                                        split='compositional-split-natural')

        test_dataset = CompositionDataset_u(dataset_path,
                                        phase='test',
                                        split='compositional-split-natural')
    
    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = get_optimizer(model, config)

    os.makedirs(config.save_path, exist_ok=True)
    

    if not config.path_2:
        cfm_models = [DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda() for i in range(3)]
    else:
        cfm_models = [None,DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda(), DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda()]
    
    
    
    all_params = []
    for m in cfm_models:
        if m is not None:
            all_params += list(m.parameters())
    optimizer_cfm = torch.optim.AdamW(all_params, lr=config_cfm.lr, weight_decay=config_cfm.weight_decay)
    
    


    train_model(model, optimizer,cfm_models ,optimizer_cfm ,config, config_cfm,train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
