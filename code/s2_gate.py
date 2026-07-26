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

import test as test
from dataset import CompositionDataset,CompositionDataset_u
from utils import *

from config import DefaultConfig, set_seed
from fm import DeepFlowMatchingNet

# import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import torch

@torch.no_grad()
def solve_ab_least_squares(vel_attr: torch.Tensor,
                           vel_obj: torch.Tensor,
                           gt_comp_vel: torch.Tensor,
                           lam: float = 1e-3):
    """
    最小二乘求解 a,b 使 a*v_attr + b*v_obj ≈ v_comp
    输入:  vel_attr, vel_obj, gt_comp_vel: [B, D]
    输出:  a, b: [B, 1]
    lam:   Tikhonov 正则（>0 更稳，0 为标准最小二乘）
    """
    assert vel_attr.shape == vel_obj.shape == gt_comp_vel.shape
    B, D = vel_attr.shape
    device = vel_attr.device
    dtype  = vel_attr.dtype

    # 组装 A: [B, D, 2]
    A = torch.stack([vel_attr, vel_obj], dim=2)  # 每个样本的设计矩阵

    # 正规方程: (A^T A + lam I) c = A^T y
    AT = A.transpose(1, 2)                       # [B, 2, D]
    G  = AT @ A                                   # [B, 2, 2]
    if lam is not None and lam > 0:
        I = torch.eye(2, dtype=dtype, device=device).unsqueeze(0)  # [1,2,2]
        G = G + lam * I

    y  = AT @ gt_comp_vel.unsqueeze(-1)           # [B, 2, 1]

    # 解线性方程组，优于直接求逆
    coeff = torch.linalg.solve(G, y).squeeze(-1)  # [B, 2]
    a = coeff[:, 0:1]
    b = coeff[:, 1:2]
    return a, b


class VelocityGater(nn.Module):
    """
    学习 a,b (以及可选的 gamma)，使 a*v_attr + b*v_obj ≈ g_c
    - sum_to_one=True: 通过softmax约束 a+b=1, a,b>=0
    - with_scale=True: 额外学习 gamma≥0 调整合成速度的模长
    """
    def __init__(self, dim: int, hidden: int = 256,
                 sum_to_one: bool = True, with_scale: bool = True, dropout: float = 0.0):
        super().__init__()
        self.sum_to_one = sum_to_one
        self.with_scale = with_scale

        in_dim = 2 * dim
        out_dim = 2 + (1 if with_scale else 0)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim)
        )
        # xavier 初始化更稳
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, v_attr: torch.Tensor, v_obj: torch.Tensor, tau: float = 1.0):
        """
        v_attr, v_obj: (N, D)
        返回: a (N,1), b (N,1), gamma (N,1 or None), v_mix (N,D)
        """
        x = torch.cat([v_attr, v_obj], dim=-1)   # (N, 2D)
        logits = self.net(x)                     # (N, 2 [+1])

        if self.with_scale:
            logits_ab, logit_gamma = logits[:, :2], logits[:, 2:]
        else:
            logits_ab, logit_gamma = logits, None

        if self.sum_to_one:
            w = F.softmax(logits_ab / tau, dim=-1)         # a+b=1, >=0
            a, b = w[:, :1], w[:, 1:]
        else:
            # 非约束版本：正权（softplus），不强制和为1
            a = F.softplus(logits_ab[:, :1])
            b = F.softplus(logits_ab[:, 1:])

        if self.with_scale:
            gamma = F.softplus(logit_gamma)                 # ≥0
        else:
            gamma = None

        v_mix = a * v_attr + b * v_obj                      # (N, D)
        if gamma is not None:
            v_mix = v_mix * gamma                           # (N, D)

        return a, b, gamma, v_mix


def train_model(model,cfm_models ,gater,optimizer_gate, config,config_cfm, train_dataset, val_dataset, test_dataset):
    def seed_worker(worker_id):
        worker_seed = config.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(config.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    best_auc = 0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gater.train()
    for m in cfm_models:
        if m is not None:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    # scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    
    num_training_steps,warmup_steps = config_cfm.epochs*len(train_dataloader), config_cfm.warmup_epochs*len(train_dataloader)
    scheduler_cfm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=num_training_steps-warmup_steps, eta_min=0)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []
    cur_step = 0
    evaluate(model,cfm_models,gater, test_dataset, config,steps=[1],compose=False)
    for epoch in range(config.epoch_start, config.epochs):
        
            
        # evaluate(model,cfm_models, test_dataset, config,compose=True)
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (epoch + 1)
        )

        epoch_train_losses = []
        # evaluate(model,cfm_models,gater, test_dataset, config,steps=[1],compose=True)
        for bid, batch in enumerate(train_dataloader):
            loss = 0
            fm_loss = 0
            gater_loss = 0
            predict = list()
            image,att_label,obj_label ,pair = batch
            pair,att_label,obj_label  = pair.cuda(), att_label.cuda(), obj_label.cuda()
            image = image.cuda()
            labels_list = [pair.cuda(), att_label.cuda(), obj_label.cuda()]
            
            # image_features = model.encode_image(image)
            # image_features = image_features / image_features.norm(
            #         dim=-1, keepdim=True
            #     )
           
            with torch.no_grad():
                text_embeddings, normalized_img_features = model(batch, train_pairs)
            # print("text_embeddings ",text_embeddings.__len__())
            pair_text_embeddings, attr_text_embeddings, obj_text_embeddings = text_embeddings
            
            b = batch[0].size()[0]
            norm_pair_text_features = pair_text_embeddings[torch.arange(b), pair, :].cuda()
            norm_attr_text_features = attr_text_embeddings[torch.arange(b), att_label, :].cuda()
            norm_obj_text_features = obj_text_embeddings[torch.arange(b), obj_label, :].cuda()
            
            norm_text_features = [norm_pair_text_features, norm_attr_text_features, norm_obj_text_features]
  
            
            t_shared = torch.rand(image.size(0), 1, device='cuda')
            t_list = [t_shared, t_shared, t_shared]
            
            vel_obj_list = [None,None]
            vel_attr_list = [None,None]
            vel_comp_list = [None,None]
            
            if 1:
                for i in range(3):
                    if config.path_2 and i == 0:
                        continue
                    gt_v = norm_text_features[i] - normalized_img_features[i]
                    interpolated_features = (1 - t_list[i]) * normalized_img_features[i].detach() + t_list[i] * norm_text_features[i].detach()
                    with torch.no_grad():
                        pred_vel = cfm_models[i](interpolated_features, t_list[i]) 
                    if i == 1:
                        vel_attr_list[0] = pred_vel
                    elif i == 2:
                        vel_obj_list[1] = pred_vel
                    # velocity_list.append(pred_vel)
                  
                    for j in range(1,3):
                        gt_v = norm_text_features[i] - normalized_img_features[i-j]
                        interpolated_features = (1 - t_list[i]) * normalized_img_features[i-j].detach() + t_list[i] * norm_text_features[i].detach()
                        with torch.no_grad():
                            pred_vel = cfm_models[i](interpolated_features, t_list[i]) 
                        if i-j ==1:
                            vel_attr_list[1] = pred_vel
                        elif i-j ==2 or i-j==-1:
                            vel_obj_list[0] = pred_vel
                        elif i-j ==0:
                            vel_comp_list[i-1] = pred_vel
                        # velocity_list.append(pred_vel)
                        
                    
                
                        
                a, b_, gamma, v_mix = gater(vel_attr_list[0].detach(), vel_attr_list[1].detach())
                gt_a, gt_b = solve_ab_least_squares(vel_attr_list[0].detach(), vel_attr_list[1].detach(), (norm_text_features[0].detach() - normalized_img_features[1].detach() ), lam=0.01)
                gater_loss += torch.sum( (a - gt_a)**2,dim=1).mean()
                gater_loss += torch.sum( (b_ - gt_b)**2,dim=1).mean()
                # gater_loss += torch.sum( (v_mix - (norm_text_features[0].detach() - normalized_img_features[1].detach() ) )**2,dim=1).mean()
                v_mix_n = v_mix / (v_mix.norm(dim=1, keepdim=True) + 1e-8)
                g_comp_n = (norm_text_features[0].detach() - normalized_img_features[1].detach() ) / ((norm_text_features[0].detach() - normalized_img_features[1].detach() ).norm(dim=1, keepdim=True) + 1e-8)
                gater_loss += 0.1*(1.0 - (v_mix_n * g_comp_n).sum(dim=1)).mean()
                
                a, b_, gamma, v_mix = gater(vel_obj_list[0].detach(), vel_obj_list[1].detach())
                gt_a, gt_b = solve_ab_least_squares(vel_obj_list[0].detach(), vel_obj_list[1].detach(), (norm_text_features[0].detach() - normalized_img_features[2].detach() ), lam=0.01)
                gater_loss += torch.sum( (a - gt_a)**2,dim=1).mean()
                gater_loss += torch.sum( (b_ - gt_b)**2,dim=1).mean()
                # gater_loss += torch.sum( (v_mix - (norm_text_features[0].detach() - normalized_img_features[2].detach() ) )**2,dim=1).mean()
                v_mix_n = v_mix / (v_mix.norm(dim=1, keepdim=True) + 1e-8)
                g_comp_n = (norm_text_features[0].detach() - normalized_img_features[2].detach() ) / ((norm_text_features[0].detach() - normalized_img_features[2].detach() ).norm(dim=1, keepdim=True) + 1e-8)
                gater_loss += 0.1*(1.0 - (v_mix_n * g_comp_n).sum(dim=1)).mean()
                
                a, b_, gamma, v_mix = gater(vel_comp_list[0].detach(), vel_comp_list[1].detach())
                gt_a, gt_b = solve_ab_least_squares(vel_comp_list[0].detach(), vel_comp_list[1].detach(), (norm_text_features[0].detach() - normalized_img_features[0].detach() ), lam=0.01)
                gater_loss += torch.sum( (a - gt_a)**2,dim=1).mean()
                gater_loss += torch.sum( (b_ - gt_b)**2,dim=1).mean()
                # gater_loss += torch.sum( (v_mix - (norm_text_features[0].detach() - normalized_img_features[0].detach() ) )**2,dim=1).mean()
                v_mix_n = v_mix / (v_mix.norm(dim=1, keepdim=True) + 1e-8)
                g_comp_n = (norm_text_features[0].detach() - normalized_img_features[0].detach() ) / ((norm_text_features[0].detach() - normalized_img_features[0].detach() ).norm(dim=1, keepdim=True) + 1e-8)
                gater_loss += 0.1*(1.0 - (v_mix_n * g_comp_n).sum(dim=1)).mean()

            
            
               
                
            optimizer_gate.zero_grad()
            gater_loss.backward()
            optimizer_gate.step()
                
                
            epoch_train_losses.append(gater_loss.item())
                
            if cur_step < warmup_steps:
                lr_scale = float(cur_step) / float(warmup_steps)
                for param_group in optimizer_gate.param_groups:
                    param_group['lr'] = config.lr_gate * lr_scale
            else:
                scheduler_cfm.step()
            cur_step += 1

            
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(f"epoch {epoch+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        # if (i + 1) % config.save_every_n == 0:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating test dataset:")
        
        auc = evaluate(model,cfm_models,gater, test_dataset, config,steps=[1,8],compose=True)
        evaluate(model,cfm_models,gater, test_dataset, config,steps=[1],compose=True, end_points=0.1)
        # auc = evaluate(model,cfm_models,gater, test_dataset, config,steps=[1],compose=False)
  
        torch.save(gater.state_dict(), os.path.join(config.save_path, f'gate_{epoch}.pth'))

        torch.cuda.empty_cache()



def evaluate(model, cfm_models, gater, dataset, config, compose=False, steps=[1], end_points=1.0):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    for step in steps:
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, composed_logits = test.predict_logits_fm_joint_troika_v2_3path_wGate_v2(
            model, cfm_models, gater, dataset, step, path_2=config.path_2,
            compose=compose, end_points=end_points
        )
        test_stats = test.test(
            dataset, evaluator, composed_logits, all_attr_gt, all_obj_gt, all_pair_gt, config
        )
        key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
        result = "".join(f"{key}  {test_stats[key]:.4f}| " for key in key_set)
        print(f"Evaluating with step {step} compose {compose} end_points {end_points}")
        print(result)
        best_auc = round(test_stats["AUC"], 4)
        torch.cuda.empty_cache()
        print("Best AUC: ", best_auc)
    return best_auc



if __name__ == "__main__":
    
    config_cfm = DefaultConfig()
    
    config = parser.parse_args()
    model_name = copy.deepcopy(config.model_name)
    if config.yml_path:
        load_args(config.yml_path, config)
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
    model.load_state_dict(torch.load(config.save_model_path, map_location="cuda"))

    os.makedirs(config.save_path, exist_ok=True)
    
    # clip_model, transform = clip.load("ViT-L/14", device='cpu')
    # dim = clip_model.visual.output_dim
    # del clip_model
    if not config.path_2:
        cfm_models = [DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda() for i in range(3)]
    else:
        cfm_models = [None,DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda(), DeepFlowMatchingNet(in_channels=768,model_channels=768, out_channels=768,num_res_blocks=config_cfm.blocks).cuda()]
    gater = VelocityGater(dim=768, hidden=256, sum_to_one=config.sum_to_one, with_scale=True).cuda()
    
    cfm_models[1].load_state_dict(torch.load(config.save_cfm1_path, map_location="cuda"))
    cfm_models[2].load_state_dict(torch.load(config.save_cfm2_path, map_location="cuda"))
    
    
    
    optimizer_gate = torch.optim.AdamW(
        [
            {"params": gater.parameters(),    "lr": config.lr_gate, "weight_decay": config_cfm.weight_decay},
        ],
        weight_decay=config_cfm.weight_decay,
    )
    


    train_model(model,cfm_models ,gater,optimizer_gate ,config, config_cfm,train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
