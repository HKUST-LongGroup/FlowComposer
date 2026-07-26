import datetime
import torch
import random
import numpy as np

def set_seed(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class DefaultConfig:
    def __init__(self):
        # =================================================================
        # 基本训练参数
        # =================================================================
        self.epochs = 10
        self.warmup_epochs = 1
        self.batch_size = 64
        self.lr = 2e-4
        self.lr_distangle = 1e-4
        self.weight_decay = 1e-4
        self.optimizer = 'adamw'
        self.seed = 66
        
        self.loss_weight=1
        self.logits_weights=1
        
        # =================================================================
        # 模型和数据集特定参数
        # =================================================================
        self.clip_type = 'ViT-L/16'
        self.dataset = 'mit-states'
        self.dataset_name = 'mit-states'
        self.num_shots = 16
        self.num_interpolation = 0  # 
        # if coop: load embedding at "checkpoints/coop/stanford_cars.pth"
        self.target_prompt = "a photo of {}" #coop: using coop's embedding, "a photo of {}", "This is an {}"
        self.compositional_prompt = "a photo of {} {}"
        self.blocks = 24
        self.target_scale = False # True: scale the magnitude of the target to [0.5,1.5]
        self.stepsize = 1
        self.device = 'cuda' 

        # =================================================================
        # 运行时生成的参数 (这些参数在加载时会被覆盖)
        # =================================================================
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'outputs/{self.timestamp}_{self.dataset}/'

        ######################## Project Parameters #################
        self.mixup = "none" # where to perform the mixup: None, images, features
        self.time_sample="continuous"
