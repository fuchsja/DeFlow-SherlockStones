import logging, sys
import torch
from configs import config

from models.deflow import DeFlowNet
from models.raft import RAFT
from libs.dataloader import get_dataloaders, get_full_dataloader
from libs.trainer import Trainer
from libs.losses import SceneFlow_Loss
from toolbox.utils import setup_seed
from omegaconf import DictConfig
import argparse

def instantiate_config(cfg: dict):
    """instantiate all obejct from configurations
    
    Args:
        cfg (dict): dict from yaml

    Returns:
        args (dict): dict with instances
    """
    logging.info('Instantiating configurations')
    instances = dict()
    ## device
    if cfg.misc.use_gpu == True:
        instances['device'] = torch.device('cuda')
    else:
        instances['device'] = torch.device('cpu')
        
    ## dataloader
    if cfg.misc.mode == "train":
        instances['dataloader'] = get_dataloaders(cfg)
    else: # Evaluation mode
        cfg.augmentation.enabled = False  # Disable augmentation - images should not be cropped
        instances["dataloader"] = get_full_dataloader(cfg) # Load all images for evaluation
        
    ## model, optimizer, scheduler
    if cfg.network.model == 'deflow':
        instances['model'] = DeFlowNet(cfg)
        instances['loss'] = SceneFlow_Loss(cfg)
    else: 
        raise NotImplementedError('Unknown model! Instantiation fails!')
    
    instances['optimizer'] = config.get_optimizer(cfg, instances['model'])
    instances['scheduler'] = config.get_scheduler(cfg, instances['optimizer'])
    
    
    return instances

def main(cfg_path:str):
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)])
    
    cfg = config.get_config(cfg_path)
    cfg = DictConfig(cfg)
    instances = instantiate_config(cfg)

    setup_seed(cfg.misc.seed)
    
    trainer = Trainer(cfg, instances)
    if cfg.misc.mode == "train":
        trainer.run()
    if cfg.misc.mode == "eval":
        trainer.evaluate_fully_trained_model('/cluster/home/fuchsja/bsc/DeFlow/checkpoints/best_model/model_best.pth')
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = 'configs/deflow_default.yaml', help='Path to pretrained weights')
    config_path = parser.parse_args().config_path
    main(config_path)

