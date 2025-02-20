# code partially adapted from https://github.com/prs-eth/PCAccumulation
# author: Liyuan Zhu
from cmath import nan
import os, torch
import logging
# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from toolbox.utils import validate_gradient
from datetime import datetime
import wandb
from omegaconf import OmegaConf
from toolbox.utils import partial_load
import flow_vis
from PIL import Image

class Trainer(object):
    def __init__(self, args, instances) -> None:
        self.args = args
        self.mode = args.misc.mode
        
        self.start_epoch = 1
        self.max_epoch = args.train.max_epoch
        
        self.init_time = self._get_time_str()
        self.save_dir = args.misc.ckpt_path + self.init_time
        self.device = instances['device']
        self.model = instances['model'].to(self.device)
        self.optimizer = instances['optimizer']
        self.scheduler = instances['scheduler']
        self.scheduler_freq = 1  # learning rate scheduler
        
        self.clip = args.train.grad_clip
        self.n_verbose = args.train.n_verbose
        self.iter_size = args.train.iter_size
        
        self.loader = dict()
        if self.args.misc.mode == "train":
            self.loader['train'] = instances['dataloader'][0]
            self.loader['val'] = instances['dataloader'][1]
        else:
            self.loader["all"] = instances["dataloader"]
        
        self.loss = instances['loss']
        self.best_loss = 99999
        self.best_metric = None
        
        if self.args.wandb == True:
            # os.makedirs(self.save_dir+'/wandb', mode=0o777, exist_ok=True)
            wandb.init(project='DeFlow', sync_tensorboard=True, config=OmegaConf.to_container(self.args), settings=wandb.Settings(program_relpath="main.py", disable_git=True, save_code=False))
            
        self.writer = SummaryWriter(log_dir=f'runs/{self.init_time}')
        
        self.global_step = 0
        self.eval_counter = 0
        
        # self.compute_depth = args['train']['compute_depth']
    def _get_time_str(self):
        dt = datetime.now()
        date = dt.strftime('%Y-%m-%d-%H-%M')
        return date
        
    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
    
    def _save_checkpoint(self, epoch=0, name=None):
        """
        Save current model
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            # 'best_loss': self.best_loss
        }
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        if name is None:
            filename = os.path.join(self.save_dir, f'model_e{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
            
        logging.info(f'Saving model to {filename}.')
        torch.save(state, filename)
    
    def _dump_config(self):
        # save the configuration under savedir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        OmegaConf.save(self.args, f"{self.save_dir}/config.yaml")
        
    def _load_checkpoint(self, ckpt_path):
        """
        Load checkpoint
        """
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path)
            
            partial_load(state['state_dict'], self.model)
            self.start_epoch = state['epoch'] + 1
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            # self.best_loss = state['best_loss']
        else: 
            raise ValueError(f"=> no checkpoint found at '{ckpt_path}'")
    
    ### ORIGINAL VERSION

    # def update_tensorboard_vis(self, phase, output, input):
    #     assert phase in ['train', 'val']
    #     #if phase == 'val': return
    #     if phase not in ['train', 'val']: return

    #     # flow_vis = tensor_flow2rgb(output['flow_f'][0][0])
    #     flow_color = flow_vis.flow_to_color(output['flow_f'][0][0].permute(1,2,0).detach().cpu().numpy())
    #     flow_color = np.transpose(flow_color, (2,0,1))
    #     self.writer.add_image(f'{phase}/flow_image', flow_color, global_step=self.global_step)
    #     self.writer.add_image(f'{phase}/raw_image1', input['imgT_1'][0]/2+0.5, global_step=self.global_step)
    #     self.writer.add_image(f'{phase}/raw_image2', input['imgT_2'][0]/2+0.5, global_step=self.global_step)
        
    
    #     if self.loss.compute_depth == True:
    #         self.writer.add_image(f'{phase}/depth_image', output['depth_t1'][0][0], global_step=self.global_step)
    #         self.writer.add_image(f'{phase}/static_mask', output['static_mask'][0], global_step=self.global_step)
    #         # self.writer.add_image(f'{phase}/input_depth', input['input_depth_t1'][0], global_step=self.global_step)
    #         # self.writer.add_scalar(f'{phase}/depth_loss', stats['depth_loss'], global_step=self.global_step)
                

    def save_velocity_matrices(flow, step, batch_idx, base_dir='results/'):
        """
        Saves the velocity matrices (u, v, magnitude) in separate subfolders.
    
        Parameters:
        - flow: Optical flow tensor of shape (B, 2, H, W)
        - step: Global step in training
        - batch_idx: Batch index
        - base_dir: Base directory to save velocity matrices
        """
        base_dir = 'results/'

        subfolders = {
            'u': os.path.join(base_dir, 'u'),
            'v': os.path.join(base_dir, 'v'),
            'magnitude': os.path.join(base_dir, 'magnitude')
        }
    
        for folder in subfolders.values():
            os.makedirs(folder, exist_ok=True)
    
        flow_np = flow.detach().cpu().numpy()
        u, v = flow_np[:, 0, :, :], flow_np[:, 1, :, :]
        velocity_magnitude = np.sqrt(u**2 + v**2)
    
        np.save(os.path.join(subfolders['u'], f"u_{step}_{batch_idx}.npy"), u)
        np.save(os.path.join(subfolders['v'], f"v_{step}_{batch_idx}.npy"), v)
        np.save(os.path.join(subfolders['magnitude'], f"velocity_mag_{step}_{batch_idx}.npy"), velocity_magnitude)
    
        print(f"Saved velocity matrices for step {step}, batch {batch_idx}")
    
    def update_tensorboard_vis(self, phase, output, input):
        assert phase in ['train', 'val']
    
        base_dir = 'results/'
        subfolders = {
            'raw_img1': os.path.join(base_dir, 'raw_images/imgT_1'),
            'raw_img2': os.path.join(base_dir, 'raw_images/imgT_2'),
            'optical_flow': os.path.join(base_dir, 'optical_flow'),
            'depth_maps': os.path.join(base_dir, 'depth_maps'),
            'static_masks': os.path.join(base_dir, 'static_masks')
        }
    
        for folder in subfolders.values():
            os.makedirs(folder, exist_ok=True)
    
        for i in range(input['imgT_1'].size(0)):
            raw_img1 = input['imgT_1'][i].detach().cpu().numpy().transpose(1, 2, 0)
            raw_img2 = input['imgT_2'][i].detach().cpu().numpy().transpose(1, 2, 0)
            flow_img = flow_vis.flow_to_color(output['flow_f'][i][0].permute(1, 2, 0).detach().cpu().numpy())
    
            Image.fromarray((raw_img1 * 255).astype('uint8')).save(os.path.join(subfolders['raw_img1'], f'raw_img1_{self.global_step}_{i}.png'))
            Image.fromarray((raw_img2 * 255).astype('uint8')).save(os.path.join(subfolders['raw_img2'], f'raw_img2_{self.global_step}_{i}.png'))
            Image.fromarray(flow_img.astype('uint8')).save(os.path.join(subfolders['optical_flow'], f'flow_img_{self.global_step}_{i}.png'))
    
            self.save_velocity_matrices(output['flow_f'], self.global_step, i)
    
            print("Saved raw images, optical flow, and velocity matrices")
    
        if self.loss.compute_depth:
            depth_img = np.squeeze(output['depth_t1'][0][0].detach().cpu().numpy())
            static_mask = np.squeeze(output['static_mask'][0].detach().cpu().numpy())
    
            depth_img_uint8 = (depth_img * 255).astype('uint8')
            static_mask_uint8 = (static_mask * 255).astype('uint8')
    
            Image.fromarray(depth_img_uint8).save(os.path.join(subfolders['depth_maps'], f'depth_image_{self.global_step}.png'))
            Image.fromarray(static_mask_uint8).save(os.path.join(subfolders['static_masks'], f'static_mask_{self.global_step}.png'))
            
            print("Saved depth map and static mask")


    
    def update_tensorborad_stats(self, phase, stats):
        assert phase in ['train', 'val']
        
        if phase == 'train':
            self.writer.add_scalar(f'{phase}/learning_rate', self._get_lr(), global_step=self.global_step)
            for key in stats:
                self.writer.add_scalar(f'{phase}/{key}', stats[key], global_step=self.global_step)
        else:
            for key in stats:
                self.writer.add_scalar(f'{phase}/{key}', stats[key], global_step=self.eval_counter)
        
    def update_wandb(self):
        pass
    
    def forward_one_batch(self, input_dict, phase):
        assert phase in ['train', 'val']
        # to GPU
        for key, value in input_dict.items():
            if(not isinstance(value, list)):
                input_dict[key] = value.to(self.device)
        
        if phase == 'train':
            self.model.train()
            ###############################################
            # forward pass
            output_dict = self.model(input_dict)

            loss_dict = self.loss(input_dict, output_dict)
            loss = loss_dict['total_loss'] / self.iter_size
            loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                output_dict = self.model(input_dict)
                loss_dict = self.loss(input_dict, output_dict)
        
        ##################################        
        # detach the gradients for loss terms
        for key, value in loss_dict.items():
            # if(key.find('loss')!=-1):
            loss_dict[key] = float(value.detach())

        return output_dict, loss_dict
    
    def forward_one_epoch(self, epoch, phase):
        assert phase in ['train', 'val']
        logging.info(f'{phase} epoch: {epoch}')
        
        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        self.verbose_freq = num_iter // self.n_verbose
        c_loader_iter = self.loader[phase].__iter__()
        self.optimizer.zero_grad()
        
        epoch_loss = []
        for c_iter in range(num_iter): # loop through this epoch   
            ##################################
            
            inputs = next(c_loader_iter)
            ##################################
            # forward pass
            # with torch.autograd.detect_anomaly():
            output_dict, loss_dict = self.forward_one_batch(inputs, phase)
            epoch_loss.append(loss_dict)
            ###################################################
            # run optimisation
            if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                if(gradient_valid):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                else:
                    logging.info('gradient not valid\n')
                self.optimizer.zero_grad()
                self.global_step += 1
            ################################

            
            torch.cuda.empty_cache()
            ################################
            # update to the wandb/tensorboard/logger of training stats

            ############# ORIGINAL VERSION
            # if (c_iter + 1) % self.verbose_freq == 0:
            #     curr_iter = num_iter * (epoch - 1) + c_iter
            #     total_loss = loss_dict['total_loss']
            #     flow_loss = loss_dict['flow_loss']
            #     if self.loss.compute_depth == True:
            #         depth_loss = loss_dict['depth_total_loss']
            #     else: 
            #         depth_loss = 0.0
                
            #     if phase == 'train':
            #         logging.info(f'epoch:{epoch}, steps:{self.global_step}, total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')
            #     # elif phase == 'val':
            #     #     logging.info(f'Evaluation num:{self.eval_counter}, total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')
            #     # update tensorboard
            #     self.update_tensorboard_vis(phase, output_dict, inputs)
            #     self.update_tensorborad_stats(phase, loss_dict)
                

            #### MY VERSION
            curr_iter = num_iter * (epoch - 1) + c_iter
            total_loss = loss_dict['total_loss']
            flow_loss = loss_dict['flow_loss']
            if self.loss.compute_depth == True:
                depth_loss = loss_dict['depth_total_loss']
            else: 
                depth_loss = 0.0
                
            if phase == 'train':
                logging.info(f'epoch:{epoch}, steps:{self.global_step}, total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')
            elif phase == 'val':
                logging.info(f'Evaluation num:{self.eval_counter}, total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')
            # update tensorboard
            self.update_tensorboard_vis(phase, output_dict, inputs)
            self.update_tensorborad_stats(phase, loss_dict)
                

        # epoch_loss = torch.as_tensor(epoch_loss).mean()
        epoch_metrics = self.compute_epoch_metrics(epoch_loss)
        if phase == 'val':
            logging.info(f'{phase}: total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')
        if phase == 'val':
            self.update_tensorborad_stats(phase, epoch_metrics)
            
        return epoch_metrics
            
    def run(self):
        self._dump_config()
        logging.info(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M')
        logging.info('Start training')

        # remove these two lines to go back to the original
        #checkpoint_path = '/cluster/home/fuchsja/bsc/DeFlow/checkpoints/best_model/model_best.pth'
        #self._load_checkpoint(checkpoint_path)

        for epoch in range(self.start_epoch, self.max_epoch):
            
            if epoch >= self.args.train.epoch_depth:
                self.loss.compute_depth = True
            
            self.forward_one_epoch(epoch,'train')
            self.scheduler.step()
            self._save_checkpoint(epoch)

            # evaluate after every epoch
            self.eval()
            
    def eval(self):
        logging.info('Start evaluating on validation set')
        epoch_metric = self.forward_one_epoch(self.eval_counter, 'val')
        self.eval_counter += 1
        if epoch_metric['total_loss'] < self.best_loss:
            self.best_loss = epoch_metric['total_loss']
            self._save_checkpoint(name='best')
        return epoch_metric
    
    def compute_epoch_metrics(self, epoch_losses):
        epoch_metrics = dict()
        for key in epoch_losses[0].keys():
            curr_losses = [loss[key] for loss in epoch_losses]
            epoch_metrics[key] = np.asarray(curr_losses, dtype=np.float32).mean()
        return epoch_metrics
    

    def evaluate_fully_trained_model(self, checkpoint_path):
        """Evaluate the model after loading from a checkpoint path."""
        # Load the checkpoint
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        self._load_checkpoint(checkpoint_path)
        
        # Set the model to evaluation mode
        self.model.eval()

        # Set up the validation loader
        val_loader = self.loader['all']
        
        # Initialize loss accumulator
        total_loss = 0
        num_batches = len(val_loader)

        print(f"In total there are: {num_batches} batches")

        # Iterate through validation dataset
        with torch.no_grad():
            for i, inputs in enumerate(val_loader):
                # Forward pass
                for key, value in inputs.items():
                    if(not isinstance(value, list)):
                        inputs[key] = value.to(self.device)

                output_dict = self.model(inputs)
                loss_dict = self.loss(inputs, output_dict)
                
                # Accumulate total loss
                total_loss += loss_dict['total_loss']
                flow_loss = loss_dict['flow_loss']
                if self.loss.compute_depth == True:
                    depth_loss = loss_dict['depth_total_loss']
                
                # Update TensorBoard visualization after each batch
                self.update_tensorboard_vis('val', output_dict, inputs)
                self.global_step += 1
                #print(f"Images in step:{i} saved successfully. Yay!")
                # Optional: Log the loss every 10 batches (for example)
                if i % 10 == 0:
                    logging.info(f'steps:{i}, total_loss:{total_loss:2.2f}, flow_loss:{flow_loss:2.2f}, depth_loss:{depth_loss:2.2f}')

            # Compute the average loss over the entire validation set
            logging.info(f"Validation completed.")

            # Return the final evaluation metrics (e.g., average loss)
            return 1

    # ... [Your existing code]
