from copy import deepcopy
import os
import pickle
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from diffusion.network import get_resnet, replace_bn_with_gn, ConditionalUnet1D, DiffusionModel
from diffusion.clip_pretraining import modified_resnet18
from diffusion.dataset import load_data 
#from utils import get_norm_stats

from datetime import datetime

# from diffusion.visualization import debug, visualize
from diffusion.visualize_waypts import predict_diff_actions

from diffusion.train_args import CKPT_DIR, SIM_TASK_CONFIGS, DEVICE_STR, START_TIME

def main(args):
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    ckpt_dir = args['ckpt_dir']
    enc_type = args['enc_type']
    task_name = args['task_name']

    if enc_type not in ['clip','resnet18']:
        raise ValueError("only 'clip' or 'resnet18' accepted as encoder types")

    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    split_ratio = 0.8
    obs_horizon = 1

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'lr': args['lr'],
        'seed': args['seed'],
        'policy_config': {'num_queries': args['chunk_size'], }, #@Chris: ensure the chunk size is 3 and not 100
        # 'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,

        # for unveiler
        'split_ratio': split_ratio
    }

    if enc_type == 'clip':
        # load modified CLIP pretrained resnet 
        # image_weights = torch.load(IMAGE_WEIGHTS_PATH)

        image_encoders = []
        for i in range(len(camera_names)): #subtract one to account for gelsight
            image_encoders += [modified_resnet18()]
            # image_encoders[i].load_state_dict(image_weights)
            image_encoders[i] = nn.Sequential(image_encoders[i],nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten())

        # modified_resnet18 uses groupnorm instead of batch already

    elif enc_type == 'resnet18':
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        image_encoders = []
        for i in range(len(camera_names)):
            image_encoders += [get_resnet('resnet18')]
            image_encoders[i] = replace_bn_with_gn(image_encoders[i])


    # Encoders have output dim of 512
    vision_feature_dim = 512
    # agent_pos is 4 dimensional
    lowdim_obs_dim = 4
    # observation feature has [  ] dims in total per step
    #7 cameras including gelsight
    obs_dim = vision_feature_dim * len(camera_names) + lowdim_obs_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=args['chunk_size'],
        global_cond_dim=obs_dim*obs_horizon
    )


    train_dataloader, val_dataloader, stats, _ = load_data(config, dataset_dir, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    nets = nn.ModuleDict({
        'noise_pred_net': noise_pred_net
    })

    for i, cam_name in enumerate(camera_names):
        nets[f"{cam_name}_encoder"] = image_encoders[i]

    train(config, nets, train_dataloader, val_dataloader, enc_type)
    
def _save_ckpt(start_time:datetime,epoch,enc_type,
               nets,train_losses,val_losses,test=False):
    
    ckpt_dir=CKPT_DIR
    # noise_pred_net
    model_checkpoint = {}
    for i in nets.keys():
        model_checkpoint[i] = nets[i]
        
    now = datetime.now()
    now_time = now.strftime("%H-%M-%S_%Y-%m-%d")
    today = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    ckpt_dir = ckpt_dir+today+'_'+enc_type
    os.makedirs(ckpt_dir,exist_ok=True)

    save_dir = os.path.join(ckpt_dir,f'{enc_type}_epoch{epoch}_{now_time}')
    torch.save(model_checkpoint, save_dir)
    
    np.save(
        os.path.join(ckpt_dir,f'{enc_type}_trainlosses_{today}.npy'),
        train_losses)
    np.save(
        os.path.join(ckpt_dir,f'{enc_type}_vallosses_{today}.npy'),
        val_losses)
    

    ##test
    if test:
        model_dict = torch.load(save_dir)

        model1 = model_dict['gelsight_encoder']
        model2 = nets['gelsight_encoder']

        bool = True
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                bool = False

        print("noise model is same as saved model:", bool)

        model_dict = torch.load('checkpoints/2024-02-25/clip_epoch0_2024-02-25_20:31:31')
        model1 = model_dict['noise_pred_net']

        bool = True
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                bool = False

        print("noise model is same as random model:", bool)
        input("press any key to continue")

def train(config, nets:nn.ModuleDict, train_dataloader, val_dataloader, enc_type, device=torch.device(DEVICE_STR)):
    
    # debug.print=True
    # debug.plot=True 
    # debug.dataset=('validation')
    # today = START_TIME.strftime("%Y-%m-%d_%H-%M-%S")
    # debugdir = CKPT_DIR+today+'_plots'+'_'+enc_type
    # debug.visualizations_dir=debugdir

    #TODO:
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights

    # ema = EMAModel(
    #     parameters=nets.parameters(),
    #     power=0.75)

    nets.to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    model = DiffusionModel(nets, config['camera_names'], device, noise_scheduler).to(device)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * config['num_epochs']
    )
    

    with tqdm(range(config['num_epochs']), desc='Epoch') as tglobal:
        # epoch loop
        train_losses = list()
        val_losses = list()
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            nets.train()
            with tqdm(train_dataloader, desc='Train Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    
                    noise_pred, noise = model(nbatch)
                    # print("noise_pred, noise", noise_pred.shape, noise.shape)
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # TODO:
                    # update Exponential Moving Average of the model weights
                    #ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            
            train_losses.append(np.mean(epoch_loss))
                
            nets.eval()
            val_loss=list()
            min_val_loss = np.inf
            best_ckpt_info = None

            with tqdm(val_dataloader, desc='Val_Batch', leave=False) as tepoch:
                with torch.no_grad():
                    for i, nbatch in enumerate(tepoch):
                        noise_pred, noise = model(nbatch)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        loss_cpu = loss.item()
                        val_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                        if loss_cpu < min_val_loss:
                            min_val_loss = loss_cpu
                            best_ckpt_info = (epoch_idx, min_val_loss, deepcopy(model.state_dict()))

                        # #save plot of first batch
                        # if i == 0:
                        #     # debug.epoch = epoch_idx
                            
                        #     mdict = dict()
                        #     for i in nets.keys():
                        #         mdict[i] = nets[i]

                        #     all_images,qpos,preds,gt= predict_diff_actions(nbatch,
                        #         val_dataloader.dataset.action_qpos_normalize,
                        #         mdict,
                        #         camera_names,device
                        #     )
                        #     print('all_images',len(all_images),'0:',all_images[0].shape)
                        #     print('qpos',qpos.shape)
                        #     print('preds', preds.shape)
                        #     print('gt',gt.shape)
                        #     visualize(all_images,qpos,preds,gt)
            
            val_losses.append(np.mean(val_loss))
            

            if epoch_idx % 200 == 0: 
                # _save_ckpt(START_TIME,epoch_idx,enc_type,nets,train_losses,val_losses)

                ckpt_path = os.path.join(config['ckpt_dir'], f"policy_epoch_{epoch_idx}_seed_{config['seed']}.ckpt")
                torch.save(model.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(config['ckpt_dir'], f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

    print(f"Training finished:\nSeed {config['seed']}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
   
    # _save_ckpt(START_TIME,num_epochs,enc_type,nets,train_losses,val_losses) #final save