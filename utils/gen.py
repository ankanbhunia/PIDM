import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb, shutil
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
from model import UNet
from PIL import Image

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True



if __name__ == "__main__":

    init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--sample_algorithm', type=str, default='ddim') # ddpm, ddim
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cond_scale', type=float, default=2.0)
    parser.add_argument('--checkpoint_name', type=str, default="last.pt")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)

    cond_scale = args.cond_scale
    sample_algorithm = args.sample_algorithm # options: DDPM, DDIM

    _folder = args.checkpoint_name+'-'+sample_algorithm+'-'+'scale:'+str(cond_scale)

    fake_folder = 'images/'+args.exp_name+'/'+_folder

    if is_main_process():
        if not os.path.isdir( 'images/'):
            os.mkdir( 'images/')

        if not os.path.isdir( 'images/'+args.exp_name):
            os.mkdir( 'images/'+args.exp_name)

        if os.path.isdir(fake_folder):
            shutil.rmtree(fake_folder)

        os.mkdir(fake_folder)


    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)
    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path
    DataConf.data.val.batch_size = args.batch_size
    val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = True)
    val_dataset = iter(val_dataset)

    ckpt = torch.load(args.save_path+"/"+args.exp_name+'/'+args.checkpoint_name)
    
    model = get_model_conf().make_model()
    model = model.to(args.device)
    model.load_state_dict(ckpt["ema"])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)
    model.eval()

    with torch.no_grad():

        for batch_it in range(len(val_dataset)):

            batch = next(val_dataset)

            print ('batch_id-'+str(batch_it))

            img = batch['source_image'].cuda()
            target_pose = batch['target_skeleton'].cuda()

            if args.sample_algorithm == 'DDPM' or args.sample_algorithm == 'ddpm' :
                
                sample_fn = diffusion.ddim_sample_loop

                samples = sample_fn(model.module, x_cond = [img, target_pose], progress = True, cond_scale = cond_scale)

                target_output = torch.clamp(samples, -1., 1.)
                numpy_imgs = (target_output.permute(0,2,3,1).detach().cpu().numpy() + 1.0)/2.0
                fake_imgs = (255*numpy_imgs).astype(np.uint8)

                img_save_names =  batch['path'] 

                [Image.fromarray(im).save(os.path.join(fake_folder, img_save_names[idx])) for idx, im in enumerate(fake_imgs)]

            elif args.sample_algorithm == 'DDIM' or args.sample_algorithm == 'ddim' :

                nsteps = 100

                noise = torch.randn(img.shape).cuda()
                seq = range(0, 1000, 1000//nsteps)
                xs, x0_preds = ddim_steps(noise, seq, model.module, betas.cuda(), [img, target_pose], diffusion=diffusion, cond_scale=cond_scale)
                samples = xs[-1].cuda()
                
                target_output = torch.clamp(samples, -1., 1.)
                numpy_imgs = (target_output.permute(0,2,3,1).detach().cpu().numpy() + 1.0)/2.0
                fake_imgs = (255*numpy_imgs).astype(np.uint8)

                img_save_names =  batch['path']

                [Image.fromarray(im).save(os.path.join(fake_folder, img_save_names[idx])) for idx, im in enumerate(fake_imgs)]

            else:

                print ('ERROR! Sample algorithm not defined.')
