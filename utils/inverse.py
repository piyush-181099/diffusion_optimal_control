import os
import yaml
import argparse

import torch
import numpy as np
import lpips
from torchvision import transforms
from functools import partial
from PIL import Image

from envs.diffusion import DPSDiffusion

from optimizers.mfddp import MFDDP
from optimizers.adjoint_sensitivity import AdjointSensitivity
from optimizers.bayes import Bayes

from utils.unet import create_model
from utils.gaussian_diffusion import create_sampler
from utils.measurements import get_noise, get_operator
from utils.dataloader import get_dataset, get_dataloader
from utils.img_utils import mask_generator
from utils.condition_methods import get_conditioning_method

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def tuple_from_str(s):
    return tuple([int(x) for x in s.strip('()').split(', ')])

def get_state_cost(operator, shape):
    def state_cost(x, target, mask=None):
        downsampled = operator.forward(x.reshape(-1, *shape), mask=mask).reshape(len(x), -1)
        assert downsampled.shape == target.shape, f"downsampled.shape: {downsampled.shape}, target.shape: {target.shape}"
        return (downsampled - target).norm()
    return state_cost

def process(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        assert isinstance(x, np.ndarray)
    x = x.clip(-1, 1)
    x = x.transpose(1, 2, 0)
    return ((x + 1) * 127.5).astype(np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description='Compute DDP on MNIST.')
    parser.add_argument('--task', help='Task to benchmark: box_inpainting / random_inpainting / motion_deblur / gaussian_deblur / super_resolution / phase_retrieval', default='super_resolution', type=str)
    parser.add_argument('--dataset', help='Dataset: ffhq / imagenet', default='ffhq', type=str)
    parser.add_argument('--num_iters', help='Number of DDP iterations', default=50, type=int)
    parser.add_argument('--num_steps', help='Number of steps of the diffusion process', type=int, default=20)
    parser.add_argument('--seed', help='Random seed, set to -1 if no seed', type=int, default=42)
    parser.add_argument('--outdir', help='Output directory', default='out/', type=str)
    parser.add_argument('--data_root', help='Root directory for data', default='', type=str)
    parser.add_argument('--model_root', help='Root directory for model weights', default='', type=str)
    
    args = parser.parse_args()
    return args

def create_dirs(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir(os.path.join(outdir, 'sample/')):
        os.mkdir(os.path.join(outdir, 'sample/'))
    if not os.path.isdir(os.path.join(outdir, 'ref/')):
        os.mkdir(os.path.join(outdir, 'ref/'))
    if not os.path.isdir(os.path.join(outdir, 'measurement/')):
        os.mkdir(os.path.join(outdir, 'measurement/'))
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class Experiment:
    def __init__(self, args=None, device='cuda'):
        if args is None:
            args = parse_args()

        # maybe set seed
        if args.seed != -1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        # load and process configs
        task_config, data_config, model_config, diffusion_config = self.load_configs(args)

        # load data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.loader = get_dataloader(
            get_dataset(**data_config, transforms=transform), 
            batch_size=1, num_workers=0, train=False, shuffle=args.seed != -1)

        # initialize variables
        self.device = device
        self.outdir = args.outdir
        self.weight_scale = float(task_config['parameters']['weight_scale'])
        self.lr = float(task_config['parameters']['lr'])
        self.shape = tuple_from_str(model_config['shape'])
        self.num_iterations = args.num_iters
        self.alg_class = partial(MFDDP, seed=args.seed, lr=self.lr)
        self.model = create_model(**model_config).to(device).eval()
        self.noiser = get_noise(**task_config['measurement']['noise'])
        self.lpips_model = lpips.LPIPS(net='alex', verbose=True).to(device)
        self.operator = get_operator(device=device, **task_config['measurement']['operator'])
        self.state_cost = get_state_cost(self.operator, self.shape)

        self.sampler = create_sampler(
          **diffusion_config,
          timestep_respacing=args.num_steps)
        self.env_class = partial(
          DPSDiffusion,
          model=self.model, 
          sampler=self.sampler,
          shape=self.shape, 
          num_steps=args.num_steps,
          device=self.device)
        
        if task_config['measurement']['operator']['name'] == 'inpainting':
            self.mask_gen = mask_generator(**task_config['measurement']['mask_opt'])
        else:
            self.mask_gen = None
        
        create_dirs(self.outdir)

    def load_configs(self, args):
        task_config = load_yaml(f'inverse_configs/{args.task}_config.yaml')
        
        data_config = task_config['data']
        if args.data_root:
            data_config['root'] = os.path.join(args.data_root, args.dataset)
        data_config['name'] = args.dataset
        
        model_config = load_yaml(f'inverse_configs/{args.dataset}_model_config.yaml')
        if args.model_root:
            model_config['model_dir'] = args.model_root
        model_config['model_path'] = os.path.join(model_config['model_dir'], model_config['model_name'])

        diffusion_config = load_yaml('inverse_configs/diffusion_config.yaml')
      
        return task_config, data_config, model_config, diffusion_config

    def lpips_fn(self, x, y):
        x = x.reshape(-1, *self.shape).to(self.device)
        y = y.reshape(-1, *self.shape).to(self.device)
        return self.lpips_model(x, y)

    def get_solver(self, env):
        return self.alg_class(env)

    def get_env(self, *args, mask=None, **kwargs):
        return self.env_class(
          *args, 
          weight_scale=self.weight_scale, 
          state_cost=lambda x, y: self.state_cost(x, y, mask=mask),
          **kwargs)

    def get_mask(self, reference):
        if self.mask_gen is not None:
            mask = self.mask_gen(reference)
            mask = mask[:, 0:1, :, :]
        else:
            mask = None

        return mask

    def compute_metrics(self, reference, sample, states=None):
        f = lambda x: x * .5 + .5
        lpips = self.lpips_fn(f(reference), f(sample))
        
        if states is not None:
            tiled_ref_img = reference[None].tile([len(states),] + [1] * len(reference.shape))
            lpips_over_iterations = self.lpips_fn(f(tiled_ref_img), f(torch.stack(states))).squeeze()
            lpips_over_iterations = lpips_over_iterations.reshape(len(states), len(reference))
        else:
            lpips_over_iterations = None

        result = {
          'lpips': lpips.detach().cpu(),
          'lpips_over_iterations': lpips_over_iterations.detach().cpu(),
        }
        return result

    def save_images(self, idx, reference, sample, measurement):
        im_hat = Image.fromarray(process(sample[0]))
        im_hat.save(os.path.join(os.path.join(self.outdir, 'sample/'), f"{idx}.jpg"))

        im = Image.fromarray(process(reference[0]))
        im.save(os.path.join(os.path.join(self.outdir, 'ref/'), f"{idx}.jpg"))

        y = Image.fromarray(process(measurement[0]))
        y.save(os.path.join(os.path.join(self.outdir, 'measurement/'), f"{idx}.jpg"))
    