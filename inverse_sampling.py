import os, sys

data_root = './data/'
model_root = './models/'

import argparse
import time
import pickle
import yaml
from PIL import Image
import matplotlib.pyplot as plt

import torch
import numpy as np
import lpips
from torchvision import transforms
from torchvision.utils import make_grid
from functools import partial
from skimage.metrics import structural_similarity as ssim

from envs.diffusion import DPSDiffusion

from optimizers.lmmfddp import MFDDP
from optimizers.adjoint_sensitivity import AdjointSensitivity
from optimizers.bayes import Bayes
# from optimizers.fbps import ForwardBackwardPosteriorSampler
from optimizers.ddnm import DDNM

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

# The purpose of this function is to generate a terminal-image(s) at t=eps given an initial-image(s) at t=1.0 
# and a sequence of controls.
def get_img(env, init_state, actions=None):
    with torch.no_grad():
        if actions is None:
            null_action = torch.zeros(size=(len(init_state), env.control_dim)).to(init_state.device)
            actions = [null_action] * env.num_steps
            
        states = simulate(env, init_state, actions)

        return states[-1].reshape(-1, *shape)
    
def simulate(env, init_state, actions):
    with torch.no_grad():
        states = [init_state]
        
        for t in range(env.num_steps - 1):
            states.append(env.step(env.timesteps[t], states[t], actions[t][None]).detach())

        return states

parser = argparse.ArgumentParser(description='Compute DDP on MNIST.')
parser.add_argument('-i','--iters', help='Number of DDP iterations', default=50, type=int)
parser.add_argument('-s','--diffusion_steps', help='Number of diffusion iterations', default=50, type=int)
parser.add_argument('-b','--batch_size', help='Model batch size', default=1, type=int)
parser.add_argument('-a','--algorithm', help='Algorithm to use: ddp / adjoint_sensitivity / bayes', default='adjoint_sensitivity', type=str)
parser.add_argument('-t','--task', help='Task to benchmark: box_inpainting / random_inpainting / motion_deblur / gaussian_deblur / super_resolution / phase_retrieval', default='super_resolution', type=str)
parser.add_argument('-o','--outdir', help='Output directory', default='out/', type=str)
parser.add_argument('-d','--deterministic', help='Using DDIM instead of DDPM', type=int, default=0)
parser.add_argument('-ds','--dataset', help='Dataset: ffhq / imagenet', default='ffhq', type=str)
parser.add_argument('-si','--start_index', help='Data index to start from', type=int, default=-1)
parser.add_argument('-ws','--warm_start', help='Initializing with Bayes', type=int, default=0)
parser.add_argument('-se','--seed', help='Seed', type=int, default=42)

args = parser.parse_args()

if args.seed == -1:
    args.seed = np.random.randint(1000)

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

device = 'cuda'
control_type = 'pixel' # 'pixel' / 'pixel_after' / 'h'

if args.task == 'super_resolution':
    task_config = load_yaml('inverse_configs/super_resolution_config.yaml')
elif args.task == 'box_inpainting':
    task_config = load_yaml('inverse_configs/box_inpainting_config.yaml')
elif args.task == 'random_inpainting':
    task_config = load_yaml('inverse_configs/inpainting_config.yaml')
elif args.task == 'motion_deblur':
    task_config = load_yaml('inverse_configs/motion_deblur_config.yaml')
elif args.task == 'gaussian_deblur':
    task_config = load_yaml('inverse_configs/gaussian_deblur_config.yaml')
elif args.task == 'phase_retrieval':
    task_config = load_yaml('inverse_configs/phase_retrieval_config.yaml')

measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])

# Prepare dataloader
data_config = task_config['data']
if args.dataset == 'ffhq':
    data_config['root'] = os.path.join(data_root, 'ffhq')
    model_config = load_yaml('inverse_configs/model_config.yaml')
    model_config['model_path'] = os.path.join(model_root, 'ffhq_10m.pt')
elif args.dataset == 'imagenet':
    data_config['root'] = os.path.join(data_root, 'imagenet256')
    model_config = load_yaml('inverse_configs/imagenet_model_config.yaml')
    model_config['model_path'] = os.path.join(model_root, 'imagenet256.pt')
else:
    raise NotImplementedError()
    
np.random.seed(args.seed)
torch.manual_seed(args.seed)
shuffle = args.start_index < 0
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
loader = get_dataloader(
  get_dataset(**data_config, transforms=transform), batch_size=args.batch_size, 
    num_workers=0, train=False, shuffle=shuffle)

form = lambda x: x.reshape(-1, *shape).cpu().numpy()
lpips_model = lpips.LPIPS(net='alex', verbose=True).to(device)

def form(x):
    x = (x.clip(0, 1) * 255.).int()
    assert torch.all(x < 256) and torch.all(x >= 0)
    return x.reshape(-1, *shape).cpu().numpy()

def lpips_fn(x, y):
    x = x.reshape(-1, *shape).to(device)
    y = y.reshape(-1, *shape).to(device)
    return lpips_model(x, y).detach().cpu().numpy()

def ssim_fn(x, y):
    return np.array([ssim(a, b, channel_axis=0, data_range=255.) for a, b in zip(form(x), form(y))])

def nmse_fn(x, y):
    x, y = form(x), form(y)
    
    mse = np.mean((x - y) ** 2, axis=(1, 2, 3)) / np.mean(x ** 2, axis=(1, 2, 3))
    return mse

def psnr_fn(x, y):
    x, y = form(x), form(y)
    
    mse = np.mean((x - y) ** 2, axis=(1, 2, 3))
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse + 1e-7)) 
    return psnr
    
def state_cost(x, target, mask=None):
    downsampled = operator.forward(x.reshape(-1, *shape), mask=mask).reshape(len(x), -1)
    assert downsampled.shape == target.shape, f"downsampled.shape: {downsampled.shape}, target.shape: {target.shape}"
    return (downsampled - target).norm()

torch2img = lambda x, **kwargs: make_grid(x, normalize=True, **kwargs).permute(1, 2, 0).cpu()

diffusion_config = load_yaml('inverse_configs/diffusion_config.yaml')
diffusion_config['sampler'] = 'ddpm'

diffusion_config['timestep_respacing'] = args.diffusion_steps
diffusion_config['model_var_type'] = 'fixed_small'
device = 'cuda'

model_config['use_fp16'] = True
model = create_model(**model_config)
model.convert_to_fp16()
model = model.to(device)
model.eval()
sampler = create_sampler(**diffusion_config)

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning

if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(
       **measure_config['mask_opt']
    )

shape = (3, 256, 256)
env_class = partial(DPSDiffusion,
                    model=model, 
                    sampler=sampler,
                    shape=shape, 
                    num_steps=args.diffusion_steps,
                    device=device)

use_running_state_cost = False
if args.algorithm == 'ddp':
    num_iterations = args.iters
    kwargs = {
        'k_jacobian': 0,
        'lr_mode': 'k',
        'k_hessian': 0,
        'verbose': 1,
        'chunk_size': 12, # i.e., 'batch_size' of the vmap computations
        # 'eps': 1e-3,
        'eps': 1,
        'print_every': 10,
        'use_running_state_cost': use_running_state_cost,
        'success_multiplier': 1.,
        'debugging_mode': False,
        'rrf_iters': 2,
        'lr_identity': False,
        'seed': args.seed,
        'k_mf': 32,
        'lr': 1e-3
    }
    alg_class = MFDDP
elif args.algorithm == 'adjoint_sensitivity':
    num_iterations = args.iters
    kwargs = {
        'use_running_state_cost': use_running_state_cost,
        'print_every': 10,
        'lr': 1e-3,
    }
    alg_class = AdjointSensitivity
elif args.algorithm == 'bayes':
    num_iterations = args.iters
    kwargs = {
        'classifier_weight': 1.,
        'print_every': 1,
        'use_running_state_cost': use_running_state_cost,
    }
    alg_class = Bayes
elif args.algorithm == 'fbps':
    num_iterations = args.iters
    kwargs = {
        'classifier_weight': 1.,
        'print_every': 25,
        'lr': 5e-4,
        'use_running_state_cost': use_running_state_cost,
    }
    alg_class = ForwardBackwardPosteriorSampler
elif args.algorithm == 'ddnm':
    num_iterations = args.iters
    kwargs = {
        'print_every': 25,
        'operator': operator,
        'classifier_weight': 1,
    }
    alg_class = DDNM

weight_scale = 1e-4
# weight_scale = 1e-3
# weight_scale = 1.

process = lambda x: ((x.clip(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
scale = lambda x: x / 2 + .5
# scale = lambda x: x

ref_root = os.path.join(args.outdir, 'ref/')
if not os.path.isdir(ref_root):
    os.mkdir(ref_root)
save_root = os.path.join(args.outdir, 'jpg/')
if not os.path.isdir(save_root):
    os.mkdir(save_root)
save_idx = 0

ref_imgs = []
y_ns = []
samples = []
metrics_over_times = []
actions_vals = []

def get_metrics(ref_img, states):
    tiled_ref_img = ref_img[None].tile([len(states),] + [1] * len(ref_img.shape))
    states = torch.stack(states)
    metrics_over_time = []
    
    for fn in [lpips_fn, psnr_fn, ssim_fn, nmse_fn]:
        metric_over_time = fn(scale(tiled_ref_img), scale(states)).squeeze()
        metric_over_time = metric_over_time.reshape(len(states), len(ref_img))
        metrics_over_time.append(metric_over_time)
        
    metrics_over_time = np.concatenate(metrics_over_time, axis=-1)
    return metrics_over_time

def print_metrics(metrics_over_times):
    metrics_over_times = np.stack(metrics_over_times)
    string = f"""
    LPIPS: {metrics_over_times[-1, -1, 0]} -- {metrics_over_times[:, -1, 0].mean()}
    PSNR: {metrics_over_times[-1, -1, 1]} -- {metrics_over_times[:, -1, 1].mean()}
    SSIM: {metrics_over_times[-1, -1, 2]} -- {metrics_over_times[:, -1, 2].mean()}
    NMSE: {metrics_over_times[-1, -1, 3]} -- {metrics_over_times[:, -1, 3].mean()}
    """
    print(string)
    
def plot_metrics(metrics_over_times):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    metrics_over_time = np.stack(metrics_over_times).mean(axis=0)
    for i, (ax, name) in enumerate(zip(axs, ['LPIPS', 'PSNR', 'SSIM', 'NMSE'])):
        metric = metrics_over_time[:, i]
        ax.plot(metric, label=name)
        ax.legend()
    plt.savefig(os.path.join(args.outdir, 'average_metrics_over_time.png'))
    plt.close(fig)

# Do Inference
for i, ref_img in enumerate(loader):
    if (save_idx + len(ref_img)) < args.start_index:
        save_idx += len(ref_img)
        continue
    ref_img = ref_img.to(device)
    
    if measure_config['operator'] ['name'] == 'inpainting':
        mask = mask_gen(ref_img)
        mask = mask[:, 0:1, :, :]
    else:
        mask = None
                        
    # Forward measurement model (Ax + n)
    y = operator.forward(ref_img, mask=mask)
    y_n = noiser(y)
    
    env = env_class(
      state_cost=lambda x, y: state_cost(x, y, mask=mask),
      target=y_n.reshape(len(y_n), -1),
      control_type=control_type,
      weight_scale=weight_scale)
    init_state = env.initialize_state(n=len(y_n))
    
    if args.warm_start:
        solver = Bayes(env, classifier_weight=1., 
                       use_running_state_cost=use_running_state_cost)
        init_actions, state = solver.solve(init_state, num_iterations=num_iterations)
        env.u_nominal = init_actions.clone()
    else:
        init_actions = None
    
    solver = alg_class(env, **kwargs)
    actions, states = solver.solve(
      init_state, actions=init_actions, num_iterations=num_iterations)
    sample = states[-1].reshape(-1, *shape)
    
    metrics_over_time = get_metrics(ref_img, states)
    
    metrics_over_times.append(metrics_over_time)
    ref_imgs.append(ref_img.detach().cpu())
    y_ns.append(y_n.detach().cpu())
    samples.append(sample.detach().cpu())
    actions_vals.append(actions.detach().cpu())
    
    print_metrics(metrics_over_times)
    
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    f = r - a  # free inside reserved
    
    print(f"Torch GPU memory (GB) - total: {t:.2f}, free: {f:.2f}, reserved: {r:.2f}, alloc: {a:.2f}")
    
    for root, im_list in zip([ref_root, save_root], [ref_img, sample]):
        for i, x in enumerate(im_list):
            x = process(x)
            im = Image.fromarray(x)
            im.save(os.path.join(root, f"{save_idx + i}.jpg"))
    save_idx += i + 1
        
    plot_metrics(metrics_over_times)
    
    fig, ax = plt.subplots()
    mean_actions = (torch.concatenate(actions_vals, axis=1) ** 2).sum(dim=-1).sqrt()
    ax.plot(mean_actions.detach().cpu())
    plt.savefig(os.path.join(args.outdir, 'mean_actions.png'))
    plt.close(fig)
        
    if (i + 1) % 10 == 0:
        print('', '*' * 40, '\n' * 2, f"Finished {i + 1} / {len(loader)} batches.", '\n' * 2, '*' * 40)
        with open(os.path.join(args.outdir, 'tmp.pkl'), 'wb') as f:
            obj = {
                'samples': torch.cat(samples),
                'y_ns': torch.cat(y_ns),
                'ref_imgs': torch.cat(ref_imgs),
                'lpips_vals': torch.cat(lpips_vals),
                'lpips_over_times': torch.cat(lpips_over_times)
            }
            pickle.dump(obj, f)
                            
samples = torch.cat(samples)
y_ns = torch.cat(y_ns)
ref_imgs = torch.cat(ref_imgs)

print(env.terminal_cost(states[-1].to(device)).item())

fig, axs = plt.subplots(1, 3, figsize=(5, 20))
zip_obj = zip(axs, [samples, ref_imgs, y_ns], ['recon', 'reference', 'measurement'])
for ax, x, label in zip_obj:
    img_grid = make_grid(x[:10], normalize=True, nrow=1).permute(1, 2, 0).detach().cpu()
    ax.imshow(img_grid)
    ax.axis('off')
    ax.set_title(label)
    
plt.savefig(os.path.join(args.outdir, 'reconstructed_images.png'))


plot_metrics(metrics_over_times)


with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
    obj = {
        'samples': samples,
        'y_ns': y_ns,
        'ref_imgs': ref_imgs,
        'metrics_over_times': metrics_over_times,
    }
    pickle.dump(obj, f)
