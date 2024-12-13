import numpy as np
import torch
from utils.sde_lib import VPSDE

class Diffusion:
    def __init__(
        self,
        shape,
        weight_scale=1.,
        num_steps=100,
        mode='ddim',
        control_type='pixel',
        u_nominal=None,
        device='cpu',
        eps=1e-3,
        verbose=False,
        seed=None,
    ):
        self.shape = shape
        self.weight_scale = weight_scale
        self.no_running_cost = weight_scale == 0.
        self.ndims = np.prod(shape) # eg. a shape of (1, 28, 28) would give 1*28*28=784
        self.num_steps = num_steps
        self.timesteps = torch.linspace(1, eps, num_steps, device=device)
        self.inv_timesteps = lambda t: ((1 - t) * num_steps).long()
        self.device = device
        self.eps = eps # we simulate in the time interval t \in [eps, 1.0]
        self.verbose = verbose
        self.control_type = control_type
        self.mode = mode
        self.u_nominal = u_nominal
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000)

        if control_type == 'pixel' or control_type == 'pixel_after':
            self.control_dim = self.ndims
        elif control_type == 'h':
            self.control_dim = np.prod(self.h_dim)
        else:
            raise NotImplementedError(f"control_type: {control_type} not supported")

        self.state_dim = self.ndims

        self.dt = (1 - self.eps) / self.num_steps

    def initialize_state(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        device = self.device
        t = torch.ones((n, 1), device=device) # this means we start at terminal-time, t=1.0
        init_x = torch.randn((n, self.ndims), device=device)

        return init_x

    def u_diff(self, t, u):
        if self.u_nominal is not None:
            t_idx = self.inv_timesteps(t)
            u_nominal = self.u_nominal[t_idx].reshape(-1, self.ndims)
            assert u.shape[1:] == u_nominal.shape[1:]
            # print(f't: {t}, diff: {(u - u_nominal).norm()}')
            return u - u_nominal

        return u

    def weight(self, t):
        t = t - self.dt
        weight = (1 - t).reshape(-1, 1)
        # t = t.reshape(-1, 1)
        # weight = (1 - VPSDE().marginal_prob(t, t)[1] ** 2).reshape(-1, 1)
        weight = self.weight_scale * weight

        # weight = torch.where(t > 0.8, weight * 0., weight).reshape(-1, 1)

        return weight * self.num_steps

    def running_state_cost(self, t, x, u):
        weight = self.weight(t).squeeze() # shape: (1, 1)
        x = x.reshape(-1, *self.shape)
        cost = self.state_cost(x, self.target)
        return weight * cost

    def running_control_cost(self, t, x, u):
        weight = self.weight(t) # shape: (1, 1)
        return torch.squeeze(0.5 * ((self.u_diff(t, u) ** 2) * weight).sum(axis=1))

    def terminal_cost(self, x, reduce_sum=True):
        x = x.reshape(-1, *self.shape)
        cost = self.state_cost(x, self.target)
        return cost

    def h_cost(self, t, x, u, low_mem_mode=False):
        weight = self.weight(t)
        if low_mem_mode:
            ones = torch.ones_like(weight, device=x.device)
            return (ones * 0, ones * 0), (ones * 0, ones * weight)
        else:
            z = lambda *shape: torch.zeros(size=shape, device=x.device)
            n, d_s = x.shape
            _, d_c = u.shape

            luu = torch.diag_embed(torch.ones_like(u) * weight)
            lxu, lux, lxx = z(n, d_s, d_c), z(n, d_c, d_s), z(n, d_s, d_s)
            return (lxx, lxu), (lux, luu)

    def j_cost(self, t, x, u):
        weight = self.weight(t)
        z = lambda *shape: torch.zeros(size=shape, device=x.device)

        lx, lu = z(len(x), self.state_dim, 1), (self.u_diff(t, u) * weight).reshape(len(x), self.control_dim, 1)

        return lx, lu

class DDPMDiffusion(Diffusion):
    def __init__(
        self,
        state_cost,
        target,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target = target
        self.state_cost = state_cost

    def step(self, t, x, u):
        t_int = self.discretize_time(t - self.dt)

        if self.control_type == 'pixel':
            assert x.shape == u.shape, f"x.shape: {x.shape}, u.shape: {u.shape}"
            return self.step_fn((x + u).reshape(-1, *self.shape), t_int).reshape(-1, self.ndims)
        elif self.control_type == 'pixel_after':
            assert x.shape == u.shape, f"x.shape: {x.shape}, u.shape: {u.shape}"
            return self.step_fn(x.reshape(-1, *self.shape), t_int).reshape(-1, self.ndims) + u
        elif self.control_type == 'h':
            return self.step_fn(x.reshape(-1, *self.shape), t_int, u=u.reshape(-1, *self.h_dim)).reshape(-1, self.ndims)

    def std(self, t):
        std = VPSDE().marginal_prob(t.reshape(-1), t.reshape(-1))[1]
        return std.squeeze()

    def g(self, t):
        std = VPSDE().sde(t.reshape(-1), t.reshape(-1))[1]
        return std.squeeze()

    def discretize_time(self, t):
        t_int = (t * (self.get_model_steps() - 1)).int().reshape(-1)
        torch.manual_seed((self.seed + t_int).median().item())
        return t_int

class HFDiffusion(DDPMDiffusion):
    def __init__(
        self,
        pipeline,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline
        self.pipeline.scheduler.set_timesteps(self.num_steps)
        self.step_fn = self.pipeline.get_step_fn()
        self.pipeline.unet.eval()

    def denoise(self, t, x, u=None):
        shape = (-1,) + self.shape
        if self.control_type == 'pixel' or self.control_type == 'pixel_after':
            if u is None:
                u = torch.zeros_like(x)
            assert x.shape == u.shape
            x = x + u
        x_0_hat = self.denoise_image(x.reshape(shape), t - self.dt).reshape(-1, self.ndims)
        return x_0_hat

    def denoise_image(self, x, t):
        t_int = self.discretize_time(t)
        return self.pipeline.denoise_image(x, t_int)

    def get_model_steps(self):
        return self.pipeline.scheduler.num_train_timesteps

class DPSDiffusion(DDPMDiffusion):
    def __init__(
        self,
        model,
        sampler,
        *args,
        **kwargs,
    ):
        #self.h_dim = model.h_dim
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.model = model
        self.step_fn = step_fn = lambda x, t, u=None: sampler.p_sample(x=x, t=t, model=model)['sample']

    def get_model_steps(self):
        return self.num_steps

    def denoise(self, t, x, u=None):
        shape = (-1,) + self.shape
        t_int = self.discretize_time(t) - 1
        t_int = t_int.clip(0, self.sampler.num_timesteps - 1)

        denoise_fn = lambda x, u=None: self.sampler.p_sample(x=x.reshape(shape), t=t_int, u=u, model=self.model)['pred_xstart'].reshape(-1, self.ndims)

        if self.control_type == 'pixel' or self.control_type == 'pixel_after':
            if u is None:
                u = torch.zeros_like(x)
            assert x.shape == u.shape
            return denoise_fn(x + u)
        elif self.control_type == 'h':
            if u is None:
                return denoise_fn(x)
            return denoise_fn(x, u=u.reshape(-1, *self.h_dim))


