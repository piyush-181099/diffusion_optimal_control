import torch
import numpy as np
import time
from optimizers.util import *

class AdjointSensitivity:
    def __init__(self, env, use_running_state_cost=True, lr=1e-3, print_every=10, scheduled_lr=False, seed=None):
        self.env = env
        self.use_running_state_cost = use_running_state_cost
        self.lr = lr
        self.print_every = print_every
        self.seed = seed
        self.scheduled_lr = scheduled_lr
        
    def get_cost(self, states, actions):
        """
        Takes states and actions as input and returning running / terminal costs.
        
        Args:
            states:  tensor of shape (num_steps, state_dim)
            actions: tensor of shape (num_steps - 1, control_dim)
            
        Returns:
            running_cost:  float
            terminal_cost: float
        """
        running_cost = 0
        for t, state, action in zip(self.env.timesteps[:-1], states[:-1], actions):
            state, action = state.reshape(-1, self.env.state_dim), action.reshape(-1, self.env.control_dim)
            running_cost += self.env.running_control_cost(t, x=state, u=action).squeeze()
            if self.use_running_state_cost:
                running_cost += self.env.running_state_cost(t, x=state, u=action)
            
        terminal_cost = self.env.terminal_cost(states[-1].reshape(-1, self.env.state_dim))
        return running_cost, terminal_cost
        
    def solve(self, init_state, actions=None, num_iterations=500, init_fn=torch.zeros):
        n_batch = len(init_state)
        action_shape = (self.env.num_steps - 1, n_batch, self.env.control_dim)
        
        if actions is None:
            actions = init_fn(size=action_shape, device=init_state.device)
            
        actions = [action.requires_grad_() for action in actions.reshape(action_shape)]
        
        ts = torch.linspace(1, 0, self.env.num_steps)[:-1]
        stds = self.env.g(ts) ** 2 / self.env.num_steps
        # stds = ts
        param_groups = []
        for action, std in zip(actions, stds):
            lr = std.item() * self.lr if self.scheduled_lr else self.lr
            param_groups.append({'params': action, 'lr': lr})

        optim = torch.optim.AdamW(param_groups, lr=self.lr)
                
        final_states = []
        time0 = time.time()
        
        try:
            for i in range(num_iterations):
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                optim.zero_grad()
                running_cost, terminal_cost, final_state = self.iterate(init_state, actions)
                final_states.append(final_state.cpu().detach())
                optim.step()
                
                if (i + 1) % self.print_every == 0:
                    print(f"iter: {i + 1}, norm: {np.sum([action.norm().item() for action in actions]):2.2e},",
                          f"cost: {(running_cost + terminal_cost).mean().item():2.4e}",
                          f"-- running: {running_cost.mean().item():2.2e} /",
                          f"terminal: {terminal_cost.mean().item():2.2e}, lr: {optim.param_groups[0]['lr']}")
        except KeyboardInterrupt as e:
            pass
        
        with torch.no_grad():
            states = [init_state]
            for t, action in zip(self.env.timesteps, actions):
                state = self.env.step(t, states[-1], action)
                states.append(state)
            final_states.append(state.reshape(-1, self.env.state_dim).cpu().detach())
            
        actions = torch.stack(actions)
            
        return actions.squeeze(), final_states

    def iterate(self, init_state, actions):
        time0 = time.time()
        with torch.no_grad():
            states = [init_state]
            for t, action in zip(self.env.timesteps, actions):
                state = self.env.step(t, states[-1], action)
                states.append(state)
                
            running_cost, terminal_cost = self.get_cost(states, actions)
            
            Vx_fn = torch.func.grad(lambda x: self.env.terminal_cost(x).sum())
            Vx = Vx_fn(state.reshape(-1, self.env.state_dim)).reshape(-1, self.env.state_dim)
                
            for t in range(self.env.num_steps - 2, -1, -1):
                step_fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
                Vx, ju = torch.func.vjp(step_fn, states[t], actions[t])[1](Vx)
                lu = self.env.j_cost(self.env.timesteps[t], states[t], actions[t])[1].reshape(-1, self.env.control_dim)
                assert actions[t].grad is None
                actions[t].grad = ju + lu
                                
        return running_cost, terminal_cost, states[-1]