import torch
import numpy as np
import time
from optimizers.util import *
from utils.sde_lib import VPSDE

class DDNM:
    def __init__(self, env, operator=None, use_running_state_cost=True, classifier_weight=1., print_every=1):
        self.env = env
        self.use_running_state_cost = use_running_state_cost
        self.classifier_weight = classifier_weight
        self.print_every = print_every
        self.operator = operator
        
    def solve(self, init_state, num_iterations=1, actions=None):
        if actions is None:
            actions = torch.zeros(size=(self.env.num_steps - 1, len(init_state), self.env.control_dim), device=init_state.device)
        
        final_states = []
        with torch.no_grad():
            for i in range(num_iterations):
                nominal_actions = actions
                # generate trajectory w.r.t. nominal actions
                actions = []
                state = init_state
                for t, action in zip(self.env.timesteps, nominal_actions):
                    x_0_hat = self.env.denoise(t, state, torch.zeros_like(action)).reshape(-1, *self.env.shape)
                    target_reshaped = self.env.target.reshape(-1, 3, 64, 64)
                    ATy = self.operator.transpose(target_reshaped)
                    ATAx = self.operator.transpose(self.operator.forward(x_0_hat))
                    x_0 = x_0_hat + (ATy - ATAx) * self.classifier_weight
                    mean, std = VPSDE().marginal_prob(x_0, t[None])
                    state = mean + std * torch.randn_like(mean)
                    state = state.reshape(-1, np.prod(self.env.shape))
                    actions.append(action)

                final_states.append(state)
                actions = torch.stack(actions)
                
                if i % self.print_every == 0:
                    print(f"iter: {i}, cost: {self.env.terminal_cost(state).sum().item()}")
                                                
        return actions, final_states