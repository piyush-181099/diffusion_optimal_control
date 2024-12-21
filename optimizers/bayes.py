import torch
import numpy as np
import time
from optimizers.util import *

class Bayes:
    def __init__(self, env, use_running_state_cost=True, classifier_weight=1.):
        self.env = env
        self.use_running_state_cost = use_running_state_cost
        self.classifier_weight = classifier_weight
        
    def solve(self, init_state, num_iterations=0, actions=None):
        with torch.no_grad():
            nominal_actions = torch.zeros(size=(self.env.num_steps - 1, len(init_state), self.env.control_dim), device=init_state.device)

            # generate trajectory w.r.t. nominal actions
            states = [init_state]
            actions = []
            for t, action in zip(self.env.timesteps, nominal_actions):
                fn = lambda u: self.env.terminal_cost(self.env.denoise(t, states[-1], u)).sum()
                grad_log_p_y_x = torch.func.jacrev(fn)(action).detach()
                action = -grad_log_p_y_x * self.classifier_weight
                state = self.env.step(t, states[-1], action).detach()
                states.append(state)
                actions.append(action)

            states = torch.cat(states)
            actions = torch.stack(actions)

            # generate trajectory w.r.t. nominal actions
            states = [init_state]
            for t, action in zip(self.env.timesteps, actions):
                state = self.env.step(t, states[-1], action)
                states.append(state)
                                    
        return actions.squeeze(), [state.detach()]