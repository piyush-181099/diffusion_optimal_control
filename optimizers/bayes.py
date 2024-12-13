import torch
import numpy as np
import time
from optimizers.util import *

class Bayes:
    def __init__(self, env, use_running_state_cost=True, classifier_weight=1., print_every=1):
        self.env = env
        self.use_running_state_cost = use_running_state_cost
        self.classifier_weight = classifier_weight
        self.print_every = print_every
        
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
                    fn = lambda u: self.env.terminal_cost(self.env.denoise(t, state, u)).sum()
                    grad_log_p_y_x = torch.func.jacrev(fn)(action).detach()
                    action += -grad_log_p_y_x * self.classifier_weight
                    state = self.env.step(t, state, action).detach()
                    actions.append(action)

                final_states.append(state)
                actions = torch.stack(actions)
                
                if i % self.print_every == 0:
                    print(f"iter: {i}, cost: {self.env.terminal_cost(state).sum().item()}")
                                                
        return actions, final_states