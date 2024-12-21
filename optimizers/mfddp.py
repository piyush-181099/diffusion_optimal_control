import torch
import numpy as np
import time
from optimizers.util import *
from optimizers.ddp import DDP
import sys

class MFDDP(DDP):
    """
    Matrix-Free DDP --- DDP without ever instantiating any quadratic (i.e. Hessian) matrices.
    """
    def __init__(self, *args, k_mf=16, update_mode='adam', lr=5e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_mf = k_mf
        self.optimizers = None
        self.update_mode = update_mode
        self.lr = lr
        
    def update_actions(self, *args, **kwargs):
        if self.update_mode == 'adam':
            return self.adam(*args, **kwargs)
        elif self.update_mode == 'backtrack':
            return self.backtrack(*args, **kwargs)
        elif self.update_mode == 'adam_backtrack':
            return self.adam_backtrack(*args, **kwargs)
        
    def adam(self, states, actions, ks, qTKs, qs, iter_num, last=False):
        """
        Computes a backtracking line search on the actions given states, actions,
        update directions ks, and feedback gains Ks.
        
        Args:
            states:    tensor of shape (num_steps, state_dim)
            actions:   tensor of shape (num_steps - 1, control_dim)
            ks:        tensor of shape (num_steps, control_dim, 1)
            Ks:        tensor of shape (num_steps, control_dim, state_dim)
            iter_num:  int - for printing purposes
            last:      bool - for printing purposes
            
        Returns:
            cand_actions: tensor of shape (num_steps - 1, state_dim)
        """
        d_s, d_c = self.env.state_dim, self.env.control_dim
        n_steps, n_batch, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        orig_actions = actions.clone()
        
        if self.optimizers is None:
            actions.requires_grad_(True)
            self.optimizers = [torch.optim.AdamW([action], lr=self.lr) for action in actions]
        
        cur_state = states[0]
        new_states = [cur_state]
        for t in range(n_steps):
            self.optimizers[t].param_groups[0]['params'][0].grad = -ks[t].reshape(n_batch, d_c)
            self.optimizers[t].step()
            
            res = cur_state - states[t]
            actions[t] += (qs[t] @ (qTKs[t] @ (qs[t].mT @ res.unsqueeze(-1)))).reshape(n_batch, d_c)
            cur_state = self.env.step(self.env.timesteps[t], cur_state.reshape(n_batch, d_s), actions[t].reshape(n_batch, d_c))
            new_states.append(cur_state)

        running_cost, terminal_cost = self.get_cost(new_states, actions)
        
        if last or (self.verbose == 1 and (iter_num + 1) % self.print_every == 0):
            print(
                f"iter: {iter_num + 1}, norm: {actions.norm().item():2.2e}, total_cost: {(running_cost + terminal_cost).item():2.4e}",
                f"running: {running_cost.item():2.4e}, terminal: {terminal_cost.item():2.4e}",
                 )
        
        sys.stdout.flush() # this will flush out the print buffers and write to output files
                
        return torch.stack(new_states), actions
    
    def offdiag_inverse(self, A_scalar, U, C, V, mode='matrix', v=None, eps=0.):
        A_inv_scalar = 1. / (A_scalar + self.eps)
        C_inv = inv(C, eps)
        
        # based on the woodbury matrix identity: A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
        # where CRUCIALLY the first term is not used, i.e.: -A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
                        
        if mode == 'matrix':
            return -(A_inv_scalar * U) @ inv(C_inv + A_inv_scalar * V @ U, self.eps) @ (A_inv_scalar * V)
        elif mode == 'left_prod':
            return -(A_inv_scalar * v) @ U @ inv(C_inv + A_inv_scalar * V @ U, self.eps) @ (A_inv_scalar * V)
        elif mode == 'right_prod':
            return -(A_inv_scalar * U) @ inv(C_inv + A_inv_scalar * V @ U, self.eps) @ ((A_inv_scalar * V) @ v)
        else:
            raise NotImplementedError()
    
    def approx_inverse(self, A_scalar, U, C, V, eps=0.):
        n = V.shape[-1]
        
        A = lambda v: (self.offdiag_inverse(A_scalar, U, C, V, mode='left_prod', v=v.mT, eps=eps), )
        AT = lambda v: self.offdiag_inverse(A_scalar, U, C, V, mode='right_prod', v=v[0], eps=eps)
        
        (q,), (qTAUCV,) = self.low_rank_approx(
          A=A, AT=AT, input_shapes=((n,),), output_shapes=((n,),), 
          k=self.k_mf, mode='2k', device=A_scalar.device)
        
        return q, qTAUCV
    
    def compute_gradients(self, states, actions):
        '''
        Obtain the gradients k and the feedback gains K with respect to actions.
        
        Args:
            states:  tensor of shape (num_steps, state_dim) - states of the control episode
            actions: tensor of shape (num_steps - 1, control_dim) - initial actions
            
        Returns:
            ks:      tensor of shape (num_steps - 1, control_dim) - action gradients
            Ks:      tensor of shape (num_steps - 1, control_dim, state_dim) - feedback gains
        '''
        t0 = time.time()
        d_s, d_c = self.env.state_dim, self.env.control_dim
        n_steps, d_b, _ = actions.shape
        state = states[-1]
        
        # compute Vx
        Vx_fn = torch.func.jacrev(self.env.terminal_cost)
        Vx = Vx_fn(state.reshape(d_b, d_s))

        # compute Vxx
        (qv,), (qvTVxx,) = self.lr_jacobian(Vx_fn, state.reshape(d_b, d_s), k=self.k_hessian, mode=self.lr_mode)
        qv, qvTVxx = qv[0], qvTVxx[0] @ qv[0] @ qv[0].mT
        
        # compute derivatives of the running cost
        qls, lxs, lus, qTlxxs, luus, lxus, luxs = self.running_cost_derivatives(states[:-1], actions, projected=True)

        ks = [None] * n_steps
        Ks = [None] * n_steps
        qvs = [None] * n_steps
        # DDP loop
        for t in range(n_steps - 1, -1, -1):
            state, action = states[t].clone(), actions[t].clone()

            # compute Qxx, Quu, Qux, Qxu
            lx, lu = lxs[t], lus[t]
            ql, qTlxx, luu_scalar, lxu_scalar, lux_scalar = qls[t], qTlxxs[t], luus[t], lxus[t], luxs[t]
            
            # compute jacobians of the step function
            if self.lr_identity:
                fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u) - (x + u)
            else:
                fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
            
            step_fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
            fxT_Vx, fuT_Vx = torch.func.vjp(step_fn, state, action)[1](Vx.reshape(d_b, d_s))
            (q,), (qTfx, qTfu) = self.lr_jacobian(fn, state, action, k=self.k_jacobian, mode=self.lr_mode)
            
            if self.lr_identity:
                qTfx, qTfu = qTfx + q.mT, qTfu + q.mT
                
            A = lambda v: (v.mT @ qTfx.mT @ (q.mT @ qv) @ qvTVxx @ q @ qTfx,)
            AT = lambda v: qTfx.mT @ (q.mT @ qv) @ (qvTVxx @ q) @ (qTfx @ v[0])
            (qv,), (qvTV,) = self.low_rank_approx(
              A=A, AT=AT, input_shapes=((d_b, d_s),), output_shapes=((d_b, d_s),), 
              k=self.k_mf, mode=self.lr_mode, device=state.device)
            
            qvTQxx = (qv.mT @ ql) @ qTlxx + qvTV
            qvTQuu = luu_scalar * qv.mT + qvTV
            qvTQux = lux_scalar * qv.mT + qvTV
            qvTQxu = lxu_scalar * qv.mT + qvTV

            Qx = lx + fxT_Vx.reshape(d_b, d_s, 1)
            Qu = lu + fuT_Vx.reshape(d_b, d_c, 1)
            
            qu, quTinv = self.approx_inverse(luu_scalar, qv, qvTV @ qv, qv.mT, eps=self.eps)

            k = -(Qu / (luu_scalar + self.eps) + qu @ (quTinv @ Qu))
            qvTK = -0.5 * qvTQux / (luu_scalar + self.eps) + qv.mT @ qu @ quTinv @ qv @ qvTQux

            Vx = Qx - qvTK.mT @ (qvTQuu @ k)
            qvTVxx = qvTQxx - qv.mT @ qvTK.mT @ qvTQuu @ qv @ qvTK
            
            ks[t] = k.detach()
            Ks[t] = qvTK.detach() @ qv
            qvs[t] = qv.detach()
            
        self.qs = qvs
        self.qTKs = Ks
                
        return ks, Ks, qvs