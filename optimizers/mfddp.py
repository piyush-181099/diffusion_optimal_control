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
    def __init__(self, *args, k_mf=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_mf = k_mf
        self.optimizers = None
        
    def update_actions(self, *args, **kwargs):
        # return self.adam_backtrack(*args, **kwargs)
        return self.adam(*args, **kwargs)
        # return self.backtrack(*args, **kwargs)
        
    def adam(self, states, actions, ks, qTKs, qs, iter_num, num_tries=5, last=False):
        """
        Computes a backtracking line search on the actions given states, actions,
        update directions ks, and feedback gains Ks.
        
        Args:
            states:    tensor of shape (num_steps, state_dim)
            actions:   tensor of shape (num_steps - 1, control_dim)
            ks:        tensor of shape (num_steps, control_dim, 1)
            Ks:        tensor of shape (num_steps, control_dim, state_dim)
            iter_num:  int - for printing purposes
            num_tries: int - number of steps of backtracking line search
            last:      bool - for printing purposes
            
        Returns:
            cand_actions: tensor of shape (num_steps - 1, state_dim)
        """
        n_steps, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        orig_actions = actions.clone()
        
        if self.optimizers is None:
            actions.requires_grad_(True)
            self.optimizers = [torch.optim.AdamW([action], lr=1e-3) for action in actions]
        
        cur_state = states[:1]
        new_states = [cur_state]
        for t in range(n_steps):
            self.optimizers[t].param_groups[0]['params'][0].grad = -ks[t].squeeze()
            self.optimizers[t].step()
            
            res = cur_state - states[t]
            # print("feedback gains norm", (qs[t] @ (qTKs[t] @ res.unsqueeze(-1))).squeeze().norm())
            actions[t] += (qs[t] @ (qTKs[t] @ res.unsqueeze(-1))).squeeze()
            cur_state = self.env.step(self.env.timesteps[t], cur_state, actions[t].unsqueeze(0))
            new_states.append(cur_state)

        running_cost, terminal_cost = self.get_cost(new_states, actions)
        
        if last or (self.verbose == 1 and (iter_num + 1) % self.print_every == 0):
            print(
                f"iter: {iter_num + 1}, norm: {actions.norm().item():2.2e}, total_cost: {(running_cost + terminal_cost).item():2.4e}",
                f"running: {running_cost.item():2.4e}, terminal: {terminal_cost.item():2.4e}",
                 )
        
        # print("step size", (actions - orig_actions).norm().item())
        sys.stdout.flush() # this will flush out the print buffers and write to output files
                
        return actions
        
    def backtrack(self, states, actions, ks, qTKs, qs, iter_num, num_tries=5, last=False):
        """
        Computes a backtracking line search on the actions given states, actions,
        update directions ks, and feedback gains Ks.
        
        Args:
            states:    tensor of shape (num_steps, state_dim)
            actions:   tensor of shape (num_steps - 1, control_dim)
            ks:        tensor of shape (num_steps, control_dim, 1)
            Ks:        tensor of shape (num_steps, control_dim, state_dim)
            iter_num:  int - for printing purposes
            num_tries: int - number of steps of backtracking line search
            last:      bool - for printing purposes
            
        Returns:
            cand_actions: tensor of shape (num_steps - 1, state_dim)
        """
        alpha = 1.0
        n_steps, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        
        for _ in range(num_tries):
            state = states[:1]
            cand_actions = []
            new_states = [state]
            for t, orig_state, action, k, qTK, q in zip(self.env.timesteps[:-1], states, actions, ks, qTKs, qs):
                k, qTK, q = k.to(action.device), qTK.to(action.device), q.to(action.device)
                res = state - orig_state
                new_action = action.unsqueeze(-1) + alpha * k + q @ (qTK @ res.unsqueeze(-1))
                new_action = new_action.reshape(1, self.env.control_dim)
                state = self.env.step(t, state, new_action)
                new_states.append(state)
                cand_actions.append(new_action)
                
            cand_actions = torch.cat(cand_actions)
            running_cost, terminal_cost = self.get_cost(new_states, cand_actions)
            cand_cost = running_cost + terminal_cost
            
            if cand_cost < orig_cost:
                self.eps *= self.success_multiplier
                break
            alpha *= 0.1
        else:
            self.eps *= self.failure_multiplier
            print(f"linesearch failed, eps={self.eps:.3e}")
            cand_cost = orig_cost
            cand_actions = actions
        
        if last or (self.verbose == 1 and (iter_num + 1) % self.print_every == 0):
            print(
                f"iter: {iter_num + 1}, norm: {cand_actions.norm().item():2.2e}, total_cost: {cand_cost.item():2.4e}",
                f"running: {running_cost.item():2.4e}, terminal: {terminal_cost.item():2.4e}",
                 )
        
        # print("step size", (actions - cand_actions).norm().item(), alpha)
        sys.stdout.flush() # this will flush out the print buffers and write to output files
                
        return cand_actions
    
    def adam_backtrack(self, states, actions, ks, qTKs, qs, iter_num, num_tries=5, last=False):
        """
        Computes a backtracking line search on the actions given states, actions,
        update directions ks, and feedback gains Ks.
        
        Args:
            states:    tensor of shape (num_steps, state_dim)
            actions:   tensor of shape (num_steps - 1, control_dim)
            ks:        tensor of shape (num_steps, control_dim, 1)
            Ks:        tensor of shape (num_steps, control_dim, state_dim)
            iter_num:  int - for printing purposes
            num_tries: int - number of steps of backtracking line search
            last:      bool - for printing purposes
            
        Returns:
            cand_actions: tensor of shape (num_steps - 1, state_dim)
        """
        alpha = 1.0
        n_steps, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        
        if self.optimizers is None:
            actions.requires_grad_(True)
            self.optimizers = [torch.optim.AdamW([action], lr=1e-3) for action in actions]
            
        def reset_optimization_state(lr=1e-3):
            with torch.no_grad():
                actions[:] = orig_actions[:]
            
            for opt, state_dict in zip(self.optimizers, state_dicts):
                opt.load_state_dict(state_dict)
                for g in opt.param_groups:
                    g['lr'] = lr * 1e-3
        
        state_dicts = [opt.state_dict() for opt in self.optimizers]
        orig_actions = actions.clone()
        for _ in range(num_tries):
            reset_optimization_state(alpha)
            cur_state = states[:1]
            new_states = [cur_state]
            for t in range(n_steps):
                self.optimizers[t].param_groups[0]['params'][0].grad = -ks[t].squeeze()
                self.optimizers[t].step()

                res = cur_state - states[t]
                actions[t] += (qs[t] @ (qTKs[t] @ res.unsqueeze(-1))).squeeze()
                cur_state = self.env.step(self.env.timesteps[t], cur_state, actions[t].unsqueeze(0))
                new_states.append(cur_state)

            running_cost, terminal_cost = self.get_cost(new_states, actions)
            cand_cost = running_cost + terminal_cost
            
            if cand_cost < orig_cost:
                self.eps *= self.success_multiplier
                break
            alpha *= 0.5
        else:
            reset_optimization_state(alpha)
            self.eps *= self.failure_multiplier
            print(f"linesearch failed, eps={self.eps:.3e}")
            cand_cost = orig_cost
        
        if last or (self.verbose == 1 and (iter_num + 1) % self.print_every == 0):
            print(
                f"iter: {iter_num + 1}, norm: {actions.norm().item():2.2e}, total_cost: {cand_cost.item():2.4e}",
                f"running: {running_cost.item():2.4e}, terminal: {terminal_cost.item():2.4e}",
                 )
        
        # print("step size", (orig_actions - actions).norm().item(), alpha)
        sys.stdout.flush() # this will flush out the print buffers and write to output files
                
        return actions
    
    def offdiag_inverse(self, A_scalar, U, C, V, mode='matrix', v=None):
        A_inv_scalar = 1. / (A_scalar + self.eps)
        C_inv = inv(C, self.eps)
        
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
    
    def approx_inverse(self, A_scalar, U, C, V):
        n = V.shape[-1]
        
        A = lambda v: (self.offdiag_inverse(A_scalar, U, C, V, mode='left_prod', v=v.mT), )
        AT = lambda v: self.offdiag_inverse(A_scalar, U, C, V, mode='right_prod', v=v[0])
        
        (q,), (qTAUCV,) = self.low_rank_approx(
          A=A, AT=AT, input_shapes=((n,),), output_shapes=((n,),), 
          k=self.k_mf, mode=self.lr_mode, device=A_scalar.device)
        
        return q, qTAUCV
    
    def loop(self, fn, *xs, batch_size=25):
        def tree_cat(trees, axis=0):
            # flatten list and transpose
            flat_list = [torch.utils._pytree.tree_flatten(tree)[0] for tree in trees]
            flat_list = [list(x) for x in zip(*flat_list)]
            
            # assume all trees have the same spec
            spec = torch.utils._pytree.tree_flatten(trees[0])[1]
            return torch.utils._pytree.tree_unflatten([torch.cat(xs, axis=axis) for xs in flat_list], spec)
        
        batched_inputs = zip(*[x.split(batch_size) for x in xs])
        outputs = [fn(*inputs) for inputs in batched_inputs]
            
        return tree_cat(outputs)
    
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
        d_s, d_c = self.env.state_dim, self.env.control_dim
        n_steps, _ = actions.shape
        state = states[-1]
        
        # compute Vx
        Vx_fn = torch.func.jacrev(self.env.terminal_cost)
        Vx = Vx_fn(state.reshape(1, d_s))

        # compute Vxx
        (qv,), (qvTVxx,) = self.lr_jacobian(Vx_fn, state.reshape(1, d_s), k=self.k_hessian, mode=self.lr_mode)
        qv, qvTVxx = qv[0], qvTVxx[0] @ qv[0] @ qv[0].mT

        # compute jacobians of the step function
        if self.lr_identity:
            get_fn = lambda t: lambda x, u: self.env.step(t, x, u) - (x + u)
        else:
            get_fn = lambda t: lambda x, u: self.env.step(t, x, u)

        lrj_fn = lambda t, x, u: self.lr_jacobian(get_fn(t), x, u, k=self.k_jacobian, mode=self.lr_mode)
        (qs,), (qTfxs, qTfus) = self.loop(lrj_fn, self.env.timesteps[:-1], states[:-1], actions)
        
        # compute derivatives of the running cost
        qls, lxs, lus, qTlxxs, luus, lxus, luxs = self.running_cost_derivatives(states[:-1], actions, projected=True)

        # for debugging recompute the jacobians:
        if self.debugging_mode:
            def print_dist(w_true, w_hat, print_str='', k=self.k_jacobian):
                assert w_true.shape == w_hat.shape, f"{w_true.shape}, {w_hat.shape}"
                s_true = torch.svd(w_true)[1]
                dist = torch.svd(w_true - w_hat)[1].sum()
                norm_dist = dist / (s_true.sum() + 1e-6)
                optimal = s_true[:, k:].sum() / s_true.sum()
                print(f"{print_str}{dist:.3f}, normalized: {norm_dist:.3f}, optimal: {optimal:.3f}")

            get_fn = lambda t: lambda x, y: self.env.step(self.env.timesteps[t], x, y)
            get_l_fn = lambda t: lambda x: self.env.running_state_cost(self.env.timesteps[t], x, actions[t])

            fxs_true = []
            fus_true = []
            lxx_true = []
            for i in range(n_steps):
                fx_true, fu_true = torch.func.jacrev(get_fn(i), argnums=(0, 1))(states[i], actions[i])
                fxs_true.append(fx_true.squeeze())
                fus_true.append(fu_true.squeeze())
                lxx_true.append(torch.func.hessian(get_l_fn(i))(states[i]).squeeze())

            lxx_true = torch.stack(lxx_true)
            fxs_true = torch.stack(fxs_true)
            fus_true = torch.stack(fus_true)
            Vxx_true = torch.func.hessian(self.env.terminal_cost)(states[-1]).squeeze()
            if self.lr_identity:
                I = torch.eye(d_s, device=state.device)
                fxs, fus = qs @ qTfxs + I.unsqueeze(0), qs @ qTfus + I.unsqueeze(0)
            else:
                fxs, fus = qs @ qTfxs, qs @ qTfus

            print(f"Shapes are fxs: {fxs.shape} and fus: {fus.shape}")
            print(f"shapes are fxs_debug: {fxs_true.shape}") 
            print_dist(fxs_true, fxs, "norm of difference between fxs: ")
            print_dist(fus_true, fus, "norm of difference between fus: ")
            print_dist(lxx_true, lxxs, "norm of difference between lxxs: ", k=self.k_hessian)
            print_dist(Vxx_true[None], qv @ qvTVxx[None], "norm of difference between Vxxs: ", k=self.k_hessian)

        ks = [None] * n_steps
        Ks = [None] * n_steps
        qvs = [None] * n_steps
        # DDP loop
        for t in range(n_steps - 1, -1, -1):
            state, action = states[t].clone(), actions[t].clone()

            # compute Qxx, Quu, Qux, Qxu
            lx, lu = lxs[t], lus[t]
            ql, qTlxx, luu_scalar, lxu_scalar, lux_scalar = qls[t], qTlxxs[t], luus[t], lxus[t], luxs[t]
            q, qTfx, qTfu = qs[t], qTfxs[t], qTfus[t]
            
            if self.lr_identity:
                qTfx, qTfu = qTfx + q.mT, qTfu + q.mT
                
            A = lambda v: (v.mT @ qTfx.T @ (q.mT @ qv) @ qvTVxx @ q @ qTfx,)
            AT = lambda v: qTfx.T @ (q.mT @ qv) @ (qvTVxx @ q) @ (qTfx @ v[0])
            (qv,), (qvTV,) = self.low_rank_approx(
              A=A, AT=AT, input_shapes=((d_s,),), output_shapes=((d_s,),), 
              k=self.k_mf, mode=self.lr_mode, device=state.device)
            
            qvTQxx = (qv.mT @ ql) @ qTlxx + qvTV
            qvTQuu = luu_scalar * qv.mT + qvTV
            qvTQux = lux_scalar * qv.mT + qvTV
            qvTQxu = lxu_scalar * qv.mT + qvTV
            
            # the below would be equivalent to using the true jacobian without instantiating it
            step_fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
            fxT_Vx, fuT_Vx = torch.func.vjp(step_fn, state, action)[1](Vx.reshape(1, d_s))

            Qx = lx + fxT_Vx.reshape(d_s, 1)
            Qu = lu + fuT_Vx.reshape(d_c, 1)

            qu, quTinv = self.approx_inverse(luu_scalar, qv, qvTV @ qv, qv.mT)
            Quu_inv_prod = lambda B: B / (luu_scalar + self.eps) + qu @ (quTinv @ B)

            k = -Quu_inv_prod(Qu)
            # qvTK = (-0.5 * qv.mT) @ Quu_inv_prod((qv @ qvTQxu).mT + qv @ qvTQux)
            qvTK = -0.5 * qvTQux / (luu_scalar + self.eps) + qv.mT @ qu @ quTinv @ qv @ qvTQux

            Vx = Qx - qvTK.mT @ (qvTQuu @ k)
            qvTVxx = qvTQxx - qv.mT @ qvTK.mT @ qvTQuu @ qv @ qvTK

            ks[t] = k.detach()
            Ks[t] = qvTK.detach()
            qvs[t] = qv.detach()
                
        return ks, Ks, qvs