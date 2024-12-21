import torch
import numpy as np
import time
from optimizers.util import *
import sys

import gc

class DDP:
    def __init__(self, env, k_jacobian=0, k_hessian=0, lr_mode='k', rrf_iters=2,
                 eps=1e-3, success_multiplier=1., failure_multiplier=10., min_eps=1e-8, 
                 chunk_size=8, verbose=1, print_every=10, use_running_state_cost=False,
                 lr_identity=False, debugging_mode=False, seed=None):

        """
        Class initializer.
        
        Args:
            env:                    dynamical system class
            k_jacobian:             slicing dimension for low-rank Jacobian computation
            k_hessian:              slicing dimension for low-rank Hessian computation
            lr_mode:                low-rank Jacobian computation mode ('k' / '2k')
            rff_iters:              number of iterations of the randomized range finder (Halko et. al, 2011)
            eps:                    Tikhonov regularization constant
            success_multiplier:     factor by which to decrease eps when line search succeeds
            failure_multiplier:     factor by which to increase eps when line search fails
            min_eps:                minimum value of eps (to prevent div by zero during Quu inversion)
            verbose:                verbosity of the algorithm
            full_hess:              whether or not to compute the full terminal hessain Vxx (bool)
            print_every:            how frequently to print
            use_running_state_cost: whether to use running state cost which by default is 0.
            lr_identity:            testing -- obtain low rank plus identity approximation
            debugging_mode:         boolean to indicate if debugging mode is ON/OFF and to compare jacobian computation
            seed:                   seed that is used per iteration (to seed stochastic dynamics)
        """
        self.env = env
        self.k_jacobian = k_jacobian
        self.k_hessian = k_hessian
        self.lr_mode = lr_mode
        self.eps = eps
        self.success_multiplier = success_multiplier
        self.failure_multiplier = failure_multiplier
        self.min_eps = min_eps
        self.chunk_size = chunk_size
        self.rrf_iters = rrf_iters
        self.verbose = verbose
        self.print_every = print_every
        self.use_running_state_cost = use_running_state_cost
        self.lr_identity = lr_identity
        self.debugging_mode = debugging_mode
        self.seed = seed
        if self.debugging_mode:
            print(f"\nDEBUG MODE\n")
        
    def solve(self, init_state, actions=None, num_iterations=25, early_stopping=True):
        """
        Main function that implements the DDP algorithm. Takes init_state as input and
        (optionally) actions, and returns a tuple (actions, states)
        containing the actions returned by DDP, as well as a sequence of states containing
        the generated image at the end of each DDP iteration.
        
        Args:
            init_state:      tensor of shape (1, state_dim) - initial state of DDP
            actions:         tensor of shape (num_steps - 1, control_dim) - initial actions
            num_iterations:  int - number of iterations to run DDP
            early_stopping:  bool - whether or not DDP iterations should be cut short given
                             some notion of convergence
            
        Returns: 
            actions:         tensor of shape (num_steps - 1, control_dim) - the result of the 
                             DDP algorithm
            states:          tensor of shape (num_iterations, state_dim) - the generated
                             images at the end of each DDP iteration
        """
        actions_shape = (self.env.num_steps - 1, len(init_state), self.env.control_dim)
        if actions is None:
            actions = torch.zeros(size=actions_shape, device=init_state.device)
        else:
            actions = actions.reshape(actions_shape)
        
        states = None
        end_states = []
        
        if self.verbose == 1:
            print(f"||u_init|| = {actions.norm(dim=1).mean():.3e}, is the norm of the entire control trajectory")
        
        time0 = time.time()
        try:
            for iter_num in range(num_iterations):
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                gc.collect()
                torch.cuda.empty_cache()
                states, actions = self.iterate(init_state.clone(), actions, 
                                               iter_num=iter_num, states=states,
                                               last=iter_num == (num_iterations - 1))
                end_states.append(states[-1].cpu())
        except KeyboardInterrupt as e:
            pass

        print(f"||u_nominal|| = {actions.norm(dim=1).mean():.3e}")
        sys.stdout.flush() # this will flush out the print buffers and write to output files
        
        return actions.squeeze(), end_states
        
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
            state, action = state.reshape(1, -1), action.reshape(1, -1)
            running_cost += self.env.running_control_cost(t, x=state, u=action).squeeze()
            if self.use_running_state_cost:
                running_cost += self.env.running_state_cost(t, x=state, u=action)
            
        terminal_cost = self.env.terminal_cost(states[-1].reshape(1, -1))
        return running_cost, terminal_cost

    def update_actions(self, states, actions, ks, Ks, iter_num, num_tries=20, last=False):
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
        assert len(state) == 1
        alpha = 1.0
        n_steps, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        
        for _ in range(num_tries):
            state = states[:1]
            cand_actions = []
            new_states = [state]
            for t, orig_state, action, k, K in zip(self.env.timesteps[:-1], states, actions, ks, Ks):
                res = state - orig_state
                new_action = action.unsqueeze(-1) + alpha * k + K @ res.unsqueeze(-1) 
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
            alpha *= 0.5
        else:
            self.eps *= self.failure_multiplier
            print(f"linesearch failed, eps={self.eps:.3e}")
            cand_cost = orig_cost
            cand_actions = actions
        
        if last or (self.verbose == 1 and (iter_num + 1) % self.print_every == 0):
            print(f"iter: {iter_num + 1}, norm: {cand_actions.norm().item():2.2e}, cost: {cand_cost.item():2.4e}")
        
        sys.stdout.flush() # this will flush out the print buffers and write to output files
                
        return new_states, cand_actions
                
    def running_cost_derivatives(self, states, actions, projected=False):
        """
        Computes all first- and second- order derivatives of the running cost function
        of the optimal control system system
        
        Args:
            states:  tensor of shape (num_steps, state_dim)
            actions: tensor of shape (num_steps - 1, control_dim)
            
        Returns:
            lx:  tensor of shape (num_steps, state_dim, 1)
            lu:  tensor of shape (num_steps, control_dim, 1)
            lxx: tensor of shape (num_steps, state_dim, state_dim)
            luu: tensor of shape (num_steps, control_dim, control_dim)
            lxu: tensor of shape (num_steps, state_dim, control_dim)
            lux: tensor of shape (num_steps, control_dim, state_dim)
        """
        
        n_steps, d_b, _ = states.shape
        d_s, d_c = self.env.state_dim, self.env.control_dim
        
        states = states.reshape(n_steps * d_b, d_s)
        actions = actions.reshape(n_steps * d_b, d_c)
        if not self.use_running_state_cost:
            batched_timesteps = self.env.timesteps[:-1].tile((d_b,))
            lx, lu = self.env.j_cost(batched_timesteps, states, actions)
            (lxx, lxu), (lux, luu) = self.env.h_cost(batched_timesteps, states, actions, low_mem_mode=projected)
            q = torch.ones((len(states), d_s, 1), device=states.device)
            qTlxx = torch.zeros_like(q.mT)
        else:
            # compute first derivative terms
            _, lu = self.env.j_cost(self.env.timesteps[:-1], states, actions)
            
            running_state_cost = lambda x: self.env.running_state_cost(self.env.timesteps[:-1], x, actions).sum()
            lx_fn = torch.func.jacrev(running_state_cost)
            lx = lx_fn(states)

            # compute second derivative terms
            (_, lxu), (lux, luu) = self.env.h_cost(self.env.timesteps[:-1], states, actions, low_mem_mode=projected)
            
            (q,), (qTlxx,) = self.lr_jacobian(lx_fn, states, k=self.k_hessian, mode=self.lr_mode)
            qTlxx = qTlxx @ q @ q.mT
                            
        lx = lx.reshape(n_steps, d_b, d_s, 1)
        lu = lu.reshape(n_steps, d_b, d_c, 1)
        
        if not projected:
            lxx = q @ qTlxx
            
            lxx = lxx.reshape(n_steps, d_s, d_s)
            lxu = lxu.reshape(n_steps, d_s, d_c)
            lux = lux.reshape(n_steps, d_c, d_s)
            luu = luu.reshape(n_steps, d_c, d_c)
            
            return lx, lu, lxx, luu, lxu, lux
                    
        return q, lx, lu, qTlxx, luu, lxu, lux
      
    def lr_jacobian(self, fn, *xs, k=1, mode='2k', qs=None):
        """
        Computes a low rank approximation of the Jacobian of fn.
        
        Args:
            fn:   the function in question
            xs:   inputs to fn
            k:    the rank of the approximation
            mode: the approximation strategy ('k' / '2k')
            
        Returns:
            qs:  a low rank projection matrix of shape (fn(x).shape, k)
            qTJ: the low rank approximation of J of shape (fn(x).shape, k)
        """
        if mode == 'k':
            output, mjp_fn = get_mjp_fn(fn, *xs, return_output=True, chunk_size=self.chunk_size)
            qs, qTJs = self.low_rank_approx(
              mjp_fn, output_shapes=(output.shape,), input_shapes=[x.shape for x in xs],
              k=k, mode=mode, qs=qs, device=xs[0].device)
        elif mode == '2k':
            A = get_mjp_fn(fn, *xs, chunk_size=self.chunk_size)
            AT = get_jmp_fn(fn, *xs, chunk_size=self.chunk_size)
            qs, qTJs = self.low_rank_approx(
              A, AT=AT, input_shapes=[x.shape for x in xs], k=k, mode=mode, qs=qs, device=xs[0].device)
            
        return qs, qTJs
      
    def low_rank_approx(self, A, input_shapes=None, output_shapes=None, AT=None, k=1, mode='2k', qs=None, device='cpu'):
        """
        Computes a low rank approximation of an arbitrary linear transformation A.
        
        Args:
            A:             a function that performs a matrix vector product, i.e. A(v) = Av
            AT:            a function that performs a vector matrix product, i.e., A(v) = v^T A (used in '2k')
            input_shapes:  the shape of v
            output_shapes: the shape of A(v)
            k:             the rank of the approximation
            mode:          the approximation strategy ('k' / '2k')
            device:        the device that all tensors should be on
            
        Returns:
            qs:  a low rank projection matrix of shape (fn(x).shape, k)
            qTJ: the low rank approximation of J of shape (fn(x).shape, k)
        """
        if mode == 'k':
            if qs is None:
                qs = tuple(get_Q(shape, k, device=device) for shape in output_shapes)
            if k > 0:
                qTAs = A(*qs)
            else:
                qs = tuple(get_Q(shape, 1, device=device) for shape in output_shapes)
                b, d_out = output_shapes[0]
                qTAs = tuple([torch.zeros(size=(b, 1, d_out), device=device) for _ in range(len(input_shapes))])
        elif mode == '2k':
            if qs is None:
                qs = tuple(get_Q(shape, k, device=device) for shape in input_shapes)
            for _ in range(self.rrf_iters - 1):
                qs = AT(qs)
                qTAs = A(qs)
                assert isinstance(qTAs, tuple)
                qs = tuple(q.mT for q in qTAs)
                
            qs = AT(qs)
            # always return a tuple
            if isinstance(qs, tuple):
                qs = tuple(torch.linalg.qr(q)[0] for q in qs)
                qTAs = A(qs)
            else:
                qs = torch.linalg.qr(qs)[0]
                qTAs = A(qs)
                qs = (qs,)
            
        return qs, qTAs
      
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
        (qv,), (qTVxx,) = self.lr_jacobian(Vx_fn, state.reshape(1, d_s), k=self.k_hessian, mode=self.lr_mode)
        qv, qTVxx = qv[0], qTVxx[0]
        Vxx_asym = qv @ qTVxx
        Vxx = Vxx_asym @ qv @ qv.mT

        # compute jacobians of the step function
        if self.lr_identity:
            I = torch.eye(d_s, device=state.device)
            fn = lambda x, u: self.env.step(self.env.timesteps[:-1], x, u) - (x + u)
        else:
            fn = lambda x, u: self.env.step(self.env.timesteps[:-1], x, u)

        (qs,), (qTfxs, qTfus) = self.lr_jacobian(fn, states[:-1], actions, k=self.k_jacobian, mode=self.lr_mode)
        
        # compute derivatives of the running cost
        lxs, lus, lxxs, luus, lxus, luxs = self.running_cost_derivatives(states[:-1], actions)

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
                fxs, fus = qs @ qTfxs + I.unsqueeze(0), qs @ qTfus + I.unsqueeze(0)
            else:
                fxs, fus = qs @ qTfxs, qs @ qTfus

            print(f"Shapes are fxs: {fxs.shape} and fus: {fus.shape}")
            print(f"shapes are fxs_debug: {fxs_true.shape}") 
            print_dist(fxs_true, fxs, "norm of difference between fxs: ")
            print_dist(fus_true, fus, "norm of difference between fus: ")
            print_dist(lxx_true, lxxs, "norm of difference between lxxs: ", k=self.k_hessian)
            print_dist(Vxx_true[None], Vxx[None], "norm of difference between Vxxs: ", k=self.k_hessian)

        ks = [None] * n_steps
        Ks = [None] * n_steps
        # DDP loop
        for t in range(n_steps - 1, -1, -1):
            state, action = states[t].clone(), actions[t].clone()

            # compute Qxx, Quu, Qux, Qxu
            lx, lu = lxs[t], lus[t]
            lxx, luu, lxu, lux = lxxs[t], luus[t], lxus[t], luxs[t]
            q, qTfx, qTfu = qs[t], qTfxs[t], qTfus[t]
            if self.lr_identity:
                fx, fu = q @ qTfx + I, q @ qTfu + I
            else:
                fx, fu = q @ qTfx, q @ qTfu

            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx
            Qxu = lxu + fx.T @ Vxx @ fu

            # the below would be equivalent to using the true jacobian without instantiating it
            step_fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
            fxT_Vx, fuT_Vx = torch.func.vjp(step_fn, state, action)[1](Vx.reshape(1, d_s))

            Qx = lx + fxT_Vx.reshape(d_s, 1)
            Qu = lu + fuT_Vx.reshape(d_c, 1)

            Quu_inv = inv(Quu, self.eps)

            k = -Quu_inv @ Qu # (control_dim, 1)
            K = -0.5 * Quu_inv @ (Qxu.mT + Qux) # (control_dim, state_dim+1)

            Vx = Qx - (K.mT @ Quu @ k)
            Vxx = Qxx - (K.mT @ Quu @ K)

            ks[t] = k.detach()
            Ks[t] = K.detach()
            
            # print('K norm', K.norm(), 'fx norm', fx.norm())

        # collected k and K
        ks = torch.stack(ks)
        Ks = torch.stack(Ks)
        
        return ks, Ks
    
    def iterate(self, init_state, actions, iter_num, states=None, last=False):
        '''
        A single iteration of DDP.
        
        Args:
            init_state: tensor of shape (1, state_dim) - initial state of DDP
            actions:    tensor of shape (num_steps - 1, control_dim) - initial actions
            iter_num:   int - for printing purposes
            last:       bool - for printing purposes
            
        Returns:
            actions:    tensor of shape (num_steps - 1, control_dim) - updated actions
            state:      tensor of shape (1, state_dim) - final state after current iteration of DDP
        '''
        with torch.no_grad():
            # generate trajectory w.r.t. nominal actions
            self.eps = np.clip(self.eps, a_min=self.min_eps, a_max=np.inf)
            if states is None:
                states = [init_state]
                for t, action in zip(self.env.timesteps[:-1], actions):
                    state = self.env.step(t, states[-1], action)
                    states.append(state)

                states = torch.stack(states, axis=0)

            # compute DDP gradients and feedback gains
            gradients = self.compute_gradients(states, actions)

            # update actions
            states, actions = self.update_actions(states, actions, *gradients, iter_num=iter_num, last=last)
            
        return states.detach(), actions.detach()
      
class SimpleDDP(DDP):
    """
    Maintain backwards compatibility with previous code.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
