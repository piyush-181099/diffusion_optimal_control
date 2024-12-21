import os
import sys
import torch
import matplotlib.pyplot as plt

from utils.inverse import Experiment

device = 'cuda'
experiment = Experiment(device=device)

lpips = []
for i, ref_img in enumerate(experiment.loader):
    ref_img = ref_img.to(device)

    mask = experiment.get_mask(ref_img)

    # forward measurement model (y = Ax)
    y = experiment.operator.forward(ref_img, mask=mask)

    # measurement = y + noise
    measurement = experiment.noiser(y)

    env = experiment.get_env(target=measurement.reshape(len(measurement), -1), mask=mask)
    init_state = env.initialize_state(n=len(measurement))

    solver = experiment.get_solver(env)
    actions, states = solver.solve(
      init_state, num_iterations=experiment.num_iterations)
    sample = states[-1].reshape(-1, *experiment.shape)

    experiment.save_images(i, ref_img, sample, measurement)
    metrics = experiment.compute_metrics(ref_img, sample, states)

    lpips.append(metrics['lpips'])
    print(f"lpips: {metrics['lpips'].squeeze()}, {torch.stack(lpips).mean().item()}")

    plt.plot(metrics['lpips_over_iterations'].detach().cpu())
    plt.savefig(os.path.join(experiment.outdir, 'average_lpips_over_iterations.png'))
    plt.close()

    if (i + 1) % 10 == 0:
        print('', '*' * 40, '\n' * 2, f"Finished {i + 1} / {len(experiment.loader)} batches.", '\n' * 2, '*' * 40)
