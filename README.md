# Solving Inverse Problems with Diffusion Optimal Control [**NeurIPS 2024**]

[![Venue: NeurIPS 2024](https://img.shields.io/badge/Venue-NeurIPS%202024-blue.svg)](https://nips.cc/)

To run the code, please read the following instructions.

## Getting started 

### 1) Clone the repository

```
git clone [https://github.com/DPS2022/diffusion-posterior-sampling](https://github.com/lihenryhfl/diffusion_optimal_control)

cd diffusion_optimal_control
```

<br />

### 2) Download pretrained checkpoint from DPS
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/

<br />

Install dependencies

```
pip install -r requirements.txt
```

<br />

### 3) Inference

```
python3 inverse_sampling.py \
--dataset=ffhq \
--task=super_resolution
```

## Contact

For any questions, issues, or inquiries, please contact [henry.li@yale.edu](mailto:<henry.li@yale.edu) or [marcus.pereira@us.bosch.com](mailto:<marcus.pereira@us.bosch.com).

