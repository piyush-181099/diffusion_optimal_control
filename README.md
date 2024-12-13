# Solving Inverse Problems with Diffusion Optimal Control [**NeurIPS 2024**]

[![Venue: NeurIPS 2024](https://img.shields.io/badge/Venue-NeurIPS%202024-blue.svg)](https://nips.cc/)

To run the code, please read the following instructions.

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/DPS2022/diffusion-posterior-sampling

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

:speaker: Checkpoint for imagenet is uploaded.

<br />

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<br />

### 4) Inference

```
python3 inverse_sampling.py \
--dataset='ffhq' \
--algorithm='ddp'
```

## Contact

For any questions, issues, or inquiries, please contact [henry.li@yale.edu](mailto:<henry.li@yale.edu) or [marcus.pereira@us.bosch.com](mailto:<marcus.pereira@us.bosch.com).

