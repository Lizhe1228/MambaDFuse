# MambaDFuse

Official implementation for “MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion.”

## Update

- [2023-4] Codes and config files are public available.

## Citation

```tex
@misc{li2024mambadfuse,
      title={MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion}, 
      author={Zhe Li and Haiwei Pan and Kejia Zhang and Yuhua Wang and Fengming Yu},
      year={2024},
      eprint={2404.08406},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Abstract

Multi-modality image fusion (MMIF) aims to integrate complementary information from different modalities into a single fused image to represent the imaging scene and facilitate downstream visual tasks comprehensively. In recent years, significant progress has been made in MMIF tasks due to advances in deep neural networks. However, existing methods cannot effectively and efficiently extract modality-specific and modality-fused features constrained by the inherent local reductive bias (CNN) or quadratic computational complexity (Transformers). To overcome this issue, we propose a Mamba-based Dual-phase Fusion (MambaDFuse) model. Firstly, a dual-level feature extractor is designed to capture long-range features from single-modality images by extracting low and highlevel features from CNN and Mamba blocks. Then, a dual-phase feature fusion module is proposed to obtain fusion features that combine complementary information from different modalities. It uses the channel exchange method for shallow fusion and the enhanced Multi-modal Mamba (M3) blocks for deep fusion. Finally, the fused image reconstruction module utilizes the inverse transformation of the feature extraction to generate the fused result. Through extensive experiments, our approach achieves promising fusion results in infrared-visible image fusion and medical image fusion. Additionally, in a unified benchmark, MambaDFuse has also demonstrated improved performance in downstream tasks such as object detection.

## Usage

### 1.Network Architecture

Our MambaDFuse is implemented in `models/network.py`.

### 2.Main Environment

```bash
conda create -n MambaDFuse python=3.8.18
conda activate MambaDFuse
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

The .whl files of causal_conv1d and mamba_ssm could be found [here(Baidu)](https://pan.baidu.com/s/1SURk7jV1zfe-3QpZpbtljQ?pwd=mpc8). After installing the Mamba library, replace the mamba_simple.py file in the installation directory with the `./mamba_simple.py` in this repository. The implementation of the Multi-modal Mamba Block (M3 Block) is located in this file.

### 3.Checkpoint  Download 

### 4.Data Preparation

[MSRS](https://github.com/Linfeng-Tang/MSRS), [RoadScene](https://github.com/hanna-xu/RoadScene), [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), [Harvard medical dataset](http://www.med.harvard.edu/AANLIB/home.html)

Download the Infrared-Visible Fusion (IVF) and Medical Image Fusion (MIF) dataset and place the paired images in the folder `'./Datasets'`. Such as :

```
MambaDFuse
├── Datasets
│   ├── trainsets
│   │   ├── VIR
│   │   │   ├── VI_Y
│   │   │   ├── IR
│   │   ├── CT-MRI
│   │   │   ├── CT
│   │   │   ├── MRI
│   │   ├── PET-MRI
│   │   │   ├── PET_Y
│   │   │   ├── MRI
│   │   ├── SPECT-MRI
│   │   │   ├── SPECT_Y
│   │   │   ├── MRI
│   ├── valsets
│   │   ├── VIR
│   │   │   ├── VI_Y
│   │   │   ├── IR
│   │   ├── CT-MRI
│   │   │   ├── CT
│   │   │   ├── MRI
│   │   ├── PET-MRI
│   │   │   ├── PET_Y
│   │   │   ├── MRI
│   │   ├── SPECT-MRI
│   │   │   ├── SPECT_Y
│   │   │   ├── MRI
│   ├── testsets
│   │   ├── VIR
│   │   │   ├── VI_Y
│   │   │   ├── IR
│   │   ├── CT-MRI
│   │   │   ├── CT
│   │   │   ├── MRI
│   │   ├── PET-MRI
│   │   │   ├── PET_Y
│   │   │   ├── MRI
│   │   ├── SPECT-MRI
│   │   │   ├── SPECT_Y
│   │   │   ├── MRI

```

### 5.Training Examples

You may first modify the configuration file in the folder `./options/MambaDFuse`, such as `gpu_ids`, `path.root`, `dataroot_A`, `dataroot_B`, `dataloader_batch_size` and so on.

#### VIF

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train_MambaDFuse.py --opt options/MambaDFuse/train_mambadfuse_vif.json  --dist True
```

#### MIF

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train_MambaDFuse.py --opt options/MambaDFuse/train_mambadfuse_med.json  --dist True
```

### 6.Testing Examples

#### VIF

```
python test_MambaDFuse.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --iter_number=10000 --dataset=VIR --A_dir=IR  --B_dir=VI_Y
```

#### MIF

```
python test_MambaDFuse.py --model_path=./Model/Medical_Fusion-SPECT-MRI/Medical_Fusion/models/  --iter_number=10000 --dataset=SPECT-MRI --A_dir=MRI --B_dir=SPECT_Y
```

### 7.Evaluation and Pre-Processing

Please refer to [this link](https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion) and [this link](https://github.com/hanna-xu/utils).

## MambaDFuse

### Illustration of our MambaDFuse model.

![](./img/overall.png)

### Illustration of our Shallow Fuse Module.

![](./img/shallow.png)

### Illustration of our Deep Fuse Module.

![](./img/deep.png)

## Acknowledgement

The codes are heavily based on [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion). Please also follow their licenses. Thanks for their awesome works.
