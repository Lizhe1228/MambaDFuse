# MambaDFuse

Official implementation for “MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion.”

## Update

## Citation

## Abstract

Multi-modality image fusion (MMIF) aims to integrate complementary information from different modalities into a single fused image to represent the imaging scene and facilitate downstream visual tasks comprehensively. In recent years, significant progress has been made in MMIF tasks due to advances in deep neural networks. However, existing methods cannot effectively and efficiently extract modality-specific and modality-fused features constrained by the inherent local reductive bias (CNN) or quadratic computational complexity (Transformers). To overcome this issue, we propose a Mamba-based Dual-phase Fusion (MambaDFuse) model. Firstly, a dual-level feature extractor is designed to capture long-range features from single-modality images by extracting low and highlevel features from CNN and Mamba blocks. Then, a dual-phase feature fusion module is proposed to obtain fusion features that combine complementary information from different modalities. It uses the channel exchange method for shallow fusion and the enhanced Multi-modal Mamba (M3) blocks for deep fusion. Finally, the fused image reconstruction module utilizes the inverse transformation of the feature extraction to generate the fused result. Through extensive experiments, our approach achieves promising fusion results in infrared-visible image fusion and medical image fusion. Additionally, in a unified benchmark, MambaDFuse has also demonstrated improved performance in downstream tasks such as object detection.
