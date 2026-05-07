# Brain Tumor Segmentation using Attention U-Net

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brain-tumor-segmentation-attention-unet-ae9ijbms3cktrws6asgu3t.streamlit.app/)

This project implements a brain tumor segmentation pipeline for MRI scans using an Attention U-Net built from scratch in PyTorch. The model is designed to segment tumor regions from FLAIR MRI slices by improving the standard U-Net skip-connection pathway with learnable attention gates.

In a standard U-Net, encoder features are passed directly to the decoder through skip connections. While this helps recover spatial detail, it also forwards background activations that may not be relevant to the tumor region. This implementation adds attention gates at each skip connection so the decoder receives spatially filtered encoder features, allowing the network to focus more strongly on tumor-relevant regions before mask reconstruction.

The project includes training code, dataset handling, augmentation, evaluation metrics, and a deployed Streamlit application for real-time inference.

## Live Demo

Streamlit app: https://brain-tumor-segmentation-attention-unet-ae9ijbms3cktrws6asgu3t.streamlit.app/

The app accepts a brain MRI image and returns:

- Original MRI input
- Predicted binary tumor mask
- Overlay visualization with the segmented region highlighted
- Probability heatmap showing spatial confidence
- Tumor coverage percentage
- Maximum prediction confidence
- Average confidence inside the detected region
- Adjustable threshold control for sensitivity tuning

## Project Summary

| Item | Details |
| --- | --- |
| Architecture | Attention U-Net |
| Parameters | 31,388,201 |
| Dataset | LGG Brain MRI Segmentation dataset |
| Patients | 110 lower-grade glioma patients |
| Image-mask pairs | 3,929 |
| Loss | Dice Loss + BCEWithLogitsLoss |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 8 |
| Training | 50 epochs on a T4 GPU |
| Best checkpoint | Epoch 44 |
| Test Dice | 0.8207 |
| Test IoU | 0.7903 |
| Deployment | Streamlit Community Cloud, CPU inference |

## Model Architecture

The model follows an Attention U-Net design with an encoder, bottleneck, decoder, and attention-gated skip connections.

The encoder contains four convolutional blocks. Each block uses two Conv2D layers followed by BatchNorm and ReLU activation, with MaxPooling used for downsampling. The bottleneck operates at 1024 channels.

The decoder mirrors the encoder using transposed convolutions for upsampling. At each decoder level, an attention gate receives:

- the decoder gating signal
- the corresponding encoder skip feature map

The gate computes a soft spatial attention map using a sigmoid activation and multiplies it element-wise with the encoder features before concatenation. This helps suppress irrelevant background responses and emphasizes spatial regions that are more useful for tumor segmentation.

## Dataset

The model was trained on the LGG Brain MRI Segmentation dataset published by Mateusz Buda et al. and available on Kaggle. The dataset contains FLAIR MRI slices from 110 lower-grade glioma patients sourced from The Cancer Genome Atlas.

Each patient folder contains MRI slices in `.tif` format paired with binary tumor masks. The dataset was shuffled with a fixed seed and split at the slice level:

- 70% training
- 15% validation
- 15% testing

## Training

The training pipeline uses a combined Dice Loss and BCEWithLogitsLoss objective. Dice Loss directly optimizes mask overlap, while BCEWithLogitsLoss provides pixel-level supervision. This combination is useful for tumor segmentation because the foreground tumor region is usually much smaller than the healthy/background region.

Training configuration:

- 50 epochs
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: 3e-4
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR
- Hardware: T4 GPU on Google Colab

The best model checkpoint was selected using validation Dice score and was saved at epoch 44.

## Augmentation

Training augmentations were implemented with Albumentations and applied only to the training set:

- Horizontal flip
- Vertical flip
- Random 90-degree rotation
- Elastic transform
- Grid distortion
- Random brightness and contrast adjustment

Validation and test data used resizing and normalization only.

## Results

On the held-out test set, the model achieved:

| Metric | Score |
| --- | ---: |
| Dice | 0.8207 |
| IoU | 0.7903 |

The model produces segmentation masks that capture tumor presence and irregular boundaries across a range of tumor sizes and shapes. These results are in a competitive range for Attention U-Net-style approaches on this dataset, depending on training setup, augmentation strategy, and evaluation protocol.

## Deployment

The Streamlit app runs inference on CPU and loads trained model weights at startup using `gdown` from Google Drive. This avoids the need for a dedicated GPU server while keeping the demo accessible from a browser.

The app provides four visual outputs side by side:

- MRI input
- Binary segmentation mask
- Overlay mask
- Probability heatmap

## Tech Stack

- PyTorch
- Albumentations
- OpenCV
- NumPy
- Streamlit
- gdown
- Google Drive
- Google Colab

## Repository Structure

```text
.
├── src/
│   ├── dataset loading
│   ├── model architecture
│   ├── loss functions
│   ├── metrics
│   └── training loop
├── app.py
├── requirements.txt
└── README.md
```

## Important Note

This project is for research, learning, and demonstration purposes only. It is not a clinical diagnostic tool and should not be used for medical decision-making.

## Author

Ankush Jha  
BS, IIT Patna  
GitHub: [@ankushjha556](https://github.com/ankushjha556)
