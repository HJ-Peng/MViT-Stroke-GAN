# Stroke-MViT-GAN

🎨 **Stroke-MViT-GAN** is a sketch style transfer framework built on the CycleGAN architecture, enhanced with a **Mobile Vision Transformer (MobileViT)** backbone and a **stroke-aware module** for generating high-quality, hand-drawn-like pencil sketches from natural images.

Unlike standard CycleGANs with ResNet generators, our model leverages a U-Net-style generator enriched with global context modeling and directional stroke modeling, enabling better structural coherence and artistic expressiveness — all in an **unpaired training setting**.

![Sample Results](img/example.png) 

### 📝 Code Authorship & Attribution
This codebase is primarily developed by **[JiaPeng He](https://github.com/HJPeng)**, and supported by **[Xingchu Zhang](https://github.com/20031112)** and **[Yiming Xu](https://github.com/Ababaruaa)**.

While the core innovations — including the **MobileViT integration**, **stroke-aware module**, **custom loss design**, and **enhanced training pipeline** — are original, this project is built upon the official [CycleGAN PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Zhu et al. The following components are adapted or derived from their codebase:

- `networks.py`: Base generator (U-Net) and discriminator architectures
- `image_pool.py`: Image buffer for stabilizing discriminator training
- Core training/inference logic in `CycleGAN.py`
- Dataset loading and preprocessing patterns in `dataset.py`
We gratefully acknowledge the authors of CycleGAN for their open-source contribution, which greatly facilitated the development of this work

🔧 All modifications and new modules (e.g., Stroke.py, load_mobilevit.py,edge_loss.py) are authored by us.

## 📁 Code Structure

Below is a detailed overview of the key files and directories in this project:

### 📁 pretrained/
- **Purpose**: Contains pre-trained MobileViT weights downloaded from HuggingFace. These weights are used to initialize the MobileViT blocks in our model for better performance.
- **Files**:
  - `mobilevit_xxs-ad385b40.pth`: Pre-trained MobileViT weights.

### 📄 CycleGAN.py
- **Purpose**: This file retains parts of the original CycleGAN code but has been modified according to our needs. It serves as the core controller for model initialization, forward propagation, loss computation, and optimizer setup. It is central to both the training and inference processes.
- **Note**: Consider renaming this file to `StrokeMVitGAN.py` to better reflect its role in your specific implementation.

### 📄 Stroke.py
- **Purpose**: Implements the stroke module embedded within the generator. It extracts stroke features and calculates stroke losses between the original image and generated image, as well as between the sketch image and generated image. These losses are then weighted to form the total stroke loss.
  
### 📄 dataset.py
- **Purpose**: Handles dataset processing, including loading, preprocessing, and batching of data.

### 📄 edge_loss.py
- **Purpose**: Computes the edge loss, which helps preserve edges in the generated images.

### 📄 gradient_loss.py
- **Purpose**: Computes the gradient loss, which ensures fine details are aligned between the input and output images.

### 📄 image_pool.py
- **Purpose**: Contains the `ImagePool` class from the original CycleGAN code. This class implements an image buffer that stores generated images and updates the discriminator with historical images rather than just the latest ones. This enhances training stability and effectiveness by balancing the use of new and old images through a 50% probability replacement mechanism.

### 📄 load_mobilevit.py
- **Purpose**: Initializes and loads the MobileViT blocks with pre-trained weights. It also includes logic to freeze certain layers if necessary.

### 📄 networks.py
- **Purpose**: Integrates the stroke module and MobileViT module into the fixed 8-layer U-Net generator. This file is responsible for building the network architecture.

### 📄 test.py
- **Purpose**: The testing script. Note: For best results, it is strongly recommended to run inference in train mode (`model.train()`) with gradient updates turned off. Using `model.eval()` may lead to severe distortion in the generated images due to BatchNorm behavior.

### 📄 train.py
- **Purpose**: Defines the training process, including data loading, preprocessing, training loop, loss recording, visualization, and model checkpoint saving. Training progress can be monitored via TensorBoard, and models are saved periodically.


## How to Use

Follow these steps to train and test the Stroke-MViT-GAN model.

### 1. Install Dependencies

Make sure you have the following packages installed. We recommend using `conda` or `pip` in a virtual environment.

```bash
# Create and activate virtual environment (optional but recommended)
conda create -n stroke-mvit python=3.9
conda activate stroke-mvit

# Install core dependencies
pip install torch torchvision torchaudio
pip install pillow tensorboard

# Install timm (required for MobileViT)
pip install timm

# Optional: for image processing and visualization
pip install opencv-python matplotlib
```

> 🔔 Important: The timm library is required to build the MobileViT backbone. Without it, model initialization will fail.

### 2. Prepare Your Dataset

Organize your dataset in the following structure:

```text
project_root/
├── train/
│   ├── domainA/     # Natural images (content domain)
│   └── domainB/     # Sketch images (style domain)
└── val/
    ├── domainA/     # Validation natural images
    └── domainB/     # Validation sketch images
```

- All images should be RGB and preferably resized to 256x256 (or close).
- Supported formats: .jpg, .png, etc.
- No need to manually resize — the model handles it via transforms.

### 3. Start Training

Run the training script:

```bash
python train.py
```

Key Notes on Training:
- Total epochs: Set to 300 by default, but we recommend monitoring validation outputs and adjusting dynamically.
- Model checkpoints are saved every 50 epochs in the checkpoints/ folder.
- Visualization is logged to TensorBoard every 10 epochs using validation data.
- Dropout is disabled during training to ensure consistency with inference behavior.

> 💡 Pro Tip: Use TensorBoard to monitor generated image quality and loss trends:
>
> ```bash
> tensorboard --logdir=runs
> ```

### 4. Run Inference (Testing)

After training, use `test.py` to generate sketches from new images.

> ⚠️ Critical Note:
>
> Do NOT use `model.eval()` mode for inference. Due to BatchNorm layers, calling `eval()` causes severe artifacts or distortion in output images.
>
> Instead, keep the model in `train()` mode and disable gradients:
>
> ```python
> model.train()
> with torch.no_grad():
>     output = model(input)
> ```

#### How to Run
1. Place your test images in a folder (e.g., `test_images/`).
2. Open `test.py` and manually set:
   - Input image path or directory
   - Output save path
   - Model checkpoint path (e.g., `checkpoints/Stroke_MViTGAN.pth`)
3. Run:
   ```bash
   python test.py
---

## ⚙️ Training Configuration (From Paper)

The model is trained with the following settings:

- **Framework**: PyTorch
- **Batch size**: 5
- **Image size**: 256 × 256
- **Optimizer**: Adam (β₁ = 0.5, β₂ = 0.999)
- **Learning rate**: 2 × 10⁻⁴, constant for first 200 epochs, then linearly decayed
- **Total epochs**: ~250 (adjusted based on validation performance)

### Generator Architecture
- Two MobileViT blocks (stem + Stage 0–2) embedded in U-Net.
- Stage 0 and Stage 1 are **frozen** to reduce cost and stabilize training.
- Stage 2 and other layers are **fine-tuned end-to-end**.

### Loss Weights
| Loss Type               | Weight |
|-------------------------|--------|
| Adversarial (λadv)      | 1.0    |
| Cycle Consistency (λcyc)| 10.0   |
| Identity (λidt)         | 0.5    |
| Edge-Aware (λedge)      | 2.0    |
| Gradient (λgrad)        | 0.8    |
| Stroke Consistency (λstroke) | 1.0 |

> **Stroke Loss Internal Weights**:
> - Structural loss (λstruct): 3.0
> - Stylistic loss (λstyle): 2.0


## 📦 Pretrained Weights

We have released our trained model checkpoints on Hugging Face for easy access and reuse.

🔗 **Download from Hugging Face**:  
👉 [https://huggingface.co/HJPeng/Stroke_MViTGAN](https://huggingface.co/HJPeng/Stroke_MViTGAN)

### Available Checkpoints
- `Stroke_MViTGAN.pth`: Trained model (approximately 250 epochs, recommended for inference).

### How to Use
1. Go to the [Hugging Face model page](https://huggingface.co/HJPeng/Stroke_MViTGAN) and download the `.pth` file.
2. Place it in the `checkpoints/` directory.
3. Open `test.py` and **manually set the model path**.

## 📚 Citation

If you find our work or dataset useful in your research, please cite us as follows:

For now, we provide a placeholder BibTeX entry. An updated citation will be provided upon the publication of the paper.

```bibtex
@misc{stroke-mvit-gan2025,
  title={},
  author={},
  year={2025},
  note={Work in progress}
}
```
This project is built upon the CycleGAN framework. Please also cite the original CycleGAN paper if you use this codebase:

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```
🔗 Original Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
