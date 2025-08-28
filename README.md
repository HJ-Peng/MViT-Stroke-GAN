# Stroke-MViT-GAN

🎨 **Stroke-MViT-GAN** is a sketch style transfer framework built on the CycleGAN architecture, enhanced with a **Mobile Vision Transformer (MobileViT)** backbone and a **stroke-aware module** for generating high-quality, hand-drawn-like pencil sketches from natural images.

Unlike standard CycleGANs with ResNet generators, our model leverages a U-Net-style generator enriched with global context modeling and directional stroke modeling, enabling better structural coherence and artistic expressiveness — all in an **unpaired training setting**.

![Sample Results](assets/output.jpg)  <!-- 请替换为你的效果图 -->

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


How to Use

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

After training, use test.py to generate sketches from new images:

```bash
python test.py --input_dir ./test_images --output_dir ./results --checkpoint checkpoints/Edge_Stroke_MViT_cycle_gan_epoch_250.pth
```

> ⚠️ Critical Note:
>
> Do NOT use model.eval() mode for inference. Due to BatchNorm layers, calling eval() causes severe artifacts or distortion in output images.
>
> Instead, keep the model in train() mode and disable gradients:
>
> ```python
> model.train()
> with torch.no_grad():
>     output = model(input)
> ```

---

