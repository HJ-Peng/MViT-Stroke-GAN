import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataset import CycleGAN_Dataset
from CycleGAN import MViTCycleGANModel as MViTCycleGAN
from torch.utils.tensorboard import SummaryWriter
import torch
import os

"""
# Important: Run the model in train mode (model.train()). Some components require training mode to function correctly.
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_content_dataset = CycleGAN_Dataset(img_path1="val", img_path2="domainA", transform=transform)

"""
Important: batch_size >= 2 and drop_last=True to avoid the last batch having size 1
"""
test_loader_A = DataLoader(
    test_content_dataset,
    batch_size=5,
    shuffle=False,
    drop_last=True
)


MViT_cycle_gan = MViTCycleGAN(3, 3)
MViT_cycle_gan.to(device)

# Directory to save model checkpoints
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# Load pretrained model weights
# TODO: Replace 'path/your_model.pth' with the actual path to your trained model checkpoint
savepoint = torch.load("path/your_model.pth", map_location=device)

MViT_cycle_gan.netG_A.load_state_dict(savepoint['netG_A_state_dict'])
MViT_cycle_gan.netG_B.load_state_dict(savepoint['netG_B_state_dict'])
MViT_cycle_gan.netD_A.load_state_dict(savepoint['netD_A_state_dict'])
MViT_cycle_gan.netD_B.load_state_dict(savepoint['netD_B_state_dict'])

#  Disable Dropout: Prevent the introduction of randomness when in train() mode
def disable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.eval()  # 强制关闭 Dropout
    if isinstance(m, torch.nn.Dropout2d):
        m.eval()

MViT_cycle_gan.apply(disable_dropout)
# Note: The model remains in train() mode by default and eval() is not called


# ----------------------------
# Inference: Save only the generated images, naming them sequentially as 1.png, 2.png, ...
# ----------------------------
with torch.no_grad():
    #  Create a folder for saving generated images
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)

    image_counter = 1  # Start naming from 1

    for i, a_batch in enumerate(test_loader_A):
        batch_real_A = a_batch['image'].to(device)
        batch_fake_B = MViT_cycle_gan.netG_A(batch_real_A)

        def denormalize(tensor):
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
            return tensor * std + mean

        batch_fake_B = denormalize(batch_fake_B)

        for j in range(batch_fake_B.size(0)):
            if image_counter > 10000:
                break
            fake_img = batch_fake_B[j:j+1]  # [1, 3, H, W]
            save_path = os.path.join(output_folder, f"{image_counter}.JPEG")
            save_image(fake_img, save_path, normalize=False)
            image_counter += 1

    print(f"Saved {image_counter - 1} generated images to '{output_folder}' folder.")