import sys
import time
from torch import cat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from dataset import CycleGAN_Dataset
from CycleGAN import MViTCycleGANModel as MViTCycleGAN
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import torch
import os


# Data preprocessing
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  #Ensure image size is 256*256
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load image folders
content_dataset = CycleGAN_Dataset(img_path1="train", img_path2="domainA", transform=transform)
style_dataset = CycleGAN_Dataset(img_path1="train", img_path2="domainB", transform=transform)
val_content_dataset = CycleGAN_Dataset(img_path1="val", img_path2="domainA", transform=transform)
val_style_dataset = CycleGAN_Dataset(img_path1="val", img_path2="domainB", transform=transform)

# Create DataLoaders
loader_A = DataLoader(content_dataset, batch_size=5, shuffle=True, drop_last=True)
loader_B = DataLoader(style_dataset, batch_size=5, shuffle=True, drop_last=True)
val_loader_A = DataLoader(val_content_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader_B = DataLoader(val_style_dataset, batch_size=4, shuffle=True, drop_last=True)

# Create MViTCycleGAN instance
MViT_cycle_gan = MViTCycleGAN(3, 3)
MViT_cycle_gan.to(device)

# TensorBoard logging
writer = SummaryWriter(log_dir='runs/Stroke_MViT_cycle_gan')

#  Model checkpoint directory
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# Training loop + Visualization
# ----------------------------

num_epochs = 300
global_step = 0
for epoch in range(0, 300, 1):
    start_time = time.time()

    MViT_cycle_gan.train()

    for i, (a_batch, b_batch) in enumerate(zip(loader_A, cycle(loader_B))):
        input_data = {'A': a_batch['image'], 'B': b_batch['image']}
        MViT_cycle_gan.set_input(input_data)
        MViT_cycle_gan.optimize_parameters()

        # Get current losses
        losses = MViT_cycle_gan.get_current_losses()

        # Write losses to TensorBoard
        for key, value in losses.items():
            writer.add_scalar(f'Loss/{key}', value, global_step)
        global_step += 1

    if hasattr(MViT_cycle_gan, 'schedulers') and MViT_cycle_gan.schedulers:
        for scheduler in MViT_cycle_gan.schedulers:
            scheduler.step(epoch)

    def disable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.eval()
        if isinstance(m, torch.nn.Dropout2d):
            m.eval()

    # Apply to the entire CycleGAN model
    MViT_cycle_gan.apply(disable_dropout)
    if (epoch % 10 == 0):
        # Perform visualization every 10 epochs
        with torch.no_grad():
            count = 0
            max_batches = 4  # Control how many batches to display (each batch has 4 pairs)
            image_pairs_list = []

            for a_batch, b_batch in zip(val_loader_A, cycle(val_loader_B)):
                if count >= max_batches:
                    break

                input_data = {'A': a_batch['image'], 'B': b_batch['image']}
                MViT_cycle_gan.set_input(input_data)
                MViT_cycle_gan.forward()

                real_A = MViT_cycle_gan.real_A  # shape: [5, 3, 256, 256]
                fake_B = MViT_cycle_gan.fake_B  # shape: [5, 3, 256, 256]

                # Concatenate each pair horizontally (left-right)
                for img_real_A, img_fake_B in zip(real_A, fake_B):
                    image_pair = torch.cat((img_real_A, img_fake_B), dim=2)  # [3, 256, 512]
                    image_pairs_list.append(image_pair)

                count += 1

            all_image_pairs = torch.stack(image_pairs_list[:16], dim=0)  # [16, 3, 256, 512]

            #  Denormalize
            def denormalize(tensor):
                mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
                std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
                return tensor * std + mean

            all_image_pairs = denormalize(all_image_pairs)  # [16, 3, 256, 512]
            grid_image_pairs = make_grid(all_image_pairs, nrow=4, padding=2, normalize=False)
            writer.add_image('Validation/Image_Pairs', grid_image_pairs, global_step=epoch
                                                                                     + 1, dataformats='CHW')
    if(epoch% 50 ==0):
        with torch.no_grad():
            # Save model checkpoint every 50 epochs
            checkpoint_path = os.path.join(save_dir, f'Edge_Stroke_MViT_cycle_gan_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'netG_A_state_dict': MViT_cycle_gan.netG_A.state_dict(),
                'netG_B_state_dict': MViT_cycle_gan.netG_B.state_dict(),
                'netD_A_state_dict': MViT_cycle_gan.netD_A.state_dict(),
                'netD_B_state_dict': MViT_cycle_gan.netD_B.state_dict(),
                'optimizer_G_state_dict': MViT_cycle_gan.optimizer_G.state_dict(),
                'optimizer_D_state_dict': MViT_cycle_gan.optimizer_D.state_dict()
            }, checkpoint_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {elapsed_time:.2f} seconds")

writer.close()
