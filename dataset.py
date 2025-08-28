
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms


def validate_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Invalid image: {file_path}, error: {e}")
        return False


class CycleGAN_Dataset(Dataset):
    def __init__(self, img_path1, img_path2, transform=None):
        image_folder = os.path.join(img_path1, img_path2)
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))
                            and validate_image(os.path.join(image_folder, f))
                            ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def ensure_rgb(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            image = self.ensure_rgb(image)

            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

        return {'image': image}