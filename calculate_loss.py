import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PerceptualLossEvaluator(nn.Module):
    def __init__(self):
        super(PerceptualLossEvaluator, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.content_layer_idx = 21
        self.style_layer_indices = [1, 6, 11, 20, 29]

    def forward(self, x):
        content_features = []
        style_features = []
        tmp = x
        for idx, layer in enumerate(self.vgg):
            tmp = layer(tmp)
            if idx == self.content_layer_idx:
                content_features.append(tmp)
            if idx in self.style_layer_indices:
                style_features.append(tmp)
        return content_features, style_features


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def calculate_perceptual_loss(content_img_path, generated_img_path, evaluator, avg_gram_style_feats):
    def load_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(DEVICE)
        return img

    content_img = load_image(content_img_path)
    generated_img = load_image(generated_img_path)

    with torch.no_grad():
        gen_content_feats, gen_style_feats = evaluator(generated_img)
        cont_content_feats, _ = evaluator(content_img)

    content_loss = sum(torch.mean((gc - cc) ** 2) for gc, cc in zip(gen_content_feats, cont_content_feats))

    style_loss = sum(torch.mean((gram_matrix(gs) - avg_gs_sf) ** 2) for gs, avg_gs_sf in zip(gen_style_feats, avg_gram_style_feats))

    return content_loss.item(), style_loss.item()


def evaluate_model_with_avg_style(content_dir, generated_dir, style_ref_dir, output_csv='results.csv'):
    evaluator = PerceptualLossEvaluator().to(DEVICE)

    # Compute average style features from all images in style_ref_dir
    print(f"Loading style reference images from: {style_ref_dir}")
    avg_style_features = None
    count = 0

    for fname in sorted(os.listdir(style_ref_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        style_img_path = os.path.join(style_ref_dir, fname)
        style_img = transform(Image.open(style_img_path).convert('RGB')).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, style_style_feats = evaluator(style_img)
        gram_style_feats = [gram_matrix(sf) for sf in style_style_feats]

        if avg_style_features is None:
            avg_style_features = [gf.clone().detach() for gf in gram_style_feats]
        else:
            for i in range(len(avg_style_features)):
                avg_style_features[i] += gram_style_feats[i].clone().detach()

        count += 1

    # Compute average
    for i in range(len(avg_style_features)):
        avg_style_features[i] /= count

    results = []
    print(f"Scanning generated images directory: {generated_dir}")
    for fname in sorted(os.listdir(generated_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        stem = os.path.splitext(fname)[0]
        generated_path = os.path.join(generated_dir, fname)

        # Try multiple extensions to find the corresponding content image
        content_path = None
        for ext in ['.JPEG', '.jpg', '.jpeg', '.png']:
            path = os.path.join(content_dir, f"{stem}{ext}")
            if os.path.exists(path):
                content_path = path
                break

        if not content_path:
            print(f"Skipping {fname}: corresponding content image not found")
            continue

        try:
            c_loss, s_loss = calculate_perceptual_loss(content_path, generated_path, evaluator, avg_style_features)
            results.append({'image': fname, 'content_loss': c_loss, 'style_loss': s_loss})
            print(f"{fname}: content_loss={c_loss:.4f}, style_loss={s_loss:.4f}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    if len(results) == 0:
        print("Warning: No images were successfully processed. Please check paths and filenames.")
        df = pd.DataFrame(columns=['image', 'content_loss', 'style_loss'])
        df.to_csv(output_csv, index=False)
        return df

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation completed! Results saved to: {output_csv}")
    print(f"Average Content Loss: {df['content_loss'].mean():.4f}")
    print(f"Average Style Loss: {df['style_loss'].mean():.4f}")

    return df


if __name__ == "__main__":
    CONTENT_DIR = "val/domainA"
    GENERATED_DIR = "generated_images"
    STYLE_REF_DIR = "style_ref"
    OUTPUT_CSV = "perceptual_metrics.csv"

    df = evaluate_model_with_avg_style(CONTENT_DIR, GENERATED_DIR, STYLE_REF_DIR, OUTPUT_CSV)