#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths of train dataset and output
train_data_dir = "/home/orin/SecurityData/train"
checkpoint_dir = "/home/orin/output2"
output_image_dir = "/home/orin/output2"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# Hyperparameters
latent_size = 512
hidden_size = 256
image_channels = 3
image_size = 256
num_epochs = 700
batch_size = 32
learning_rate = 0.0001 
betas = (0.0, 0.9)  
lambda_gp = 10  

# Transforming Images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

train_dataset = ImageDataset(train_data_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(discriminator, real_images, fake_images, device):
    batch_size, C, H, W = real_images.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)
    interpolated_scores = discriminator(interpolated_images)

    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# Normalization
def add_spectral_norm(layer):
    return nn.utils.spectral_norm(layer)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_channels):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 16),
            nn.ReLU(True)
        )
        self.upsample_blocks = nn.Sequential(
            nn.ConvTranspose2d(hidden_size * 16, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size, hidden_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size // 2, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.upsample_blocks(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, hidden_size, image_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            add_spectral_norm(nn.Conv2d(image_channels, hidden_size // 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            add_spectral_norm(nn.Conv2d(hidden_size // 2, hidden_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            add_spectral_norm(nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            add_spectral_norm(nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            add_spectral_norm(nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.model(x)


generator = Generator(latent_size, hidden_size, image_channels).to(device)
discriminator = Discriminator(hidden_size, image_channels).to(device)


generator.apply(weights_init)
discriminator.apply(weights_init)

# BCE Loss
criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

scaler = torch.cuda.amp.GradScaler()


g_losses = []
d_losses = []

# Training
for epoch in range(num_epochs):
    for i, real_images in enumerate(train_loader):
        real_images = real_images.to(device)

        # Train discriminator
        optimizer_D.zero_grad()
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(noise).detach()

        with torch.cuda.amp.autocast():
            real_outputs = discriminator(real_images)
            fake_outputs = discriminator(fake_images)
            d_real_loss = criterion(real_outputs, torch.ones_like(real_outputs) * 0.9)  
            d_fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs) * 0.1)  
            d_loss = d_real_loss + d_fake_loss
            gp = gradient_penalty(discriminator, real_images, fake_images, device)
            d_loss += lambda_gp * gp

        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_D)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  
        scaler.step(optimizer_D)
        scaler.update()

        # Train generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(noise)

        with torch.cuda.amp.autocast():
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, torch.ones_like(fake_outputs))

        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizer_G)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0) 
        scaler.step(optimizer_G)
        scaler.update()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Saving images after particular epochs
    if (epoch + 1) % 50 == 0:
        try:
            with torch.no_grad():
                noise = torch.randn(30, latent_size, 1, 1, device=device)
                synthetic_images = generator(noise)
                synthetic_images = (synthetic_images + 1) / 2  
                for j in range(synthetic_images.size(0)):
                    save_image(synthetic_images[j], os.path.join(output_image_dir, f'epoch_{epoch+1}_img_{j+1}.png'))
        except Exception as e:
            print(f"Error saving images at epoch {epoch + 1}: {e}")

   
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

# checkpoint to save
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, os.path.join(checkpoint_dir, 'final_checkpoint.pth'))

# Loss Curves plot
plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, len(g_losses) + 1), g_losses, label='Generator Loss')
plt.plot(np.arange(1, len(d_losses) + 1), d_losses, label='Discriminator Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(checkpoint_dir, 'training_curves.png'))
plt.show()

# Visualization
with torch.no_grad():
    noise = torch.randn(30, latent_size, 1, 1, device=device)
    synthetic_images = generator(noise)
    synthetic_images = (synthetic_images + 1) / 2  
    for j in range(synthetic_images.size(0)):
        save_image(synthetic_images[j], os.path.join(output_image_dir, f'final_img_{j + 1}.png'))

