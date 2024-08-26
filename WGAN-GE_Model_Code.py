#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for the train dataset and output
train_data_dir = "/home/orin/SecurityData/train"
output_dir = "/home/orin/output5"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

# Hyperparameters
latent_size = 128
hidden_size = 128
image_channels = 3
image_size = 256
num_epochs = 500
batch_size = 16
learning_rate = 0.0001
n_critic = 5
lambda_gp = 10  

# Transforming of Images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class ImageDataset(Dataset):
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator 
class Generator(nn.Module):
    def __init__(self, latent_size, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(latent_size, hidden_size * 16, 4, 1, 0, bias=False),  
            nn.BatchNorm2d(hidden_size * 16),
            nn.ReLU(True),
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
            nn.ConvTranspose2d(hidden_size, image_channels, 4, 2, 1, bias=False),  
            nn.Tanh(),
            nn.ConvTranspose2d(image_channels, image_channels, 4, 2, 1, bias=False),  
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_channels, hidden_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False))
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input).view(-1)

# GP 
def compute_gradient_penalty(discriminator, real_images, fake_images, device):
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates = interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


generator = Generator(latent_size, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)


optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.99)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.99)


g_losses = []
d_losses = []

# Training
for epoch in range(num_epochs):
    for i, real_images in enumerate(train_loader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        # discriminator training
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            noise = torch.randn(current_batch_size, latent_size, 1, 1, device=device)
            fake_images = generator(noise).detach()
            real_loss = -torch.mean(discriminator(real_images))
            fake_loss = torch.mean(discriminator(fake_images))
            
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, device)
            
            
            d_loss = real_loss + fake_loss + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

        # Generator training
        optimizer_G.zero_grad()
        noise = torch.randn(current_batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(noise)
        g_loss = -torch.mean(discriminator(fake_images))
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"{datetime.now()} - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    scheduler_G.step()
    scheduler_D.step()

    # Images are saving after every particular epochs
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            noise = torch.randn(30, latent_size, 1, 1, device=device)
            synthetic_images = generator(noise)
            synthetic_images = (synthetic_images + 1) / 2  
            for j in range(synthetic_images.size(0)):
                save_image(synthetic_images[j], os.path.join(output_dir, 'images', f'epoch_{epoch+1}_img_{j+1}.png'))

   
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    # loss curves can  be plotted after every particlar epoch
    if (epoch + 1) % 50 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="Generator Loss")
        plt.plot(d_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("WGAN Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"loss_curves_epoch_{epoch+1}.png"))
        plt.close()
                
# checkpoint
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, os.path.join(output_dir, 'final_checkpoint.pth'))

# Visualization
with torch.no_grad():
    noise = torch.randn(20, latent_size, 1, 1, device=device)
    synthetic_images = generator(noise)
    synthetic_images = (synthetic_images + 1) / 2  
    for j in range(synthetic_images.size(0)):
        save_image(synthetic_images[j], os.path.join(output_dir, 'images', f'final_img_{j+1}.png'))

    grid = make_grid(synthetic_images, nrow=8)
    plt.figure(figsize=(50, 50))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

