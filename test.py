# %%
# =============================== #
#        Импорт библиотек        #
# =============================== #
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# =============================== #
#        Конфигурация            #
# =============================== #
T = 1000  # шагов диффузии
EPOCHS = 500
BATCH_SIZE = 128
LR = 1e-3
SAVE_PATH = './ddpm_mnist500.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# =============================== #
#      Расчёт расписания β_t     #
# =============================== #
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    return betas

betas = get_beta_schedule(T).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# %%
# =============================== #
#         Time embedding          #
# =============================== #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# %%
# =============================== #
#            Модель              #
# =============================== #
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(32),
            nn.Linear(32, 128),
            nn.ReLU()
        )
        self.time_proj = nn.Linear(128, 64)

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        self.bot = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU()
        )
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        t = self.time_proj(t)
        t = t.unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x2 = x2 + t  # добавляем time embedding
        x3 = self.bot(x2)
        x4 = self.up1(x3)
        return self.out(x4 + x1)

# %%
# =============================== #
#        Обработка данных        #
# =============================== #
transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x * 2 - 1  # [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
# =============================== #
#         Функции диффузии       #
# =============================== #
@torch.no_grad()
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

def get_loss(model, x0, t, noise):
    xt = q_sample(x0, t, noise)
    pred_noise = model(xt, t)
    return F.mse_loss(pred_noise, noise)

# %%
# =============================== #
#         Обучение модели        #
# =============================== #
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x0, _ in pbar:
        x0 = x0.to(DEVICE)
        t = torch.randint(0, T, (x0.size(0),), device=DEVICE).long()
        noise = torch.randn_like(x0)

        loss = get_loss(model, x0, t, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

# %%
# =============================== #
#     Сохранение/загрузка модели #
# =============================== #
torch.save(model.state_dict(), SAVE_PATH)

# %%
# Загрузка:
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

# %%
# =============================== #
#          Сэмплирование         #
# =============================== #
@torch.no_grad()
def sample_ddpm(model, n_samples, img_size, T, device):
    model.eval()
    x = torch.randn(n_samples, 1, img_size, img_size).to(device)
    betas = get_beta_schedule(T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for t in reversed(range(1, T)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        pred_noise = model(x, t_tensor)
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)

        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = mean + torch.sqrt(beta_t) * noise

    return x

# %%
# =============================== #
#         Визуализация           #
# =============================== #
samples = sample_ddpm(model, n_samples=16, img_size=28, T=T, device=DEVICE)
samples = samples.cpu().clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]

fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(samples[i, 0], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('results_500ep.png')

# %%
