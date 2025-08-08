# Импортирование библиотек
# -------------------------------------------------------
#
import torch
import torch.nn.functional as F
#
import torchvision
import torchvision.transforms as transforms
#
import matplotlib.pyplot as plt
import numpy as np

#
from config import T, IMG_SIZE, DEVICE

# Дополнительные значения
# -------------------------------------------------------
# 
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

#
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.-alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Функции
# -------------------------------------------------------
#
def get_index_from_list(vals, t, x_shape):
    '''
    Возвращает определенный индекс t передаваемого списка значений
    пока расчитывается батчевое пространство
    '''
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

#
def forward_diffusion_sample(x_0, t, device='cpu'):
    ''' 
    Берет изображение и шаг как инпут,
    и возвращает зашумленную версию
    '''

    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

#
def load_transformed_dataset(root: str = 'ddpm_from_scratch/images'):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2)-1)
    ])
    return torchvision.datasets.ImageFolder(root=root, transform=transform)

#
def show_tensor_images(image):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t+1) / 2),
        transforms.Lambda(lambda t: t.permute(1,2,0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()

    ])
    # Берем первое изображения из батча
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transform(image))

# Loss
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

#
@torch.no_grad()
def sample_timestep(x, t, model):
    ''' 
    Вызывает модель для предсказывания шума на изображении
    и возвращает разшумленное изображение
    Добавляет шум к фото, если это не последняя эпоха (шаг)
    '''
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(
        sqrt_recip_alphas, t, x.shape
    )
    # Вызываем модель (текущее фото - предсказанный шум)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x,t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

#
@torch.no_grad()
def sample_plot_image():
    # Сэмпл шума
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=DEVICE)
    plt.figure(figsize=(15,2))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_images(img.detach().cpu())
    plt.show()

#
def show_images(dataset, num_samples=20, cols=4):
    '''Plots samples from the dataset'''
    plt.figure(figsize=(15, 15))
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        # Получаем изображение и метку
        img_tensor, _ = dataset[i]
        
        # 1. Преобразуем тензор в numpy-массив
        # 2. Меняем порядок осей: (C, H, W) -> (H, W, C)
        # 3. Денормализуем изображение
        img = img_tensor.numpy().transpose(1, 2, 0)
        img = img * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        img = np.clip(img, 0, 1)  # Ограничиваем значения [0, 1]
        
        # Создаем подграфик
        plt.subplot(int(np.ceil(num_samples / cols)), cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#
@torch.no_grad()
def sample_plot_image_tensorboard(model):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=DEVICE)
    num_images = 10
    stepsize = int(T / num_images)

    all_imgs = []

    for i in range(T - 1, -1, -1):
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, t, model)
        img = torch.clamp(img, -1.0, 1.0)

        if i % stepsize == 0:
            # Подготовка изображения для TensorBoard (только первое изображение в батче)
            img_cpu = img[0].detach().cpu()
            img_cpu = (img_cpu + 1) / 2  # [-1, 1] -> [0, 1]
            all_imgs.append(img_cpu)

    # Объединяем картинки в один тензор (для 1x10 картинки)
    grid = torch.stack(all_imgs, dim=0)
    return torchvision.utils.make_grid(grid, nrow=len(all_imgs))

#
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        images = batch[0].to(DEVICE)
        t = torch.randint(0, T, (images.size(0),), device=DEVICE).long()
        loss = get_loss(model, images, t)
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    import os
    print(os.getcwd())
