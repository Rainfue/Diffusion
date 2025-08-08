# Импортирование библиотек
# -------------------------------------------------------
from math import log

import torch
from torch import nn

# Архитектура модели
# -------------------------------------------------------
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Первый слой
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Временной эмбеддинг
        time_emb = self.relu(self.time_mlp(t))
        # Расширяем последние 2 измерения
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Добавляем канал времени
        h = h + time_emb
        # Второй слой
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)
    
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings
    

class SimpleUnet(nn.Module):
    ''' 
    Упрощенный вариант архитектуры U-Net
    '''
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Эмбеддинг времени
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Инициация проекции
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], 
                                          time_emb_dim) 
                                    for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], 
                                          time_emb_dim, up=True) 
                                    for i in range(len(down_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    
def load_model(root: str = None):
    model = SimpleUnet()
    # Если есть путь к уже обученным весам
    if root:
        model.load_state_dict(torch.load(root))
        return model
    # Если весов нет
    else:
        return model

if __name__ == '__main__':

    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Num params: {sum(p.numel() for p in model.parameters())}')
    print(model)
        
