# Импортирование библиотек
# -------------------------------------------------------
from torch import device
from torch.cuda import is_available

# Конфигурация
# -------------------------------------------------------
T = 300
BATCH_SIZE = 32
IMG_SIZE = 128
EPOCHS = 50
PATIENCE = 10
DEVICE = device('cuda' if is_available() else 'cpu')