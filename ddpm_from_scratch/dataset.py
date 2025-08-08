# Импортирование библиотек
# -------------------------------------------------------
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import ImageFile
import torch

# Игнорировать обрезанные изображения
ImageFile.LOAD_TRUNCATED_IMAGES = True  

from func import show_images
from config import BATCH_SIZE, IMG_SIZE

# Аугментации для диффузионных моделей
# -------------------------------------------------------

def get_train_transforms():
    """
    Аугментации для диффузионных моделей - БЕЗ RandomErasing!
    Фокус на сохранении естественного распределения изображений
    """
    return transforms.Compose([
        # Базовые геометрические преобразования
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Немного больше для кропа
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),        # Случайный кроп
        transforms.RandomHorizontalFlip(p=0.5),            # Горизонтальное отражение
        
        # Осторожные повороты (важно для сохранения естественности)
        transforms.RandomRotation(degrees=5, fill=0),      # Совсем небольшие повороты
        
        # Мягкие цветовые аугментации
        transforms.ColorJitter(
            brightness=0.05,   # Очень мягко ±5%
            contrast=0.05,     # Очень мягко ±5%
            saturation=0.05,   # Очень мягко ±5%
            hue=0.02          # Минимальные изменения оттенка
        ),
        
        # Совсем лёгкие перспективные искажения
        transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
        
        # Конвертируем в тензор и нормализуем
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1] нормализация
    ])

def get_val_transforms():
    """
    Преобразования для валидационного набора (без аугментации)
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def get_heavy_augmentation():
    """
    Агрессивные аугментации для диффузии - БЕЗ RandomErasing и Blur!
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Умеренные повороты (не больше 15 градусов)
        transforms.RandomRotation(degrees=15, fill=0),
        
        # Более заметные цветовые изменения, но всё ещё разумные
        transforms.ColorJitter(
            brightness=0.15,   # ±15%
            contrast=0.15,     # ±15%
            saturation=0.15,   # ±15%
            hue=0.05          # ±5%
        ),
        
        # Лёгкие аффинные преобразования
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),  # Уменьшили сдвиг
            scale=(0.95, 1.05),      # Минимальное масштабирование
            shear=3                  # Уменьшили сдвиг
        ),
        
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        
        # Конвертируем в тензор
        transforms.ToTensor(),
        
        # БЕЗ RandomErasing и GaussianBlur!
        
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

class CustomImageDataset:
    """
    Кастомный датасет с возможностью применения разных аугментаций
    """
    def __init__(self, dataset, transform=None, augment_factor=1):
        self.dataset = dataset
        self.transform = transform
        self.augment_factor = augment_factor  # Сколько раз увеличить датасет
        
    def __len__(self):
        return len(self.dataset) * self.augment_factor
        
    def __getitem__(self, idx):
        # Получаем оригинальный индекс
        original_idx = idx % len(self.dataset)
        img, label = self.dataset[original_idx]
        
        # Если есть кастомные преобразования, применяем их
        if self.transform:
            img = self.transform(img)
            
        return img, label

def load_transformed_dataset(root: str = 'images', 
                           augment_level: str = 'normal'):
    """
    Загружает датасет с выбранным уровнем аугментации
    
    Args:
        root: путь к папке с изображениями
        augment_level: 'light', 'normal', 'heavy'
    """
    import torchvision.datasets as datasets
    
    # Выбираем уровень аугментации
    if augment_level == 'light':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    elif augment_level == 'normal':
        transform = get_train_transforms()
    elif augment_level == 'heavy':
        transform = get_heavy_augmentation()
    else:
        # Без аугментации
        transform = get_val_transforms()
    
    return datasets.ImageFolder(root=root, transform=transform)

# Настройка датасета с аугментацией
# -------------------------------------------------------

def setup_dataloaders(augment_level='normal', augment_factor=1, val_split=0.1):
    """
    Создает DataLoader'ы с аугментацией
    
    Args:
        augment_level: уровень аугментации ('light', 'normal', 'heavy', 'none')
        augment_factor: во сколько раз увеличить тренировочный датасет
        val_split: доля валидационной выборки
    """
    
    # Загружаем базовый датасет без преобразований
    import torchvision.datasets as datasets
    base_dataset = datasets.ImageFolder(root='images')
    
    print(f"Оригинальный размер датасета: {len(base_dataset)} изображений")
    
    # Разделяем на train/val
    val_size = int(val_split * len(base_dataset))
    train_size = len(base_dataset) - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(base_dataset)))
    
    # Создаем подвыборки
    train_subset = torch.utils.data.Subset(base_dataset, train_indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices)
    
    # Применяем разные трансформации
    if augment_level == 'light':
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    elif augment_level == 'normal':
        train_transform = get_train_transforms()
    elif augment_level == 'heavy':
        train_transform = get_heavy_augmentation()
    else:
        train_transform = get_val_transforms()
    
    val_transform = get_val_transforms()
    
    # Создаем кастомные датасеты с аугментацией
    train_dataset = CustomImageDataset(train_subset, train_transform, augment_factor)
    val_dataset = CustomImageDataset(val_subset, val_transform, 1)  # Валидация без увеличения
    
    print(f"Тренировочный датасет после аугментации: {len(train_dataset)} изображений")
    print(f"Валидационный датасет: {len(val_dataset)} изображений")
    print(f"Коэффициент увеличения: x{augment_factor}")
    
    # Создаем DataLoader'ы (оптимизированные для CPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=False,  # False для CPU
        num_workers=0      # 0 для CPU (избегаем проблем с multiprocessing)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=False,  # False для CPU
        num_workers=0      # 0 для CPU
    )
    
    return train_loader, val_loader

# Инициализация (выбери нужный уровень)
# -------------------------------------------------------

# Для маленького датасета (< 1000 изображений)
if __name__ == '__main__':
    # Вариант 1: Умеренная аугментация
    train_loader, val_loader = setup_dataloaders(
        augment_level='normal',
        augment_factor=4,  # Увеличиваем в 3 раза
        val_split=0.15
    )
    
    # Вариант 2: Агрессивная аугментация для очень маленького датасета
    # train_loader, val_loader = setup_dataloaders(
    #     augment_level='heavy',
    #     augment_factor=5,  # Увеличиваем в 5 раз
    #     val_split=0.1
    # )
    
    print("Dataloaders созданы!")
    
    # Показываем примеры аугментированных изображений
    dataiter = iter(train_loader)
    sample_batch, _ = next(dataiter)
    
    # Создаем фейковый датасет для показа
    class SampleDataset:
        def __init__(self, batch):
            self.batch = batch
        def __len__(self):
            return len(self.batch)
        def __getitem__(self, idx):
            return self.batch[idx], 0
    
    sample_dataset = SampleDataset(sample_batch)
    show_images(sample_dataset, num_samples=16, cols=4)