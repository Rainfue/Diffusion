# Импортирование библиотек
# -------------------------------------------------------
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from tqdm import tqdm

from model import load_model
from dataset import setup_dataloaders
from config import EPOCHS, DEVICE, PATIENCE, BATCH_SIZE, T
from func import get_loss, evaluate, sample_plot_image_tensorboard

# Дополнительные настройки
# -------------------------------------------------------
train_loader, val_loader = setup_dataloaders(
    augment_level='normal',  # или 'light'/'heavy'
    augment_factor=4,        # увеличить в 3 раза
    val_split=0.15           # 10% на валидацию
)

# Загрузка модели
model = load_model(r'best_model.pt')
model.to(DEVICE)
# Оптимизатор
optimizer = Adam(model.parameters(), lr=0.001)
# Объект для записи в tensorboard
writer = SummaryWriter(log_dir="runs/diffusion_rei")
global_step = 0

best_loss = float("inf")
best_val_loss = float("inf")
early_stopping_counter = 0
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

sampled_grid = sample_plot_image_tensorboard(model)
writer.add_image("Samples", sampled_grid, 0)
# Тренировочный цикл
# -------------------------------------------------------

for epoch in range(EPOCHS):
    model.train()
    epoch_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]", leave=True)
    epoch_loss = 0

    for step, batch in enumerate(epoch_bar):
        optimizer.zero_grad()
        images = batch[0].to(DEVICE)
        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE).long()
        loss = get_loss(model, images, t)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_step += 1
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        epoch_bar.set_postfix(loss=loss.item())

    # Средний лосс за эпоху
    avg_train_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

    # Валидация
    val_loss = evaluate(model, val_loader)
    writer.add_scalar("Loss/val", val_loss, epoch)
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}")

    # Scheduler update
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✓ Сохранили модель с новым лучшим Val Loss = {best_val_loss:.6f}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        print(f"✗ Val Loss не улучшился ({early_stopping_counter}/{PATIENCE})")
        if early_stopping_counter >= PATIENCE:
            print("⛔ Early stopping: модель больше не улучшается")
            break

    # Сэмплирование изображений
    if (epoch + 1) % 5 == 0:
        sampled_grid = sample_plot_image_tensorboard(model)
        writer.add_image("Samples", sampled_grid, epoch + 1)
