import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        device : str = 'cuda',
        log_dir : str = './runs/',
        trace_dir : str = './traces/',
        profiler_steps : int = 50
    ) -> torch.nn.Module:
    """
    Обучает модель с профилированием производительности и логированием метрик.

    Функция выполняет обучение модели с одновременным профилированием операций,
    логированием метрик в TensorBoard и отслеживанием производительности.

    Parameters
    ----------
    model : torch.nn.Module
        Модель PyTorch для обучения
    train_loader : torch.utils.data.DataLoader
        DataLoader с тренировочными данными
    val_loader : torch.utils.data.DataLoader
        DataLoader с валидационными данными
    epochs : int, optional
        Количество эпох обучения
    lr : float, optional
        Learning rate для оптимизатора
    device : str, optional
        Устройство для выполнения вычислений ('cuda' или 'cpu')
    log_dir : str, optional
        Директория для сохранения логов TensorBoard
    trace_dir : str, optional
        Директория для сохранения трассировок профилировщика
    profiler_steps : int, optional
        Количество шагов для профилирования

    Returns
    -------
    torch.nn.Module
        Обученная модель
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    writer = SummaryWriter(log_dir=log_dir)

    with profiler.profile(
            schedule=profiler.schedule(wait=1, warmup=1, active=profiler_steps - 2, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                prof.step()

            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train/accuracy', 100 * correct / total, epoch * len(train_loader) + i)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'grad/{name}', param.grad, epoch * len(train_loader) + i)
                writer.add_histogram(f'weight/{name}', param, epoch * len(train_loader) + i)

            scheduler.step()

            # Валидация по эпохам
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            writer.add_scalar('val/loss', val_loss / len(val_loader), epoch)
            writer.add_scalar('val/accuracy', 100 * val_correct / val_total, epoch)
        writer.close()
    return model