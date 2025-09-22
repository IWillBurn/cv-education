import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

def get_dataloaders(
        batch_size : int = 32,
        part_ratio : float = 1,
        val_ratio : float = 0.1,
        image_size : int = 32,
        cifar_norm : bool = True
    ) -> tuple:
    """
    Создает DataLoader'ы для обучения и валидации на основе CIFAR-10 датасета.

    Функция загружает CIFAR-10 датасет, применяет аугментации, нормализацию и разделяет
    данные на обучающую и валидационную выборки с сохранением баланса классов.

    Parameters
    ----------
    batch_size : int, optional
       Размер батча для DataLoader'ов
    part_ratio : float, optional
       Доля всего датасета для использования (от 0.0 до 1.0)
    val_ratio : float, optional
       Доля данных для валидации от общего количества (от 0.0 до 1.0)
    image_size : int, optional
       Размер изображения после ресайза
    cifar_norm : bool, optional
       Использовать ли нормализацию для CIFAR

    Returns
    -------
    tuple
       Кортеж из трех элементов:
       - trainloader : DataLoader
           DataLoader для обучающей выборки с перемешиванием
       - valloader : DataLoader
           DataLoader для валидационной выборки без перемешивания
       - classes : list
           Список названий классов CIFAR-10
    """

    if cifar_norm:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_indices = []
    val_indices = []
    for cls in range(10):
        cls_indices = np.where(np.array(full_trainset.targets) == cls)[0]
        to_train = int(len(cls_indices) * part_ratio * (1 - val_ratio))
        end_val = int(len(cls_indices) * part_ratio)
        train_indices.extend(cls_indices[:to_train])
        val_indices.extend(cls_indices[to_train:end_val])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_set = Subset(full_trainset, train_indices)
    val_set = Subset(full_trainset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_trainset.classes