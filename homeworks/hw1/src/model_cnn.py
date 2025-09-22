import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Простая сверточная нейронная сеть для классификации изображений.

    Архитектура сети: \n
    - Два сверточных слоя с ReLU активацией \n
    - MaxPooling после каждого сверточного слоя \n
    - Два полносвязных слоя \n

    Attributes
    ----------
    conv1 : nn.Conv2d
        Первый сверточный слой (3 входных канала, 32 выходных канала, ядро 3x3)
    conv2 : nn.Conv2d
        Второй сверточный слой (32 входных канала, 64 выходных канала, ядро 3x3)
    pool : nn.MaxPool2d
        Слой макс-пулинга с размером окна 2x2 и шагом 2
    fc1 : nn.Linear
        Первый полносвязный слой (64*8*8 входных features, 128 выходных)
    fc2 : nn.Linear
        Выходной полносвязный слой (128 входных features, num_classes выходных)
    relu : nn.ReLU
        Функция активации ReLU

    Parameters
    ----------
    num_classes : int, optional
        Количество выходных классов
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x