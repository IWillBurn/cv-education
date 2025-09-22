import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv


def sanity_check(
        model : torch.nn.Module,
        model_name : str,
        train_loader : torch.utils.data.DataLoader,
        classes : list,
        device : str = 'cuda'
    ) -> None:
    """
    Выполняет расширенный sanity check модели с переобучением и последующей оценкой.

    Функция проверяет способность модели к обучению путем переобучения на небольшом
    наборе данных.

    Parameters
    ----------
    model : torch.nn.Module
        Модель PyTorch для проверки
    model_name : str
        Название модели (используется для именования результатов оценки)
    train_loader : torch.utils.data.DataLoader
        DataLoader с тренировочными данными
    classes : list
        Список названий классов для оценки модели
    device : str, optional
        Устройство для выполнения вычислений ('cuda' или 'cpu')

    Returns
    -------
    None
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batches = [next(iter(train_loader)) for _ in range(2)]
    print(f'Sanity-check: Overfitting on {len(batches)} batches ({len(batches[0][0])} samples each)...')

    for epoch in range(20):  # Больше эпох для гарантированного overfit
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc = 0

        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            for inputs, labels in batches:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print(f'Sanity epoch {epoch}: loss {total_loss / len(batches):.3f}, acc {acc:.2f}%')

        if acc > 99:
            break

    evaluate_model(model, f"{model_name}-overfit", batches, device, classes)

    print('Sanity-check completed. Ideal: loss ~0 and acc ~100%.')


def evaluate_model(model, model_name, dataloader, device='cuda', classes=None):
    """
    Оценивает производительность модели на заданном DataLoader и сохраняет результаты.

    Функция вычисляет метрики качества модели (accuracy и macro-F1), строит и сохраняет
    матрицу ошибок (confusion matrix) в виде тепловой карты.

    Parameters
    ----------
    model : torch.nn.Module
        Модель PyTorch для оценки
    model_name : str
        Название модели (используется для именования файлов результатов)
    dataloader : torch.utils.data.DataLoader
        DataLoader с данными для оценки модели
    device : str, optional
        Устройство для выполнения вычислений ('cuda' или 'cpu'), по умолчанию 'cuda'
    classes : list, optional
        Список названий классов для подписей матрицы ошибок, по умолчанию None

    Returns
    -------
    tuple
        Кортеж из трех элементов:
        - accuracy : float
            Точность модели на оценочной выборке
        - macro_f1 : float
            Macro-F1 score модели на оценочной выборке
        - confusion_matrix : numpy.ndarray
            Матрица ошибок формы (n_classes, n_classes)
    """
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.savefig(f'./results/confusion_matrix_{model_name}.png')
    plt.close()

    return acc, f1, cm


def save_metrics(metrics, filename='./results/metrics.csv'):
    """
    Сохраняет метрики моделей в CSV файл.

    Функция записывает метрики производительности различных моделей в CSV файл
    с указанием названия модели, точности (Accuracy) и Macro-F1 score.

    Parameters
    ----------
    metrics : dict
        Словарь с метриками моделей, где:
        - ключ (str): название модели
        - значение (tuple): кортеж из двух элементов (accuracy, macro_f1)
    filename : str, optional
        Путь к файлу для сохранения метрик

    Returns
    -------
    None
    """
    os.makedirs('./results', exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Macro-F1'])
        for model_name, (acc, f1) in metrics.items():
            writer.writerow([model_name, acc, f1])