# Домашнее задание 1:  
## Тренировочный цикл CNN и linear probe на ViT-Tiny

## Описание
*Решение [ДЗ1](HW1.md).*  
Реализовано:
1. Подготовка данных с аугментацией (CIFAR-10 с 10 классами)
2. sanity-checks
3. Тренировка простой CNN и linear probe на предобученном ViT-Tiny
4. Логирование в TensorBoard
5. Профилировка обучения модели
6. Насчёт метрик и сравнение моделей

## Инструкция запуска
1. Установите зависимости:
```bash
pip install -r requirements.txt
```
2. Запустите:
```bash
python src/main.py
```
3. Логи TensorBoard:
```bash
tensorboard --logdir=runs/
```
4. Trace профайлера: в папке `traces/`.
5. Метрики и confusion matrix: в папке `results/` (CSV и PNG).

## Эксперименты
- Датасет: Мини-CIFAR-10 (10 классов).~~~~
- CNN: Простая сеть с 2 conv слоями.
- ViT-Tiny: Предобученная из timm, только голова обучается.
- Sanity-check: Overfit на 2 батчах — лосс падает до ~0, accuracy ~100%.

## Ключевые наблюдения
- Sanity-check: overfit на 2 бачах достигается. [confusion_matrix_cnn-overfit](results/confusion_matrix_cnn-overfit.png), [confusion_matrix_vit-overfit](results/confusion_matrix_vit-overfit.png)
- Accuracy: CNN - ~70%, ViT-Tiny - ~80%. [metrics](metrics)
- Macro-F1: CNN - ~70%, ViT-Tiny - ~80%. [metrics](metrics)
- CNN сходится медленнее по количеству эпох, но в целом обучается существенно быстрее. (рассчёты проводились на GPU T4 google collab)  
- - [trace-cnn](traces/cnn) (12мс/итр)
- - [trace-vit](traces/vit) (74мс/итр)
- CNN сходится не стабильно - пришлось поиграться с seed.
- Узкие места: В CNN больше всего времени занимает — conv слои; в ViT — attention. [trace-cnn](traces/cnn) (функции типа aten::conv2d), [trace-vit](traces/vit) (функции типа aten::scaled_dot_product_attention)
- confusion_matrix вполне естественные: [confusion_matrix_vit](results/confusion_matrix_vit.png), [confusion_matrix_vit](results/confusion_matrix_vit.png).

## Артефакты
- Логи: `runs/`.
- Trace: `traces/`.
- Метрики: `results/`.