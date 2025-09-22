import os
import random
import torch
import numpy as np
from homeworks.hw1.src.data import get_dataloaders
from homeworks.hw1.src.model_cnn import SimpleCNN
from homeworks.hw1.src.model_vit import get_vit_tiny
from homeworks.hw1.src.train import train_model
from homeworks.hw1.src.utils import sanity_check, evaluate_model, save_metrics


def main():
    seed = 1234567
    num_classes = 10
    batch_size = 32
    epochs = 10

    os.environ['KINETO_LOG_LEVEL'] = '5'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Подготовка данных для CNN
        print('Loading data for CNN (32x32)...')
        cnn_trainloader, cnn_valloader, classes = get_dataloaders(batch_size, 1, 0.1, image_size=32, cifar_norm=True)

        # CNN
        model_name = "cnn"
        print('Sanity-check for CNN:')
        cnn_model = SimpleCNN(num_classes)
        sanity_check(cnn_model, model_name, cnn_trainloader, classes, device)
        print('Training CNN...')
        cnn_model = SimpleCNN(num_classes)
        cnn_model = train_model(cnn_model, cnn_trainloader, cnn_valloader, epochs, device=device, log_dir='./runs/cnn', trace_dir='./traces/cnn/', profiler_steps=50)
        cnn_acc, cnn_f1, _ = evaluate_model(cnn_model, model_name, cnn_valloader, device, classes)

        # Подготовка данных для ViT
        print('Loading data for ViT (224x224)...')
        vit_trainloader, vit_valloader, _ = get_dataloaders(batch_size, 1, 0.1, image_size=224, cifar_norm=False)

        # ViT-Tiny
        model_name = "vit"
        print('Sanity-check for ViT:')
        vit_model = get_vit_tiny(num_classes)
        sanity_check(vit_model, model_name, vit_trainloader, classes, device)
        print('Training ViT...')
        vit_model = get_vit_tiny(num_classes)
        vit_model = train_model(vit_model, vit_trainloader, vit_valloader, epochs, device=device, log_dir='./runs/vit', trace_dir='./traces/vit/', profiler_steps=50)
        vit_acc, vit_f1, _ = evaluate_model(vit_model, model_name, vit_valloader, device, classes)

        # Сравнение моделей: метрики и confusion matrix

        metrics = {
            'CNN': (cnn_acc, cnn_f1),
            'ViT-Tiny': (vit_acc, vit_f1)
        }
        save_metrics(metrics)

        print('Done! Check results/ for metrics and confusion matrix.')
        print('TensorBoard logs in run   s/. Traces in traces/.')

    except Exception as e:
        print(f'Error occurred: {e}')
        raise


if __name__ == '__main__':
    main()