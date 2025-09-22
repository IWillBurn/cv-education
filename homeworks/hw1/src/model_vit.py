import timm


def get_vit_tiny(
        num_classes: int = 10
    ) -> timm.models.vision_transformer.VisionTransformer:
    """
    Создает и настраивает предобученную модель Vision Transformer (ViT) tiny версии.

    Функция загружает предобученную модель ViT-tiny из библиотеки timm,
    замораживает все параметры кроме головы классификатора, которая обучается
    для решения конкретной задачи классификации.

    Parameters
    ----------
    num_classes : int, optional
        Количество выходных классов для головы классификатора

    Returns
    -------
    timm.models.vision_transformer.VisionTransformer
        Модель Vision Transformer с замороженными параметрами и обучаемой головой
    """
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    return model