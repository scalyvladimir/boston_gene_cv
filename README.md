# Computer Vision Test case for Boston Gene

Vladimir Chernyy

Навигация по репозиторию

## 1. Кодовая база
  - [models.py](models.py) - `pytorch_lightning.LightningModule` обертка для обучения нейросеток-наследников `torch.nn.Module`, а также логирование и train/val циклы
  - [data.py](data.py) - `pytorch_lightning.LightningDataModule` класс для реализации логики генерации обучающей и валидационной разбивок, обертки для `torchvision.datasets.ImageFolder`, `torch.utils.data.DataLoader`
  - [main.py](main.py) - скрипт запуска конфигов обучения из папки [configs/](configs/)

    пример обучения:
    ```
    python main.py fit -c configs/default.yaml
    ```
  - [augmentations.py](augmentations.py) - тренировачные/валидационные аугментации

## 2.  Вспомогальные папки
  - [pics/](pics/) -- картинки лоссов и метрик
  - [configs/](configs/) -- тренировочные конфигурации
  - [train_embeds](notebooks/train_embeds_tgt.npy), [val_embeds](notebooks/val_embeds_tgt.npy) -- сохраненные трейновые и валидационные фичи для картинок(в виде `numpy.ndarray`), можно потестить мои чиселки при желании

## 3. Железо
`Intel Core-i5 12400F` + `GeForce RTX 3070 8Gb` - все обучал на своем домашнем пк(веса *"домашние"*)

## 3. Отчет

В виде ноутбука доступен по [ссылке](notebooks/report.ipynb)
