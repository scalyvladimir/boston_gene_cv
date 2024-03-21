# Computer Vision Test case for Boston Gene

Навигация по репозиторию

1. Кодовая база
   - [models.py](models.py) - `pytorch_lightning.LightningModule` обертка для обучения нейросеток-наследников `torch.nn.Module`, а также логирование и train/val циклы
   - [data.py](data.py) - `pytorch_lightning.LightningDataModule` класс для реализации логики генерации обучающей и валидационной разбивок, обертки для `torchvision.datasets.ImageFolder`, `torch.utils.data.DataLoader`
   - [main.py](main.py) - скрипт запуска конфигов обучения из папки [configs/](configs/)
     пример обучения:
     ```
     python main.py fit -c configs/default.yaml
     ```
   - [augmentations.py](augmentations.py) - тренировачные/валидационные аугментации
2. Отчет
  Доступен по ссылке](notebooks/report.ipynb)
