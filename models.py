import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import F1Score
import torchvision
import torch.nn as nn
import torch

class ClassificationNet(pl.LightningModule):
    
    def __init__(self, num_classes, freeze_ratio, n_epochs, class_weights=None, model=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_epochs = n_epochs
        
        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights).to(torch.device('cuda:0' if torch.cuda.is_available else 'cpu'))
        else:
            self.class_weights = None

        if model is not None:
            self.model = model
        else:
            self.model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)

        import os
        print(os.getpid())

        params = list(self.model.parameters())
        params_t = int(len(params) * freeze_ratio)

        for param in params[:params_t]:
            param.requires_grad = False

        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        
        self.sm = nn.Softmax(dim=1)
        
        self.train_f1 = F1Score(task='multiclass', average='macro', num_classes=num_classes, multidim_average='global')
        self.val_f1 = F1Score(task='multiclass', average='macro', num_classes=num_classes, multidim_average='global')
    
    def forward(self, x):
        
        logits = self.model(x)
        
        return logits
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-3, amsgrad=True, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-4)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        logits = self.forward(x)
        
        preds = self.sm(logits)
        
        self.train_f1(preds=preds, target=y)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False)
        
        loss = nn.functional.cross_entropy(
            preds,
            y
        )
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        logits = self.forward(x)
        
        preds = self.sm(logits)
                
        self.val_f1(preds=preds, target=y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        
        loss = nn.functional.cross_entropy(
            preds,
            y
        )
        self.log('val_loss', loss)
        
        return loss
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.argmax(self.sm(self(batch)), dim=1)