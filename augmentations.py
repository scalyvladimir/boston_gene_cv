from torchvision import transforms as TT


def get_train_transform():
    return TT.Compose([
        TT.RandomHorizontalFlip(p=0.7),
        TT.RandomResizedCrop(128, scale=(0.5, 1)),  # scale - min; max area of crop
        TT.RandomRotation(25),
        TT.GaussianBlur(9), 
        TT.ToTensor(),
        TT.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform():
    return TT.Compose([
        TT.Resize((128, 128)),
        TT.ToTensor(),
        TT.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])