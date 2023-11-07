import torch

class SimpleDataset(torch.utils.data.Dataset):
    # for dataloader without target

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform
    
    def __getitem__(self, index):
        # Load data and get label
        img = self.imgs[index]
        if self.transform:
            img = self.transform(img)
        
        return {'data': img}
    
    def __len__(self):
        return len(self.imgs)