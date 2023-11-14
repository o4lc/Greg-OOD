import torchvision.datasets as dataset
import torchvision.transforms as trn
from .ImageFolder import *
import torch

def setup_imagenet(num_class, rootPath, validationOnly=False):
    crop_size = 224
    val_size = 256
    transform_train = trn.Compose([
        trn.RandomResizedCrop(crop_size),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])

    transform_test = trn.Compose([
        trn.Resize(val_size),
        trn.CenterCrop(crop_size),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])
    train_dir = rootPath + '/train'
    test_dir = rootPath + '/val'
    class_index = list(range(num_class))
    if validationOnly:
        train_dataset = None
    else:
        train_dataset = ImageFolder(train_dir, transform_train, index=class_index)

    # val_dataset = ImageFolder(test_dir, transform_test, index=class_index)

    val_dataset = dataset.ImageNet(test_dir, 'val', transform=transform_test)

    if num_class == 100:
        print("Changing indices ...")
        classToIndex = torch.load("imagenetIdClassAndIndex.pth")
        if not validationOnly:
            train_dataset = cleanDataset(train_dataset, classToIndex, onlyPermute=True)
        val_dataset = cleanDataset(val_dataset, classToIndex, onlyPermute=False)

    else:
        raise NotImplementedError


    if validationOnly:
        class_index_ood = None
        train_dataset_ood = None
    else:
        class_index_ood = list(range(num_class, 7 *num_class))
        train_dataset_ood = ImageFolder(train_dir, transform_train, index=class_index_ood)
        print("Warning. Due to the changes that are made here, the wnid to idx mapping is not correct anymore.")
    return crop_size, val_size, transform_train, transform_test, class_index, train_dataset, \
                        val_dataset, class_index_ood, train_dataset_ood

def cleanDataset(dataset, trainClassToIdx, onlyPermute=False):

    newSamples = []
    if onlyPermute:
        reverseIndexMap = {v: k for k, v in dataset.class_to_idx.items()}
        for sample in dataset.samples:
            newSamples.append((sample[0], trainClassToIdx[reverseIndexMap[sample[1]]]))
    else:
        reverseIndexMap = {v: k for k, v in dataset.wnid_to_idx.items()}
        toRemoveIndices = []
        for classLabel in dataset.wnids:
            if classLabel not in trainClassToIdx.keys():
                toRemoveIndices.append(dataset.wnid_to_idx[classLabel])

        for sample in dataset.samples:
            if sample[1] in toRemoveIndices:
                continue
            newSamples.append((sample[0], trainClassToIdx[reverseIndexMap[sample[1]]]))
    dataset.samples = newSamples
    return dataset
