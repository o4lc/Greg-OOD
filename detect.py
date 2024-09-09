'''
Detect OOD samples with CLF
'''

import argparse
import numpy as np
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from scipy import stats
# import sklearn.covariance
from PIL import Image
import os
import os.path
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import get_clf
# from utils import compute_all_metrics
import sklearn.metrics as sk
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds
import torchvision.datasets as dset
from datasets.imagenetUtil import *
from torchvision import transforms
import torchvision
from tqdm import tqdm



class SVHN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"],
        'train_and_extra': [
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]
            self.filename = self.split_list[split][0][1]
            self.file_md5 = self.split_list[split][0][2]
        else:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        if self.split == "test":
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']

            if self.split == "train_and_extra":
                extra_filename = self.split_list[split][1][1]
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))
                self.data = np.concatenate([self.data,
                                                  loaded_mat['X']], axis=3)
                self.targets = np.vstack((self.targets,
                                               loaded_mat['y']))
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == "test":
            return len(self.data)
        else:
            return len(self.data)



def get_msp_score(data_loader, clf):
    clf.eval()

    msp_score = []
    for sample in data_loader:
        # data = sample['data'].cuda()
        data = sample[0].cuda()
        with torch.no_grad():
            logit = clf(data)

            prob = torch.softmax(logit, dim=1)
            msp_score.extend(torch.max(prob, dim=1)[0].tolist())

    return msp_score

def get_abs_score(data_loader, clf):
    '''
    Probability for absent class
    '''
    clf.eval()

    abs_score = []
    feat_norm = []

    for sample in tqdm(data_loader):
        # data = sample['data'].cuda()
        data = sample[0].cuda()

        with torch.no_grad():
            logit = clf(data)
            # logit, feat = clf(data, ret_feat=True)
            # norm2 = torch.norm(feat, p=2, dim=1)

            prob = torch.softmax(logit, dim=1)
            abs_score.extend(prob[:, -1].tolist())
            # feat_norm.extend(norm2.tolist())

    # return [1 - abs for abs in abs_score], feat_norm
    return [1 - abs for abs in abs_score]

def get_logit_score(data_loader, clf):
    clf.eval()

    logit_score = []
    for sample in data_loader:
        # data = sample['data'].cuda()
        data = sample[0].cuda()

        with torch.no_grad():
            logit = clf(data)
            logit_score.extend(torch.max(logit, dim=1)[0].tolist())

    return logit_score

def get_odin_score(data_loader, clf, temperature=1000.0, magnitude=0.0014, std=(0.2470, 0.2435, 0.2616)):
    clf.eval()
    
    odin_scores = []
    
    for sample in data_loader:
        # data = sample['data'].cuda()
        data = sample[0].cuda()
        data.requires_grad = True
        logit = clf(data)
        pred = logit.detach().argmax(axis=1)
        logit = logit / temperature
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit, pred)
        loss.backward()
        
        # normalizing the gradient to binary in {-1, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2
        
        gradient[:, 0] = gradient[:, 0] / std[0]
        gradient[:, 1] = gradient[:, 1] / std[1]
        gradient[:, 2] = gradient[:, 2] / std[2]
        
        tmpInputs = torch.add(data.detach(), -magnitude, gradient)
        logit = clf(tmpInputs)
        logit = logit / temperature
        # calculating the confidence after add the perturbation
        nnOutput = logit.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        
        odin_scores.extend(nnOutput.max(dim=1)[0].tolist())
    
    return odin_scores

def sample_estimator(data_loader, clf, num_classes):
    clf.eval()
    group_lasso = EmpiricalCovariance(assume_centered=False)

    num_sample_per_class = np.zeros(num_classes)
    list_features = [0] * num_classes

    for sample in data_loader:
        # data = sample['data'].cuda()
        # target = sample['label'].cuda()
        data = sample[0].cuda()
        target = sample[1].cuda()
        with torch.no_grad():
            _, penulti_feature = clf(data, ret_feat=True)

        # construct the sample matrix
        for i in range(target.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                list_features[label] = penulti_feature[i].view(1, -1)
            else:
                list_features[label] = torch.cat((list_features[label], penulti_feature[i].view(1, -1)), 0)
            num_sample_per_class[label] += 1

    category_sample_mean = []
    for j in range(num_classes):
        category_sample_mean.append(torch.mean(list_features[j], 0))

    X = 0
    for j in range(num_classes):
        if j == 0:
            X = list_features[j] - category_sample_mean[j]
        else:
            X = torch.cat((X, list_features[j] - category_sample_mean[j]), 0)
        
        # find inverse
    group_lasso.fit(X.cpu().numpy())
    precision = group_lasso.precision_
    
    return category_sample_mean, torch.from_numpy(precision).float().cuda()

def get_mahalanobis_score(data_loader, clf, num_classes, sample_mean, precision):
    '''
    Negative mahalanobis distance to the cloest class center
    '''
    clf.eval()

    nm_score = []
    for sample in data_loader:
        # data = sample['data'].cuda()
        data = sample[0].cuda()
        with torch.no_grad():
            _, penul_feat = clf(data, ret_feat=True)

        term_gaus = torch.empty(0)
        for j in range(num_classes):
            category_sample_mean = sample_mean[j]
            zero_f = penul_feat - category_sample_mean
            # term_gau = torch.exp(-0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()) # [BATCH,]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag() # [BATCH, ]
            if j == 0:
                term_gaus = term_gau.view(-1, 1)
            else:
                term_gaus = torch.cat((term_gaus, term_gau.view(-1, 1)), dim=1)

        nm_score.extend(torch.max(term_gaus, dim=1)[0].tolist())

    return nm_score

def get_energy_score(data_loader, clf, temperature=1.0):
    clf.eval()
    
    energy_score = []

    for sample in tqdm(data_loader):
        # data = sample['data'].cuda()
        data = sample[0].cuda()
        with torch.no_grad():
            logit = clf(data)
            energy_score.extend((temperature * torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_score


def get_entropy_score(data_loader, clf):
    clf.eval()

    entropy_score = []

    for sample in data_loader:
        # data = sample['data'].cuda()
        data = sample[0].cuda()
        with torch.no_grad():
            logit = clf(data)
            num_classes = logit.shape[1]
            posit = torch.softmax(logit, dim=1)
            entropy_score.extend(((posit * torch.log(posit + 1e-8)).sum(axis=1) / np.log(num_classes)).tolist())

    return entropy_score

def get_acc(data_loader, clf, num_classes):
    clf.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for sample in data_loader:
            # data = sample['data'].cuda()
            # target = sample['label'].cuda()
            data = sample[0].cuda()
            target = sample[1].cuda()

            logit = clf(data)

            _, pred = logit[:, :num_classes].max(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    print(correct / total * 100.)
    return correct / total * 100.

score_dic = {
    'msp': get_msp_score,
    'odin': get_odin_score,
    'abs': get_abs_score,
    'logit': get_logit_score,
    'maha': get_mahalanobis_score,
    'energy': get_energy_score,
    'entropy': get_entropy_score
}

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(examples, labels, recall_level=0.95):
    # pos = np.array(_pos[:]).reshape((-1, 1))
    # neg = np.array(_neg[:]).reshape((-1, 1))
    # examples = np.squeeze(np.vstack((pos, neg)))
    # labels = np.zeros(len(examples), dtype=np.int32)
    # labels[:len(pos)] += 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def main(args):
    if args.id != "imagenet":
        _, std = get_ds_info(args.id, 'mean_and_std')

    # test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf_id)
    # test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    if args.id.startswith("cifa"):
        test_trf_id = get_ds_trf(args.id, 'test')
        if args.id == "cifar100":
            num_classes = 100
            test_set_id = dset.CIFAR100('../data/cifarpy', train=False, transform=test_trf_id, download=True)
        elif args.id == "cifar10":
            num_classes = 10
            test_set_id = dset.CIFAR10('../data/cifarpy', train=False, transform=test_trf_id, download=True)
    elif args.id == "imagenet":
        num_classes = args.imagenetNumberOfClasses
        _, _, _, transform_test_largescale, _, train_set_id, \
            test_set_id, _, train_set_all_ood = setup_imagenet(num_classes, rootPath="../data/imageNet",
                                                               validationOnly=True)
    else:
        raise NotImplementedError
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch,
                                pin_memory=True)
    test_loader_oods = []
    ### datasets:
    ood_names = []
    args.test_bs = args.batch_size


    if args.id.startswith("cifar"):
        ood_names.append("dtd")
        ood_data = dset.ImageFolder(root="../data/dtd/images",
                                    transform=get_ood_trf(args.id, 'dtd', 'test'))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)
        # /////////////// SVHN /////////////// # cropped and no sampling of the test set
        ood_names.append("svhn")
        ood_data = SVHN(root='../data/svhn/', split="test",
                             transform=get_ood_trf(args.id, 'svhn', 'test'), download=False)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)

        # # /////////////// Places365 ///////////////
        # ood_data = dset.ImageFolder(root="../../data/places365/",
        #                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
        #                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
        # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
        #                                          num_workers=2, pin_memory=True)
        # print('\n\nPlaces365 Detection')
        # get_and_print_results(ood_loader)
        # /////////////// Places365 ///////////////
        ood_names.append("places365_10k")
        ood_data = dset.ImageFolder(root="../data/test_256/",
                                    transform=get_ood_trf(args.id, 'places365_10k', 'test'))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)

        # # /////////////// SUN ///////////////
        # ood_data = dset.ImageFolder(root="../../data/SUN",
        #                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
        #                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
        # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
        #                                          num_workers=1, pin_memory=True)
        # print('\n\nSUN Detection')
        # get_and_print_results(ood_loader)

        # /////////////// LSUN-C ///////////////
        ood_names.append("lsunc")
        ood_data = dset.ImageFolder(root="../data/LSUN",
                                    transform=get_ood_trf(args.id, 'lsunc', 'test'))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)

        # # /////////////// LSUN-R ///////////////
        ood_names.append("lsunr")
        ood_data = dset.ImageFolder(root="../data/LSUN_resize",
                                    transform=get_ood_trf(args.id, 'lsunr', 'test'))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)

        # # /////////////// iSUN ///////////////
        ood_names.append("isun")
        ood_data = dset.ImageFolder(root="../data/iSUN/",
                                    transform=get_ood_trf(args.id, 'isun', 'test'))
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)
    elif args.id == "imagenet":
        # transform_test_largescale = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                      std=[0.229, 0.224, 0.225]),
        # ])

        ood_names.append("dtd")
        ood_data = dset.ImageFolder(root="../data/dtd/images",
                                    transform=transform_test_largescale)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=4, pin_memory=False)
        test_loader_oods.append(ood_loader)

        ood_names.append("Places")
        ood_data = dset.ImageFolder(root="../data/Places/",
                                    transform=transform_test_largescale)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=6, pin_memory=False)
        test_loader_oods.append(ood_loader)

        ood_names.append("iNat")
        ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder("../data/iNaturalist", transform=transform_test_largescale),
            batch_size=args.test_bs, shuffle=False, pin_memory=True, num_workers=4)
        test_loader_oods.append(ood_loader)

        ood_names.append("SUN")
        ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder("../data/SUN", transform=transform_test_largescale), batch_size=args.test_bs,
            shuffle=False, pin_memory=True, num_workers=4)
        test_loader_oods.append(ood_loader)

    else:
        raise ValueError("Invalid dataset name.")




    # load CLF

    if args.score == 'abs':
        clf = get_clf(args.arch, num_classes+1)
    elif args.score in ['maha', 'logit', 'energy', 'msp', 'odin', 'entropy']:
        clf = get_clf(args.arch, num_classes)
    else:
        raise RuntimeError('<<< Invalid score: '.format(args.score))
    
    # clf = nn.DataParallel(clf)
    clf_path = Path(args.pretrain)

    if clf_path.is_file():
        clf_state = torch.load(str(clf_path), map_location='cuda:0')
        newDictionary = {}
        for k in clf_state['state_dict'].keys():
            newDictionary[k.replace('module.', '').replace('resnet.', '')] = clf_state['state_dict'][k]
        # cla_acc = clf_state['cla_acc']

        clf.load_state_dict(newDictionary)
        # print('>>> load classifier from {} (classification acc {:.4f}%)'.format(str(clf_path), cla_acc))
    else:
        raise RuntimeError('<--- invlaid classifier path: {}'.format(str(clf_path)))

    # move CLF to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        clf.cuda()
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False

    # get_acc(test_loader_id, clf, num_classes)

    get_score = score_dic[args.score]
    if args.score == 'maha':
        train_set_id_test = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=test_trf_id)
        train_loader_id_test = DataLoader(train_set_id_test, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
        cat_mean, precision = sample_estimator(train_loader_id_test, clf, num_classes)
        get_score = partial(
            score_dic['maha'],
            num_classes=num_classes, 
            sample_mean=cat_mean, 
            precision=precision
        )
    elif args.score == 'odin':
        get_score = partial(
            score_dic['odin'],
            temperature=args.temperature,
            magnitude=args.magnitude,
            std=std
        )
    else:
        get_score = score_dic[args.score]
    # score_id, _ = get_score(test_loader_id, clf)
    score_id = get_score(test_loader_id, clf)
    label_id = np.ones(len(score_id))

    # visualize the confidence distribution
    plt.figure(figsize=(10, 10), dpi=100)
    
    score_ood_all = np.empty(0)

    fprs, aurocs, auprs = [], [], []
    # pc_sum = 0
    pc = 0
    for i, test_loader_ood in enumerate(test_loader_oods):
        print("Testing dataset: ", ood_names[i])
        # result_dic = {'name': test_loader_ood.dataset.name}

        # ood_names.append(test_loader_ood.dataset.name)

        # score_ood, norm_ood = get_score(test_loader_ood, clf)
        score_ood = get_score(test_loader_ood, clf)
        score_ood = np.array(score_ood)
        # norm_ood = np.array(norm_ood)

        # idxs = []
        # for j in range(1000):
        #     low_b = j / 1000
        #     up_b = (j+1) / 1000
        #     idxs_piece = np.where(np.logical_and(score_ood >= low_b , score_ood < up_b))[0]
        #     if len(idxs) > 10:
        #         idxs.extend(np.array(idxs_piece[:10]).tolist())
        #     else:
        #         idxs.extend(np.array(idxs_piece).tolist())
        
        # print(idxs)

        # pc_sum += np.corrcoef(np.array(score_ood), np.array(norm_ood))[0, 1]
        # score_ood = get_score(test_loader_ood, clf)
        # res = stats.spearmanr(score_ood[idxs], norm_ood[idxs])
        # pc = res.correlation
        
        label_ood = np.zeros(len(score_ood))

        # OOD detection
        score = np.concatenate([score_id, score_ood])
        label = np.concatenate([label_id, label_ood])

        # plot the histgrams
        bins = np.linspace(0.0, 1.0, 100)
        plt.subplot(3, 3, i+1)
        plt.hist(score_id, bins, color='g', label='id', alpha=0.5)
        thr_95 = np.sort(score_id)[int(len(score_id) * 0.05)]
        plt.axvline(thr_95, alpha=0.5)
        plt.hist(score_ood, bins, color='r', label='ood', alpha=0.5)
        plt.title(ood_names[i])

        auroc, aupr, fpr = get_measures(score, label)
        
        fprs.append(100. * fpr)
        aurocs.append(100. * auroc)
        auprs.append(100. * aupr)

        score_ood_all = np.concatenate([score_ood_all, score_ood[:8000]], axis=0)

    # save id and all ood scores seperately
    print('PC: ', pc)
    np.save(str(Path(args.pretrain).parent / 'id.npy'), score_id)
    np.save(str(Path(args.pretrain).parent / 'ood.npy'), score_ood_all)

    # save the figure
    plt.savefig(args.fig_name)

    # print results
    print('[ ID: {:7s} - OOD:'.format(args.id), end=' ')
    for ood_name in ood_names:
        print('{:5s}'.format(ood_name), end=' ')
    print(']')

    print('> FPR:  ', end=' ')
    for fpr in fprs:
        print('{:3.3f}'.format(fpr), end=' ')
    print('<', np.average(fprs))

    print('> AUROC:', end=' ')
    for auroc in aurocs:
        print('{:3.3f}'.format(auroc), end=' ')
    print('<', np.average(aurocs))

    print('> AUPR: ', end=' ')
    for aupr in auprs:
        print('{:3.3f}'.format(aupr), end=' ')
    print('<', np.average(auprs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialize detection')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--id', type=str, default='cifar100')
    # parser.add_argument('--oods', nargs='+', default=['svhn', 'lsunc', 'dtd', 'places365_10k', 'lsunr', 'isun'])
    parser.add_argument('--score', type=str, default='msp', choices=['msp', 'odin', 'abs', 'logit', 'maha', 'energy', 'entropy'])
    parser.add_argument('--temperature', type=int, default=1000)
    parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--prefetch', type=int, default=10)
    parser.add_argument('--arch', type=str, default='densenet101', choices=['densenet101', 'wrn40_2',
                                                                            'wrn40_4', 'resnet50', 'resnet101', 'resnet18',
                                                                            'densenet121', 'resnet18_32x32'])
    parser.add_argument('--pretrain', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--fig_name', type=str, default='test.png')
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument("--imagenetNumberOfClasses", type=int, default=10)

    args = parser.parse_args()

    main(args)