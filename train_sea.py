'''
Tuning or training with auxiliary OOD training data by classification undersampling
'''

import copy
import time
import random
import argparse
import numpy as np
from pathlib import Path
import datetime

from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader
import pdb

from models import get_clf, weights_init
# from utils import setup_logger
from datasets import get_ds_info, get_ds_trf, get_ood_trf, get_ds
from scores import get_weight
import torchvision.datasets as dset
import wandb
from datasets.imagenetUtil import *
from tqdm import tqdm


def calculateEntropy(posits, numberOfClasses):
    '''
    :param posits:  tensor with shape (numberOfSamples, numberOfClasses)
    :return: entropy with shape (numberOfSamples,)
    '''
    return -(posits * torch.log(posits + 1e-8)).sum(axis=1) / np.log(numberOfClasses)
def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

def test(data_loader, clf, num_classes):
    clf.eval()

    total, correct = 0, 0
    total_loss = 0.0

    for sample in data_loader:
        # data = sample['data'].cuda()
        # target = sample['label'].cuda()
        data = sample[0].cuda()
        target = sample[1].cuda()
        with torch.no_grad():
            # forward
            logit = clf(data)
        total_loss += F.cross_entropy(logit, target).item()

        _, pred = logit[:, :num_classes].max(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader),
        'cla_acc': 100. * correct / total
    }

# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def main(args):
    device = torch.device("cpu")
    wandb.init(project="DOS-OOD", entity="limitlessinfinite", config=args)
    timeIdentifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    init_seeds(args.seed)

    if args.repr == 'pca':
        exp_path = Path(args.output_dir) / ('s' + str(args.seed)) / (args.id  + '-' + args.ood) / '-'.join([args.arch, args.oodMethod, args.scheduler, 'b_'+str(args.beta), 'spt_' + str(args.spt), 'ept_'+str(args.ept), 'bs_'+str(args.batch_size), 'seag_k_'+str(args.num_cluster), 'pca', 'mc_'+str(args.mc), 'n_'+str(args.n_init), 'jac', str(args.jacobianLoss), timeIdentifier])
    elif args.repr == 'np':
        exp_path = Path(args.output_dir) / ('s' + str(args.seed)) / (args.id  + '-' + args.ood) / '-'.join([args.arch, args.oodMethod, args.scheduler, 'b_'+str(args.beta), 'spt_' + str(args.spt), 'ept_'+str(args.ept), 'bs_'+str(args.batch_size), 'seag_k_'+str(args.num_cluster), 'np', 'mc_'+str(args.mc), 'n_'+str(args.n_init), 'jac', str(args.jacobianLoss), timeIdentifier])
    else:
        exp_path = Path(args.output_dir) / ('s' + str(args.seed)) / (args.id  + '-' + args.ood) / '-'.join([args.arch, args.oodMethod, args.scheduler, 'b_'+str(args.beta), 'spt_' + str(args.spt), 'ept_'+str(args.ept), 'bs_'+str(args.batch_size), 'seag_k_'+str(args.num_cluster), args.repr, 'n_'+str(args.n_init), 'jac', str(args.jacobianLoss), timeIdentifier])

    exp_path.mkdir(parents=True, exist_ok=True)

    # setup_logger(str(exp_path), 'console.log')
    print('>>> Output dir: {}'.format(str(exp_path)))
    


    if args.id.startswith("cifar"):
        train_trf_id = get_ds_trf(args.id, 'train')
        train_trf_ood = get_ood_trf(args.id, args.ood, 'train')
        test_trf = get_ds_trf(args.id, 'test')

        # train_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='train', transform=train_trf_id)
        # test_set_id = get_ds(root=args.data_dir, ds_name=args.id, split='test', transform=test_trf)

        if args.id == "cifar10":
            train_set_id = dset.CIFAR10('../data/cifarpy', train=True, transform=train_trf_id, download=True)
            test_set_id = dset.CIFAR10('../data/cifarpy', train=False, transform=test_trf, download=True)
        elif args.id == "cifar100":
            train_set_id = dset.CIFAR100('../data/cifarpy', train=True, transform=train_trf_id, download=True)
            test_set_id = dset.CIFAR100('../data/cifarpy', train=False, transform=test_trf, download=True)

        if args.ood == 'tiny_images':
            train_set_all_ood = get_ds(root=args.data_dir, ds_name='tiny_images', split='wo_cifar',
                                       transform=train_trf_ood)
        elif args.ood in ['random_images', 'ti_300k', 'imagenet_64']:
            train_set_all_ood = get_ds(root=args.data_dir, ds_name=args.ood, split='train', transform=train_trf_ood)
    elif args.id == "imagenet":
        # Note that the current implementation of the imagenet dataset is not suited for calculating accuracy. This
        # is due to the fact that we only use a subset of the classes of the imagenet dataset and as a result, the
        # numbering of the classes and target truths gets messed up.
        # @TODO: Fix this issue
        num_classes = args.imagenetNumberOfClasses
        _, _, _, _, _, train_set_id, \
            test_set_id, _, train_set_all_ood = setup_imagenet(num_classes, rootPath="../data/imageNet")
    else:
        raise NotImplementedError


    print("Creating data loaders...")
    train_loader_id = DataLoader(train_set_id, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.prefetch, pin_memory=True, drop_last=True)
    test_loader_id = DataLoader(test_set_id, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.prefetch, pin_memory=True)

    if args.oodMethod == "energy":
        trainValSplit = 0.97
        lengths = [int(trainValSplit * len(train_set_all_ood))]
        lengths.append(len(train_set_all_ood) - lengths[0])
        train_set_all_ood, val_set_ood =\
            torch.utils.data.random_split(train_set_all_ood, lengths)
        print(len(train_set_all_ood), len(val_set_ood))
        valDataLoader = DataLoader(val_set_ood, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.prefetch, pin_memory=True)
        validationEnergies = []
        bestValidationEnergy = -float("inf")
    # the candidate ood idxs
    indices_candidate_ood_epochs = []

    args.size_candidate_ood = min(args.size_candidate_ood, len(train_set_all_ood))
    for i in range(args.epochs):
        indices_epoch = np.array(random.sample(range(len(train_set_all_ood)), args.size_candidate_ood))
        indices_candidate_ood_epochs.append(indices_epoch)

    print('>>> ID: {} - OOD: {}'.format(args.id, args.ood))
    if args.id == "cifar10":
        num_classes = 10
    elif args.id == "cifar100":
        num_classes = 100
    elif args.id == "imagenet":
        num_classes = args.imagenetNumberOfClasses
    else:
        raise ValueError("Unknown dataset {}".format(args.id))
    print('>>> CLF: {}'.format(args.arch))
    clf = get_clf(args.arch, num_classes + (1 if args.oodMethod in ["abs"] else 0))
    if args.finetune:
        print("Loading pretrained model")
        stateDictionary = torch.load(args.pretrainFile, map_location=torch.device('cpu'))['state_dict']
        if args.oodMethod == "abs":
            stateDictionary.pop("fc.weight")
            stateDictionary.pop("fc.bias")
        clf.load_state_dict(stateDictionary, strict=False)


    # move CLF to gpus

    # if torch.cuda.is_available():
        # gpu_idx = int(args.gpu_idx)
        # torch.cuda.set_device(gpu_idx)
        # clf.cuda()
    onGpu = True
    if args.gpu_idx[0] == -1:
        onGpu = False
    else:
        clf = torch.nn.DataParallel(clf, device_ids=args.gpu_idx)
        clf = clf.cuda()
    # clf.apply(weights_init)

    print('Optimizer: LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.momentum))
    if args.singleLrStone:
        lr_stones = [args.singleLrStone]
    else:
        lr_stones = [int(args.epochs * float(lr_stone)) for lr_stone in args.lr_stones]
    optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    
    # print('Scheduler: MultiStepLR - LMS: {}'.format(args.lr_stones))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
    
    if args.scheduler == 'multistep':
        print('LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True - LMS: {}'.format(args.lr, args.weight_decay, args.momentum, args.lr_stones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_stones, gamma=0.1)
    elif args.scheduler == 'lambda':
        print('LR: {:.2f} - WD: {:.5f} - Mom: {:.2f} - Nes: True'.format(args.lr, args.weight_decay, args.momentum))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader_id),
                1,
                1e-6 / args.lr
            )
        )
    else:
        raise RuntimeError('<<< Invalid scheduler: {}'.format(args.scheduler))

    if args.resume:
        print("Loading checkpoint to resume")
        checkpoint = torch.load(args.resumeCheckpoint, map_location=torch.device('cpu'))
        stateDictionary = checkpoint['state_dict']
        newDictionary = {}
        for k, v in stateDictionary.items():
            if k.startswith("module."):
                newDictionary[k] = v
            else:
                newDictionary["module." + k] = v
        clf.load_state_dict(newDictionary)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        cla_acc = checkpoint['cla_acc']
    else:
        start_epoch = 1
        cla_acc = 0.0
    begin_time = time.time()
    batch_size_candidate_ood = int(args.size_candidate_ood / len(train_set_id) * args.batch_size)
    batch_size_sampled_ood = int(args.size_factor_sampled_ood * args.batch_size)
    # print(batch_size_candidate_ood, batch_size_sampled_ood, args.size_candidate_ood, len(train_set_id))
    spt, ept = args.spt, args.ept



    for epoch in range(start_epoch, args.epochs+1):

        train_set_candidate_ood = Subset(train_set_all_ood, indices_candidate_ood_epochs[epoch - 1])
        train_loader_candidate_ood = DataLoader(train_set_candidate_ood, batch_size=batch_size_candidate_ood,
                                                shuffle=False, num_workers=args.prefetch, pin_memory=False)
        print("batch_size_candidate_ood: ", batch_size_candidate_ood)
        epoch_time = time.time()


        for sample_id, sample_ood in tqdm(zip(train_loader_id, train_loader_candidate_ood),
                                          total=min(len(train_loader_id), len(train_loader_candidate_ood))):
            if isinstance(sample_id, list):
                sample_id = {'data': sample_id[0], 'label': sample_id[1]}
            if isinstance(sample_ood, list):
                sample_ood = {'data': sample_ood[0], 'label': sample_ood[1]}
            data_id = sample_id['data']
            target_id = sample_id['label']
            data_batch_candidate_ood = sample_ood['data']
            if onGpu:
                data_id = data_id.cuda()
                target_id = target_id.cuda()
                data_batch_candidate_ood = data_batch_candidate_ood.cuda()
            clf.eval()

            # data_batch_candidate_ood = sample_ood['data']
            # print("Calculating features")
            with torch.no_grad():
                if onGpu:
                    logits_batch_candidate_ood, feats_batch_candidate_ood = clf.module.forward(
                        data_batch_candidate_ood, ret_feat=True)
                else:
                    logits_batch_candidate_ood, feats_batch_candidate_ood = clf.forward(
                        data_batch_candidate_ood, ret_feat=True)
            # print(logits_batch_candidate_ood.shape)
            # pdb.set_trace()
            # prob_id = torch.softmax(logits_batch_id, dim=1)

            # keep the N proximal points with small OOD-ness
            if args.oodMethod == "abs":
                prob_ood = torch.softmax(logits_batch_candidate_ood, dim=1)
                weights_batch_candidate_ood = np.array(prob_ood[:, -1].tolist())
            elif args.oodMethod == "energy":
                energies = -torch.logsumexp(logits_batch_candidate_ood, dim=1)
                weights_batch_candidate_ood = np.array(energies.tolist())
            elif args.oodMethod == "entropy":
                posits = torch.softmax(logits_batch_candidate_ood, dim=1)
                entropies = calculateEntropy(posits, num_classes)
                weights_batch_candidate_ood = np.array(entropies.tolist())
            else:
                raise NotImplementedError
            idxs_sorted = np.argsort(weights_batch_candidate_ood)
            # print("Sorting ...")
            weights_batch_proximial_ood = weights_batch_candidate_ood[list(idxs_sorted[spt:ept])]

            # diff representations: latent embedding, normalized latent embedding, output prob
            if args.repr == 'latent':
                feats_batch_candidate_ood = np.array(feats_batch_candidate_ood.cpu())
                feats_batch_proximial_ood = feats_batch_candidate_ood[list(idxs_sorted[spt:ept])]
                repr_batch_proximial_ood = feats_batch_proximial_ood
            elif args.repr == 'norm':
                repr_batch_proximial_ood = np.array(F.normalize(feats_batch_candidate_ood.cpu(), dim=-1))[list(idxs_sorted[spt:ept])]
            elif args.repr == 'normBatch':
                repr_batch_proximial_ood \
                    = (feats_batch_candidate_ood / torch.linalg.norm(feats_batch_candidate_ood, dim=1).mean())[list(idxs_sorted[spt:ept])].cpu().numpy()
            elif args.repr == 'normFeature':
                repr_batch_proximial_ood = np.array(F.normalize(feats_batch_candidate_ood.cpu(), dim=0))[list(idxs_sorted[spt:ept])]
            elif args.repr == 'output':
                repr_batch_proximial_ood = np.array(torch.softmax(logits_batch_candidate_ood[:, :-1], dim=1).tolist())[list(idxs_sorted[spt:ept])]
            elif args.repr == 'pca':
                feats_batch_candidate_ood = np.array(feats_batch_candidate_ood.cpu())
                feats_batch_proximial_ood = feats_batch_candidate_ood[list(idxs_sorted[spt:ept])]
                pca = PCA(n_components=args.mc)
                repr_batch_proximial_ood = pca.fit_transform(feats_batch_proximial_ood)
                print(pca.explained_variance_ratio_[:10])
            elif args.repr == 'np':
                repr_batch_proximial_ood = np.array(F.normalize(feats_batch_candidate_ood.cpu(), dim=-1))[list(idxs_sorted[spt:ept])]
                pca = PCA(n_components=args.mc)
                repr_batch_proximial_ood = pca.fit_transform(repr_batch_proximial_ood)
                print(pca.explained_variance_ratio_[:10])

            # clustering the N proximial points into K clusters with KMeans algorithm
            k = args.num_cluster
            # print("Clustering ...")
            kmeans = KMeans(n_clusters=args.num_cluster, n_init=args.n_init).fit(repr_batch_proximial_ood)

            clus_proximial_ood = kmeans.labels_

            idxs_sampled = []
            # print("Sampling ...")
            # --- kmeans - sub-cluster ---
            if k > batch_size_sampled_ood:
                sampled_cluster_size = 1
            else:
                sampled_cluster_size = int(batch_size_sampled_ood / k)

            for i in range(min(k, batch_size_sampled_ood)):

                valid_idxs = np.where(clus_proximial_ood == i)[0]
                
                if len(valid_idxs) <= sampled_cluster_size:
                    idxs_sampled.extend(idxs_sorted[spt:][valid_idxs])
                else:
                    idxs_valid_sorted = np.argsort(weights_batch_proximial_ood[valid_idxs])
                    toAdd = idxs_sorted[spt:][valid_idxs[idxs_valid_sorted[:sampled_cluster_size]]]
                    if args.sampleTwoWay:
                        toAdd2 = idxs_sorted[spt:][valid_idxs[idxs_valid_sorted[-sampled_cluster_size:]]]
                        toAdd = np.unique(np.concatenate([toAdd, toAdd2]))


                    idxs_sampled.extend(toAdd)

            finalBatchSize = batch_size_sampled_ood * (2 if args.sampleTwoWay else 1)
            # fill the empty: remove the already sampled, then randomly complete the sampled
            if finalBatchSize > len(idxs_sampled):
                idxs_sampled.extend(random.sample(list(set(idxs_sorted[spt:ept]) - set(idxs_sampled)), k=finalBatchSize - len(idxs_sampled)))
            # indices_sampled_ood = indices_candidate_ood[idxs_sampled]
            # pdb.set_trace()
            data_ood = data_batch_candidate_ood[idxs_sampled]

            # greedy_count += len(set(idxs_sampled) & set(idxs_sorted[:batch_size_sampled_ood]))
            # calculate metric and visulization
            # feats_batch_candidate_ood = np.array(feats_batch_candidate_ood.cpu())
            
            # num_classes = len(train_loader_id.dataset.classes)
            clf.train()
            # print("Training ...")
            total, correct, total_loss = 0, 0, 0.0

            num_id = sample_id['data'].size(0)
            num_ood = data_ood.size(0)

            # print(sample_id['data'])

            # data_id = sample_id['data']
            data = torch.cat([data_id, data_ood], dim=0)
            target_ood = (torch.ones(num_ood) * num_classes).long()
            if onGpu:
                target_ood = target_ood.cuda()
            # target_id = sample_id['label']
            # target_ood = (torch.ones(num_ood) * num_classes).long()

            def jacobianLossFunction(inputs):
                if args.oodMethod == "abs":
                    recalculatedLogits = clf(inputs)
                    inDistCE = F.cross_entropy(recalculatedLogits[:num_id], target_id)
                    outDistCE = F.cross_entropy(recalculatedLogits[num_id:], target_ood)
                    return inDistCE + args.beta * outDistCE
                elif args.oodMethod == "energy":
                    recalculatedLogits = clf(inputs)
                    energyInRecalculated = -torch.logsumexp(recalculatedLogits[:num_id], 1)
                    energyOutRecalculated = -torch.logsumexp(recalculatedLogits[num_id:], 1)
                    return (-(F.relu(args.m_in - energyInRecalculated)).mean()
                            + (F.relu(energyOutRecalculated - args.m_out)).mean())
                elif args.oodMethod == "entropy":
                    recalculatedLogits = clf(inputs)
                    recalculatedPosits = torch.softmax(recalculatedLogits, dim=1)
                    recalculatedEntropies = calculateEntropy(recalculatedPosits, num_classes)
                    entropyInRecalculated = recalculatedEntropies[:num_id]
                    entropyOutRecalculated = recalculatedEntropies[num_id:]
                    return (-(F.relu(args.m_in - entropyInRecalculated)).mean()
                     + (F.relu(entropyOutRecalculated - args.m_out)).mean())


                else:
                    raise NotImplementedError
            # forward




            logit = clf(data)
            ceInd = F.cross_entropy(logit[:num_id], target_id)
            loss = ceInd
            wandb.log({'CE Ind': ceInd.item()})
            jacLoss = 0
            if args.oodMethod == "abs":
                ceOd = args.beta * F.cross_entropy(logit[num_id:], target_ood)
                loss += ceOd
                wandb.log({'CE OOD': ceOd.item()})

            elif args.oodMethod == "energy":
                energyIn = -torch.logsumexp(logit[:num_id], 1)
                energyOut = -torch.logsumexp(logit[num_id:], 1)
                energyLoss = args.beta * (torch.pow(F.relu(energyIn - args.m_in), 2).mean()
                                          + torch.pow(F.relu(args.m_out - energyOut), 2).mean())
                loss += energyLoss
                wandb.log({'Energy Loss': energyLoss.item()})
            elif args.oodMethod == "entropy":
                posit = torch.softmax(logit, dim=1)
                entropies = calculateEntropy(posit, num_classes)
                entropyIn = entropies[:num_id]
                entropyOut = entropies[num_id:]
                entropyLoss = args.beta * (torch.pow(F.relu(entropyIn - args.m_in), 2).mean()
                                           + torch.pow(F.relu(args.m_out - entropyOut), 2).mean())
                loss += entropyLoss
                wandb.log({'Entropy Loss': entropyLoss.item()})

            if args.jacobianLoss:
                jacLoss = torch.autograd.functional.jacobian(jacobianLossFunction, data, True)
                jacLoss = jacLoss.reshape(jacLoss.shape[0], -1)
                jacLoss = args.jacobianLoss * torch.linalg.norm(jacLoss, 2, 1).mean()
                loss += jacLoss
            else:
                jacLoss = torch.autograd.functional.jacobian(jacobianLossFunction, data, False)
                jacLoss = jacLoss.reshape(jacLoss.shape[0], -1)
                jacLoss = torch.linalg.norm(jacLoss, 2, 1).mean()
            wandb.log({"Total loss": loss.item(), 'Jacobian Loss': jacLoss.item()})



            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.primalDual and args.jacobianLoss:
                jacLossThreshold = 0.01
                with torch.no_grad():
                    args.jacobianLoss += 1e-1 * (jacLoss / args.jacobianLoss - jacLossThreshold)
                wandb.log({"Jacobian Loss Weight": args.jacobianLoss})
        
            if args.scheduler == 'lambda':
                scheduler.step()

            # evaluate
            _, pred = logit[:num_id, :num_classes].max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                correct += pred.eq(target_id).sum().item()
                total += num_id



        if args.scheduler == 'multistep':
            scheduler.step()
        
        print('Epoch Time: ', time.time() - epoch_time)
        # print('Greedy Count: ', greedy_count)
        
        # average on sample
        print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(train_loader_id), 100. * correct / total))

        # --------------------------------------------------------------------- #
        if args.oodMethod == "energy":
            print("Validating ...")
            clf.eval()
            energies = []
            with torch.no_grad():
                for x in tqdm(valDataLoader):
                    if args.id == "imagenet":
                        x = x[0]
                    else:
                        x = x['data']
                    logit = clf(x)
                    energy = -torch.logsumexp(logit, 1)
                    energies.append(energy.cpu().numpy())
                energies = np.concatenate(energies)
                validationEnergy = np.mean(energies)
                validationEnergies.append(validationEnergy)
                wandb.log({"Validation Energy": validationEnergy})
                if validationEnergy > bestValidationEnergy:
                    bestValidationEnergy = validationEnergy
                    torch.save({'state_dict': clf.module.state_dict()}, str(exp_path / 'bestLoss.pth'))

                if len(validationEnergies) > 50 and (np.mean(validationEnergies[-5:]) < np.mean(validationEnergies[-10:-5])):
                    print("Early Stopping Condition Met")
                    wandb.log({"Early Stopping Condition": 1})
                    if args.earlyStopping:
                        break
                else:
                    wandb.log({"Early Stopping Condition": 0})






        val_metrics = test(test_loader_id, clf, num_classes)
        cla_acc = val_metrics['cla_acc']

        print(
            '---> Epoch {:4d} | Time {:6d}s'.format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )

        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(clf.module.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'scheduler': copy.deepcopy(scheduler.state_dict()),
                'cla_acc': cla_acc
            }, str(exp_path / (str(epoch)+'.pth')))

    torch.save({
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': copy.deepcopy(clf.module.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'cla_acc': cla_acc
    }, str(exp_path / 'cla_last.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sea')
    parser.add_argument('--seed', default=42, type=int, help='seed for init training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='../data')
    parser.add_argument('--id', type=str, default='cifar100')
    parser.add_argument('--ood', type=str, default='random_images', choices=['tiny_images', 'random_images', 'imagenet_64'])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='tuning')
    parser.add_argument('--arch', type=str, default='densenet101', choices=['densenet101', 'wrn40_2',
                                                                            'wrn40_4', "resnet50", 'resnet101',
                                                                            "resnet18"])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['lambda', 'multistep'])
    parser.add_argument('--singleLrStone', type=int, default=0)
    parser.add_argument('--lr_stones', nargs='+', default=[0.5, 0.75, 0.9])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=101)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2 ** 6) # 64
    parser.add_argument('--size_candidate_ood', type=int, default=300000)
    parser.add_argument('--size_factor_sampled_ood', type=int, default=1)
    parser.add_argument('--repr', type=str, default='norm', choices=['latent', 'norm', 'output', 'pca',
                                                                     'np', 'normBatch', 'normFeature'])
    parser.add_argument('--mc', type=int, default=64, help='main components of PCA')
    parser.add_argument('--spt', type=int, default=16)
    parser.add_argument('--ept', type=int, default=384) # 368 352
    parser.add_argument('--num_cluster', type=int, default=64) # 192: 24(8) -> 64: 8
    parser.add_argument('--n_init', type=int, default=3)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=[0], nargs='+')
    parser.add_argument("--finetune", help="fine tune", action="store_true")
    parser.add_argument("--pretrainFile", help="checkpoint location", type=str, default=None)
    parser.add_argument('--resume', help='resume from checkpoint', action="store_true")
    parser.add_argument("--resumeCheckpoint", help="checkpoint location", type=str, default=None)

    parser.add_argument("--oodMethod", help="ood method", type=str, default="abs",
                        choices=['abs', 'energy', 'entropy'])
    parser.add_argument('--jacobianLoss', help='add jacobian loss', type=float, default=0)
    parser.add_argument("--primalDual", help="use primal dual", action="store_true")
    parser.add_argument('--m_in', type=float, default=-25.,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7.,
                        help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--sampleTwoWay', help='Sample both the worst and best points from each cluster',
                        action="store_true")
    parser.add_argument('--imagenetNumberOfClasses', type=int, default=100)
    parser.add_argument('--earlyStopping', help='Early stopping', action="store_true")

    args = parser.parse_args()

    args.num_cluster = min(args.num_cluster, args.batch_size)
    assert args.imagenetNumberOfClasses <= 100

    # if args.id == "imagenet":
    #     args.size_candidate_ood = 1000 * args.imagenetNumberOfClasses
    # assert args.size_candidate_ood
    main(args)