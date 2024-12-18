import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F

def train_uni(data_loader_id, data_loader_ood, net, optimizer, scheduler=None, beta=0.5):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target)
        loss += beta * -(logit[num_id:].mean(dim=1) - torch.logsumexp(logit[num_id:], dim=1)).mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        # evaluate
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def train_abs(data_loader_id, data_loader_ood, net, optimizer, scheduler=None, beta=1.0):
    num_classes = len(data_loader_id.dataset.classes)
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        
        num_id = sample_id['data'].size(0)
        num_ood = sample_ood['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target_id = sample_id['label'].cuda()
        target_ood = (torch.ones(num_ood) * num_classes).long().cuda()

        # forward
        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target_id)
        loss += beta * F.cross_entropy(logit[num_id:], target_ood)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        # evaluate
        _, pred = logit[:num_id, :num_classes].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target_id).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }


def train_energy(data_loader_id, data_loader_ood, net, optimizer, scheduler=None, beta=0.1):
    net.train()

    total, correct = 0, 0
    total_loss = 0.0

    for sample_id, sample_ood in zip(data_loader_id, data_loader_ood):
        num_id = sample_id['data'].size(0)

        data = torch.cat([sample_id['data'], sample_ood['data']], dim=0).cuda()
        target = sample_id['label'].cuda()

        logit = net(data)
        loss = F.cross_entropy(logit[:num_id], target)
        Ec_in = -torch.logsumexp(logit[:num_id], dim=1)
        Ec_out = -torch.logsumexp(logit[num_id:], dim=1)
        m_in = -25
        m_out = -7
        loss += beta * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        _, pred = logit[:num_id].max(dim=1)
        with torch.no_grad():
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += num_id

    # average on sample
    print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(data_loader_id), 100. * correct / total))
    return {
        'cla_loss': total_loss / len(data_loader_id),
        'cla_acc': 100. * correct / total
    }

def get_trainer(name):
    if name == 'uni':
        return train_uni
    elif name == 'abs':
        return train_abs
    elif name == 'energy':
        return train_energy
    else:
        raise RuntimeError('<<< Invalid training method {}'.format(name))