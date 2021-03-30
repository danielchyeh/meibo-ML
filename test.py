import torch
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, thresholding=True):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak

    # def thresholding(predictions, targets, correction):
    #     count = 0
    #     for top1 in predictions:
    #         top1 = top1[0].cpu().numpy()
    #         label = targets.data[count].cpu().numpy()
    #         diff = abs(top1 - label)
    #         if label == 6 or label == 7: diff = diff - 1

    #         if diff == 0: correction.append(True)
    #         elif diff == 3: correction.append(True)
    #         elif diff == 4: correction.append(True)
    #         else: correction.append(False)

    #         count = count + 1

    #     print(torch.Tensor(correction).view(-1,1))
    #     corr = torch.Tensor(correction).view(-1,1)

    #     return corr

    top1 = 0.
    top5 = 0.

    correction = []

    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            if thresholding == True:
                count = 0
                for top1 in predictions:
                    top1 = top1[0].cpu().numpy()
                    label = targets.data[count].cpu().numpy()
                    diff = abs(top1 - label)
                    if label == 6 or label == 7: diff = diff - 1

                    if diff == 0: correction.append(True)
                    elif diff == 3: correction.append(True)
                    elif diff == 4: correction.append(True)
                    else: correction.append(False)

                    count = count + 1

                correct = torch.Tensor(correction).view(-1,1)
            else:
                correct = predictions.eq(targets.data.view(-1,1))


            top1 = top1 + correct.narrow(1,0,1).sum().item()
            total += targets.size(0)



    print('Top1 = {}'.format(top1*100./total))

    return top1/total

