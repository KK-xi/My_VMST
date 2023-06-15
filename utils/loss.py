import torch
import torch.nn.functional as F

def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def loss_and_accuracy(prediction, target):
    loss = F.nll_loss(prediction, target)
    pred = prediction.max(1)[1]
    accuracy = (pred.eq(target).sum().item()) / len(target)
    return loss, accuracy