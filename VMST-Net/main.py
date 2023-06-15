import torch
import random
import numpy as np
import tqdm
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter

from Net.models import VMST_Net
from utils.configs import FLAGS
from utils.loader import Loader
from utils.loss import loss_and_accuracy
from utils.dataset import My_data

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(3407)


if __name__ == '__main__':

    flags = FLAGS()

    flags.log_dir = os.path.join(flags.log_dir, flags.arch_name)
    if not os.path.exists(flags.log_dir):
        os.makedirs(flags.log_dir)
    writer = SummaryWriter(flags.log_dir)

    flags.save_dir = os.path.join(flags.save_dir, flags.arch_name)
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    device = torch.device(flags.device if torch.cuda.is_available() else "cpu")

    train_dataset = My_data(flags.train_dataset, flags)
    test_dataset = My_data(flags.test_dataset, flags)
    train_loader = Loader(train_dataset, flags, device=device, shuffle=True)
    test_loader = Loader(test_dataset, flags, device=device, shuffle=False)

    model = VMST_Net(flags)
    # for Dataparallel
    model = nn.DataParallel(model, device_ids=flags.device_ids)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=flags.lr, momentum=0.9, weight_decay=flags.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags.epochs, eta_min=flags.min_lr)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("Total number of parameters : %.4f M" %(num_params/1e6))

    iteration = 0
    test_high_accuracy = 0

    for epoch in range(flags.epochs):

        print("Training step [{:3d}/{:3d}]".format(epoch, flags.epochs))
        sum_accuracy = 0
        sum_loss = 0
        model.train()
        for coords, points, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            pred_labels = model(coords, points)
            loss, accuracy = loss_and_accuracy(pred_labels, labels)

            loss.backward()
            optimizer.step()

            sum_accuracy += accuracy
            sum_loss += loss

            iteration += 1

        training_loss = sum_loss.item() / len(train_loader)
        training_accuracy = sum_accuracy / len(train_loader)
        print("Training epoch {:5d}  Loss {:.4f}  Accuracy {:.4f}".format(epoch, training_loss, training_accuracy))

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)
        writer.add_scalar("training/lr", optimizer.state_dict()['param_groups'][0]['lr'], iteration)

        lr_scheduler.step()

        print("Testing step [{:3d}/{:3d}]".format(epoch, flags.epochs))
        sum_accuracy = 0
        sum_loss = 0
        model.eval()
        for coords, points, labels in tqdm.tqdm(test_loader):
            with torch.no_grad():
                pred_labels = model(coords, points)
                loss, accuracy = loss_and_accuracy(pred_labels, labels)

            sum_accuracy += accuracy
            sum_loss += loss

        test_loss = sum_loss.item() / len(test_loader)
        test_accuracy = sum_accuracy / len(test_loader)

        writer.add_scalar("testing/accuracy", test_accuracy, iteration)
        writer.add_scalar("testing/loss", test_loss, iteration)

        print("Testing Loss {:.4f}  Accuracy {:.4f}".format(test_loss, test_accuracy))

        if epoch % flags.save_every_n_epochs == 0:
            state_dict = model.module.state_dict()
            torch.save({
                "state_dict": state_dict,
                "epoch": epoch
            }, flags.save_dir + "/checkpoint_%04d.pth"%(epoch))

        if test_accuracy > test_high_accuracy:
            test_high_accuracy = test_accuracy
            print("test highest accuracy at ", test_high_accuracy)

    print("Test highest accuracy :", test_high_accuracy)