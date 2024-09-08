"""
python test.py --model pointMLP --msg 20220209053148-404
"""
import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import progress_bar, IOStream
# from data import ModelNet40
import sklearn.metrics as metrics
# from helper import cal_loss
import numpy as np
import torch.nn.functional as F

import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
labels = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox']
'''

labels = ['Bag', 'Bed', 'Bin', 'Box', 'Cabinet', 'Chair', 'Desk', 'Display', 'Door', 'Pillow', 'Shelf', 'Sink', 'Sofa', 'Table', 'Toilet']

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))

class args:
    checkpoint = '/home/fovea/Desktop/Saeid_3080/run_models_on_old_dataset/pointMLP/classification_ModelNet40/checkpoints/pointMLP-seed-1/'
    msg = 'test'
    batch_size = 16
    model = 'pointMLP'
    num_classes = 15
    num_points = 1024
    workers = 6

def main():
    #args = parse_args()
    #print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    #if args.msg is None:
    #    message = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    #else:
    #    message = "-"+args.msg
    #args.checkpoint = 'checkpoints/' + args.model + message

    print('==> Preparing data..')
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
     #                        batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    

    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    test_out = validate(net, test_loader, criterion, device)
    print(f"Vanilla out: {test_out}")


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    cm = confusion_matrix(test_true,test_pred,normalize='true')
    sns.heatmap(np.round(cm,2),
                annot=True,
                fmt='g',
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"size": 7})
    plt.xlabel('Prediction', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    plt.title('Confusion Matrix', fontsize=7)
    plt.show()

    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()