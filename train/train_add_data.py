import sys

sys.path.append('..')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
from model.models import get_efficientnet
from dataset.dataset import DeeperForensicsDataset, get_train_transforms, get_valid_transforms, DeeperForensicsDatasetNew
from loss.losses import LabelSmoothing
from catalyst.data.sampler import BalanceClassSampler
from utils.utils import AverageMeter, calculate_metrics, Logger

train_real_paths_npy = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c23/real_30frames_FF_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c40/real_30frames_FF_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/real_60frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFDC_original/new/real_120frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/UADFV/new/real_30frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c23/real_240frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c40/real_240frames_train.npy',
                        '/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset/split_npy/real_train.npy',
                        #'/data1/cby/py_project/FaceForensics/classification/dataset/train_test_npy/train_real_paths.npy',
                        ]

train_fake_paths_npy = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c23/fake_30frames_FF_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c40/fake_30frames_FF_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/fake_30frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFDC_original/new/fake_30frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/UADFV/new/fake_30frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c23/fake_30frames_train.npy',
                        '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c40/fake_30frames_train.npy',
                        '/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset/split_npy/fake_train.npy',
                        #'/data1/cby/py_project/FaceForensics/classification/dataset/train_test_npy/train_fake_paths.npy'
                        ]

# '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c23/real_30frames_FF_val.npy',
# '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c40/real_30frames_FF_val.npy',
val_real_paths_npy = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/real_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFDC_original/new/real_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/UADFV/new/real_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c23/real_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c40/real_30frames_test.npy',
                      '/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset/split_npy/real_val.npy',
                      #'/data1/cby/py_project/FaceForensics/classification/dataset/train_test_npy/test_real_paths.npy'
                     ]

# '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c23/fake_30frames_FF_val.npy',
# '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/c40/fake_30frames_FF_val.npy',
val_fake_paths_npy = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/fake_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFDC_original/new/fake_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/UADFV/new/fake_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c23/fake_30frames_test.npy',
                      '/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/DFD/c40/fake_30frames_test.npy',
                      '/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset/split_npy/fake_val.npy',
                      #'/data1/cby/py_project/FaceForensics/classification/dataset/train_test_npy/test_fake_paths.npy'
                     ]

def eval_model(epoch, is_save=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_score = AverageMeter()
    model.eval()
    num_steps = len(eval_loader)
    print(f'total batches: {num_steps}')
    end = time.time()
    eval_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (XI, label) in enumerate(eval_loader):
            x = Variable(XI.cuda(device_id))
            # label = Variable(torch.LongTensor(label).cuda(device_id))
            label = Variable(label.cuda(device_id))

            # Forward pass: Compute predicted y by passing x to the model
            output = model(x)
            # Compute and print loss
            loss = eval_criterion(output, label)
            losses.update(loss.data.item(), x.size(0))
            # update metrics
            output = nn.Softmax(dim=1)(output)
            confs, predicts = torch.max(output.detach(), dim=1)
            acc_score.update(calculate_metrics(predicts.cpu(), label.cpu()), 1)

            lr = optimizer.param_groups[0]['lr']
            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_FREQ == 0:
                print(f'{epoch} [{i}/{num_steps}]\t'
                      f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'acc {acc_score.val:.4f} ({acc_score.avg:.4f})\t'
                      f'lr {lr:.8f}')

    print(f' *  Eval loss {losses.avg:.4f}\t'f'acc({acc_score.avg:.4f})')
    if is_save:
        train_logger.log(phase="eval", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(acc_score.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    scheduler.step()
    return losses.avg

def train_model(epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_score = AverageMeter()
    model.train()
    num_steps = len(train_loader)
    print(f'total batches: {num_steps}')
    end = time.time()

    for i, (XI, label) in enumerate(train_loader):
        x = Variable(XI.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        label = Variable(label.cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        output = model(x)
        # Compute and print loss
        loss = criterion(output, label)
        # update metrics
        losses.update(loss.data.item(), x.size(0))
        confs, predicts = torch.max(output.detach(), dim=1)
        acc_score.update(calculate_metrics(predicts.cpu(), label.cpu()), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                  f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'acc {acc_score.val:.4f} ({acc_score.avg:.4f})\t'
                  f'lr {lr:.8f}')

    print(f' *  Train loss {losses.avg:.4f}\t'f'acc({acc_score.avg:.4f})')
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(acc_score.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    scheduler.step()
    return losses.val


if __name__ == '__main__':
    LOG_FREQ = 100
    batch_size = 64
    test_batch_size = 128
    device_id = 2
    lr = 1e-3
    epoch_start = 1
    num_epochs = epoch_start + 50
    model_name = 'efficientnet-b3'
    writeFile = '/data1/cby/temp/output_3/logs/' + model_name
    store_name = '/data1/cby/temp/output_3/weights/' + model_name
    if not os.path.isdir(store_name):
        os.makedirs(store_name)
    # model_path = None
    model_path = '/data1/cby/temp/output_3/weights/efficientnet-b3/efn-b3_LS_24_loss_0.1883.pth'
    model = get_efficientnet(model_name=model_name)
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)
    train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])

    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    is_train = False
    if is_train:
        xdl = DeeperForensicsDatasetNew(real_npys=train_real_paths_npy, fake_npys=train_fake_paths_npy,
                                        is_one_hot=True, transforms=get_train_transforms(size=300))
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=False, num_workers=4,
                                  sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
        # train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=True, num_workers=4)
        train_dataset_len = len(xdl)

        xdl_eval = DeeperForensicsDatasetNew(real_npys=val_real_paths_npy, fake_npys=val_fake_paths_npy,
                                             is_one_hot=False, transforms=get_valid_transforms(size=300))
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)
        min_loss = 100 if epoch_start == 1 else eval_model(epoch=epoch_start, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(epoch)
            loss = eval_model(epoch)
            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), '{}/efn-b5_LS_{}_loss_{:.4f}.pth'.format(store_name, epoch, loss))
            print('Current min loss:', min_loss)
        torch.save(model.state_dict(), '{}/efn-b5_LS_{}_loss_{:.4f}.pth'.format(store_name, 'last_50', loss))

    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = DeeperForensicsDataset(data_type='test', transforms=get_valid_transforms(size=300), is_one_hot=False)
        eval_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(epoch=0, is_save=False)
        print('Total time:', time.time() - start)







