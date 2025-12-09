import os
import sys
import torch
import torch.nn as nn
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        num_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=75, type=int, help='number of epoch in training') # 150
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--aug_ops', type=str,
                    default='',
                    help='Comma-separated list of data augmentations: '
                         'dropout, scale, shift, rot_z, rot_3d, rot_rand, jitter, jitter_sigma_clip, flip')
    parser.add_argument('--extra_features', type=str,
                    default='',
                    help='Comma-separated list of extra per-point features: '
                         'radius, height, pca, curvature, density')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    # ---- DEBUG: Print augmentation ops ----
    if hasattr(args, 'aug_ops'):
        active_aug = [s.strip() for s in args.aug_ops.split(',') if s.strip()]
        log_string(f"Active Data Augmentation: {active_aug if active_aug else 'None'}")
    else:
        log_string("Active Data Augmentation: None")

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    '''Extra Feature'''
    def count_extra_channel(args):
        if not hasattr(args, 'extra_features') or not args.extra_features:
            return 0
        feats = set([s.strip() for s in args.extra_features.split(',') if s.strip()])
        dim = 0
        if 'radius' in feats:
            dim += 1
        if 'height' in feats:
            dim += 1
        if 'pca' in feats:
            dim += 3
        if 'curvature' in feats:
            dim += 1
        if 'density' in feats:
            dim += 1
        return dim

    extra_channel = count_extra_channel(args)
    # ---- DEBUG: Print extra feature engineering ----
    if hasattr(args, 'extra_features') and args.extra_features.strip():
        features = [s.strip() for s in args.extra_features.split(',') if s.strip()]
        log_string(f"Extra Features Enabled: {features}, Total Extra Channels = {extra_channel}")
    else:
        log_string("Extra Features Enabled: None")

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals, extra_channel=extra_channel)
    #criterion = model.get_loss()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.2).cuda()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'AdamW':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.05    
        )
        '''
        AdamW optimizer
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.05    
        )
        '''
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-5    # small minimum LR helps PointNet
    )
    '''
    #Alternative LR scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-5    # small minimum LR helps PointNet
    )
    '''
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            aug_ops = set(args.aug_ops.split(',')) if hasattr(args, 'aug_ops') else set()
            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # ========================================================================================================================================================================
            '''
            if epoch == start_epoch and batch_id == 0:
                print("\n====== DEBUG: INPUT FEATURES ======")
                print("Raw point tensor shape (before transpose):", points.shape)  # [B, N, C]
                C = points.shape[2]

                if args.use_normals:
                    print("Normals enabled: YES (channels 3–5)")
                else:
                    print("Normals enabled: NO")

                print(f"Extra feature channels requested: {args.extra_features}")
                print(f"Total extra feature dims: {extra_channel}")

                # Preview one point from batch 0
                print("Example point[0,0]:", points[0,0,:])
                print("===================================\n")
            # ========================================================================================================================================================================
            '''
            # 1) rotations
            if 'rot_z' in aug_ops:
                points[:, :, 0:3] = provider.rotate_point_cloud_z(points[:, :, 0:3])

            if 'rot_3d' in aug_ops:
                points[:, :, 0:3] = provider.rotate_point_cloud_3d(points[:, :, 0:3])
                
            if 'rot_rand' in aug_ops:
                # Random rotation (Generally along arbitrary axis)
                points[:, :, :3] = provider.rotate_point_cloud(points[:, :, :3])

            # 2) jitter / noise
            if 'jitter_sigma_clip' in aug_ops:
                points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3], sigma=0.01, clip=0.05)

            if 'jitter' in aug_ops:
                points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])

            # 3) flips (mirror)
            if 'flip' in aug_ops:
                points[:, :, 0:3] = provider.random_flip_point_cloud(points[:, :, 0:3])

            # 4) dropout (point dropping)
            if 'dropout' in aug_ops:
                points = provider.random_point_dropout(points)

            # 5) global scale & shift
            if 'scale' in aug_ops:
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])

            if 'shift' in aug_ops:
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            '''
            # ========================================================================================================================================================================
            if epoch == start_epoch and batch_id == 0:
                print("\n====== DEBUG: AFTER DATA AUGMENTATION ======")
                print("AUG OPS:", aug_ops)
                print("XYZ after augmentation (first point):", points[0, 0, :3])
                print("Extra features unaffected:", points[0, 0, 6:] if C > 6 else "None")
                print("============================================\n")
            # ========================================================================================================================================================================
            '''
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            '''
            # ========================================================================================================================================================================
            if epoch == start_epoch and batch_id == 0:
                print("\n====== DEBUG: MODEL INPUT ======")
                print("Model input shape:", points.shape)  # [B, C, N]
                print("C =", points.shape[1], "channels")
                print("XYZ sample:", points[0, :3, 0])
                if args.use_normals:
                    print("Normals sample:", points[0, 3:6, 0])
                if extra_channel > 0:
                    start = 3 + (3 if args.use_normals else 0)
                    print("Extra feature sample:", points[0, start:, 0])
                print("=================================\n")
            # ========================================================================================================================================================================
            '''
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            #pred, trans_feat = classifier(points)
            #loss = criterion(pred, target.long(), trans_feat)
            pred, _ = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
