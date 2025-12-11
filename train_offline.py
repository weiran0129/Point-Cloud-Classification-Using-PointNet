import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader_offline import ModelNetDataLoader

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
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training') # 150
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training')
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
    parser.add_argument('--offline_aug', action='store_true', help='perform offline augmentation before training')
    return parser.parse_args()



def offline_augment_modelnet10(args):
    """
    Create an offline-augmented ModelNet10 dataset:
    - Copies the original dataset
    - Applies specified augmentations to 50% of the training samples
    - Saves augmented samples as *_aug.txt
    - Rewrites modelnet10_train.txt inside the new folder
    """
    # Find baseline data loc
    orig_root = "data/modelnet40_normal_resampled"   # original data folder
    shape_name_file = os.path.join(orig_root, "modelnet10_shape_names.txt")
    train_file = os.path.join(orig_root, "modelnet10_train.txt")
    test_file = os.path.join(orig_root, "modelnet10_test.txt")

    if not os.path.exists(train_file):
        raise FileNotFoundError("modelnet10_train.txt not found in data/modelnet40_normal_resampled")

    # New Aug Dir
    aug_tag = args.aug_ops.replace(",", "_") if args.aug_ops else "noaug"
    aug_root = f"data/modelnet10_aug_{aug_tag}"

    if os.path.exists(aug_root):
        print(f"[INFO] Augmented dataset already exists: {aug_root}")
        print("       Reusing it (delete folder manually if you want to regenerate).")
        return aug_root

    print(f"[INFO] Creating augmented dataset → {aug_root}")
    os.makedirs(aug_root, exist_ok=True)
    shutil.copy(shape_name_file, aug_root)
    shutil.copy(test_file, aug_root)

    # Copy folder sturc
    with open(shape_name_file) as f:
        classes = [c.strip() for c in f.readlines()]

    for cls in classes:
        os.makedirs(os.path.join(aug_root, cls), exist_ok=True)

    # Read train list + 50% aug
    train_items = [line.strip() for line in open(train_file)]
    total = len(train_items)
    aug_count = total // 2            
    to_augment = set(random.sample(train_items, aug_count))  #
    print(f"[INFO] Total train samples: {total}")
    print(f"[INFO] Augmenting 50%: {aug_count}")

    # Augmentation
    new_train_list = []

    for item in tqdm(train_items, desc="Copying + Augmentation"):
        cls = "_".join(item.split("_")[:-1])
        orig_path = os.path.join(orig_root, cls, item + ".txt")
        dst_path = os.path.join(aug_root, cls, item + ".txt")

        # Copy original file
        shutil.copy(orig_path, dst_path)
        new_train_list.append(item)

        # Check if this file needs augmentation
        if item in to_augment:

            pc = np.loadtxt(orig_path, delimiter=",").astype(np.float32)
            ops = set([op.strip() for op in args.aug_ops.split(",") if op.strip()])
            ops = set([op.strip() for op in args.aug_ops.split(",") if op.strip()])

            # Prepare batch view for XYZ + normals
            pc_xyz = pc[:, :3]
            pc_normal = pc[:, 3:6] if pc.shape[1] >= 6 else None

            # wrap into batch
            xyz_batch = pc_xyz.reshape(1, pc_xyz.shape[0], 3)
            if pc_normal is not None:
                normal_batch = pc_normal.reshape(1, pc_normal.shape[0], 3)

            if "rot_rand" in ops:
                if pc_normal is not None:
                    batch6 = np.concatenate([xyz_batch, normal_batch], axis=2)
                    batch6 = provider.rotate_point_cloud_with_normal(batch6)
                    xyz_batch = batch6[:, :, :3]
                    normal_batch = batch6[:, :, 3:]
                else:
                    xyz_batch = provider.rotate_point_cloud(xyz_batch)

            if "jitter_sigma_clip" in ops:
                xyz_batch = provider.jitter_point_cloud(xyz_batch, sigma=0.01, clip=0.05)

            if "scale" in ops:
                xyz_batch = provider.random_scale_point_cloud(xyz_batch)

            if "shift" in ops:
                xyz_batch = provider.shift_point_cloud(xyz_batch)

            if "dropout" in ops:
                xyz_batch = provider.random_point_dropout(xyz_batch)

            if "flip" in ops:
                if pc_normal is None:
                    xyz_batch = provider.random_flip_point_cloud(xyz_batch, prob=0.5)
                else:
                    # use per-sample version
                    for bi in range(xyz_batch.shape[0]):
                        xyz, normal = provider.flip_point_cloud_with_normal(
                            xyz_batch[bi], 
                            normal_batch[bi],
                            prob=0.5
                        )
                        xyz_batch[bi] = xyz
                        normal_batch[bi] = normal
            # unwrap batch
            pc[:, :3] = xyz_batch.reshape(pc_xyz.shape[0], 3)
            if pc_normal is not None:
                pc[:, 3:6] = normal_batch.reshape(pc_normal.shape[0], 3)

            # Saving
            aug_name = item + "_aug"
            aug_path = os.path.join(aug_root, cls, aug_name + ".txt")
            np.savetxt(aug_path, pc, delimiter=",")

            new_train_list.append(aug_name)

    # Corresponding new train file
    train_out = os.path.join(aug_root, "modelnet10_train.txt")
    with open(train_out, "w") as f:
        for name in new_train_list:
            f.write(name + "\n")
    print("[INFO] Offline augmentation completed.")
    print(f"[INFO] Folder ready: {aug_root}")

    return aug_root


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
    if args.offline_aug:
        data_path = offline_augment_modelnet10(args)
    else:
        data_path = "data/modelnet40_normal_resampled/"
    #data_path = 'data/modelnet40_normal_resampled/'
    
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
 
           
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
        
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
    # Enable augmented dataset deletion for efficient space saving
    '''
    if args.offline_aug:
        aug_dir = os.path.join('data', f"modelnet10_aug_{args.aug_ops.replace(',', '_')}")
        if os.path.exists(aug_dir):
            print(f"[INFO] Deleting offline augmented dataset: {aug_dir}")
            shutil.rmtree(aug_dir)
    '''


if __name__ == '__main__':
    args = parse_args()
    main(args)
