import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.allconv import AllConvNet
from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--ood_method', type=str, choices=['MSP', 'MLV', 'Ensemble', 'MC_dropout'], default='MSP')
parser.add_argument('--ens1', type=str, default='', help='Checkpoint path (file) for 1st model for ensemble.')
parser.add_argument('--ens2', type=str, default='', help='Checkpoint path (file) for 2nd model for ensemble.')
parser.add_argument('--ens3', type=str, default='', help='Checkpoint path (file) for 3rd model for ensemble.')
parser.add_argument('--mc_dropout_iters', type=int, default=4, help='number of forward pass for each image in Monte Carlo dropout.')
parser.add_argument('--cifar10', type=str, default='', help='path to CIFAR-10 dataset.')
parser.add_argument('--cifar100', type=str, default='', help='path to CIFAR-100 dataset.')
parser.add_argument('--texture', type=str, default='', help='path to texture dataset.')
parser.add_argument('--svhn', type=str, default='', help='path to SVHN dataset.')
parser.add_argument('--places365', type=str, default='', help='path to places365 test dataset.')
parser.add_argument('--lsun', type=str, default='', help='path to LSUN dataset.')
args = parser.parse_args()

is_ensemble = (args.ood_method == 'Ensemble')
if is_ensemble:
    args.load = ''

# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10(args.cifar10, train=False, transform=test_transform, download=True)
    num_classes = 10
else:
    test_data = dset.CIFAR100(args.cifar100, train=False, transform=test_transform, download=True)
    num_classes = 100


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if 'allconv' in args.method_name:
    net = AllConvNet(num_classes)
    if is_ensemble:
        net2 = AllConvNet(num_classes)
        net3 = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    if is_ensemble:
        net2 = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
        net3 = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
if args.load != '':
    net.load_state_dict(torch.load(args.load))
    print('Model restored!')

if is_ensemble:
    net.load_state_dict(torch.load(args.ens1))
    net2.load_state_dict(torch.load(args.ens2))
    net3.load_state_dict(torch.load(args.ens3))
    print('Models loaded.')

net.eval()
if is_ensemble:
    net2.eval()
    net3.eval()
if args.ood_method == 'MC_dropout':
    net.train()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    if is_ensemble:
        net2 = torch.nn.DataParallel(net2, device_ids=list(range(args.ngpu)))
        net3 = torch.nn.DataParallel(net3, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    if is_ensemble:
        net2.cuda()
        net3.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            output = net(data)
            if args.ood_method == 'MSP':
                score = to_np(F.softmax(output, dim=1))
            elif args.ood_method == 'MLV':
                score = to_np(output)
            elif args.ood_method == 'Ensemble':
                output2 = net2(data)
                output3 = net3(data)
                score = to_np(F.softmax(output, dim=1)) + to_np(F.softmax(output2, dim=1)) + to_np(F.softmax(output3, dim=1))
            elif args.ood_method == 'MC_dropout':
                for _ in range(args.mc_dropout_iters - 1):
                    output += net(data)
                score = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(score, axis=1))

            if in_dist:
                preds = np.argmax(score, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(score[right_indices], axis=1))
                    _wrong_score.append(-np.max(score[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nRademacher Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 32, 32, 3)))
for i in range(ood_num_examples * args.num_to_avg):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nBlob Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////

if args.texture != '':
    ood_data = dset.ImageFolder(root=args.texture,
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    print('\n\nTexture Detection')
    get_and_print_results(ood_loader)

# /////////////// SVHN ///////////////

if args.svhn != '':
    ood_data = svhn.SVHN(root=args.svhn, split="test",
                         transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    print('\n\nSVHN Detection')
    get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////

if args.places365 != '':
    ood_data = dset.ImageFolder(root=args.places365,
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    print('\n\nPlaces365 Detection')
    get_and_print_results(ood_loader)

# /////////////// LSUN ///////////////

if args.lsun != '':
    ood_data = lsun_loader.LSUN(args.lsun, classes='test',
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    print('\n\nLSUN Detection')
    get_and_print_results(ood_loader)

# /////////////// CIFAR Data ///////////////

if args.cifar10 != '' and args.cifar100 != '':
    if 'cifar10_' in args.method_name:
        ood_data = dset.CIFAR100(args.cifar100, train=False, transform=test_transform, download=True)
    else:
        ood_data = dset.CIFAR10(args.cifar10, train=False, transform=test_transform, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    print('\n\nCIFAR-100 Detection') if 'cifar100' in args.method_name else print('\n\nCIFAR-10 Detection')
    get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(ood_num_examples * args.num_to_avg, 3, 32, 32),
                      low=-1.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[-1,1] Noise Detection')
get_and_print_results(ood_loader)


# /////////////// Arithmetic Mean of Images ///////////////

class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)

if args.cifar10 != '' and args.cifar100 != '':
    if 'cifar10_' in args.method_name:
        ood_data = dset.CIFAR100(args.cifar100, train=False, transform=test_transform, download=True)
    else:
        ood_data = dset.CIFAR10(args.cifar10, train=False, transform=test_transform, download=True)
    ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                             batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    print('\n\nArithmetic Mean of Random Image Pair Detection')
    get_and_print_results(ood_loader)


# /////////////// Geometric Mean of Images ///////////////

class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)

if args.cifar10 != '' and args.cifar100 != '':
    if 'cifar10_' in args.method_name:
        ood_data = dset.CIFAR100(args.cifar100, train=False, transform=trn.ToTensor(), download=True)
    else:
        ood_data = dset.CIFAR10(args.cifar10, train=False, transform=trn.ToTensor(), download=True)
    ood_loader = torch.utils.data.DataLoader(
        GeomMeanOfPair(ood_data), batch_size=args.test_bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    print('\n\nGeometric Mean of Random Image Pair Detection')
    get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.BOX).resize((32, 32), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// RGB Ghosted/Shifted Images ///////////////

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])

print('\n\nRGB Ghosted/Shifted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Inverted Images ///////////////

# not done on all channels to make image ood with higher probability
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])

print('\n\nInverted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
