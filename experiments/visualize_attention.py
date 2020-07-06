
import os
import shutil
import time
import math
import pickle
import pdb

import numpy as np

import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch

import torchvision.transforms as transforms
import torchvision.models as models

from cirtorch.networks.imageretrievalnet import AttRetrievalNet
from cirtorch.networks.imageretrievalnet import init_network

from cirtorch.datasets.testdataset import configdataset

from cirtorch.utils.evaluate import compute_map_and_print

from cirtorch.networks.imageretrievalnet import extract_vectors_attention

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root, htime


test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
pool_names = ['mac', 'spoc', 'gem', 'gemmp']
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

# TODO load network

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')


# input file / dataset
parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained network should be saved')


parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                    help='comma separated list of test datasets: ' +
                        ' | '.join(test_datasets_names) +
                        ' (default: roxford5k,rparis6k)')
parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
                    help='dataset used to learn whitening for testing: ' +
                        ' | '.join(test_whiten_names) +
                        ' (default: None)')

parser.add_argument('--test-freq', default=1, type=int, metavar='N',
                    help='run test evaluation every N epochs (default: 1)')

# network architecture and initialization options
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                    help='pooling options: ' +
                        ' | '.join(pool_names) +
                        ' (default: gem)')
parser.add_argument('--local-whitening', '-lw', dest='local_whitening', action='store_true',
                    help='train model with learnable local whitening (linear layer) before the pooling')
parser.add_argument('--regional', '-r', dest='regional', action='store_true',
                    help='train model with regional pooling using fixed grid')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                    help='train model with learnable whitening (linear layer) after the pooling')
parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                    help='initialize model with random weights (default: pretrained on imagenet)')


parser.add_argument('--model-path', type=str, help='')
parser.add_argument('--image-path', type=str, help='')

## STORE DATA LOADER ##
parser.add_argument('--store-loader', default='', type=str, metavar='FILENAME',
                    help='')
## LOAD DATA LOADER ##
parser.add_argument('--load-loader', default='', type=str, metavar='FILENAME',
                    help='')




args = parser.parse_args()

model_params = {}
model_params['architecture'] = args.arch
model_params['pooling'] = args.pool
model_params['local_whitening'] = args.local_whitening
model_params['regional'] = args.regional
model_params['whitening'] = args.whitening
# model_params['mean'] = ...  # will use default
# model_params['std'] = ...  # will use default
model_params['pretrained'] = args.pretrained

model = init_network(model_params)
# very ghetto
model = AttRetrievalNet(model.features, model.meta)

# TODO actually load model
checkpoint = torch.load(args.model_path)
start_epoch = checkpoint['epoch']
min_loss = checkpoint['min_loss']

print('>> loading model')
model.load_state_dict(checkpoint['state_dict'])

# for testing we use image size of max 1024
image_size = 1024


# set up the transform
normalize = transforms.Normalize(
    mean=model.meta['mean'],
    std=model.meta['std']
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])



# evaluate on test datasets
datasets = args.test_datasets.split(',')
for dataset in datasets:
    start = time.time()

    print('>> {}: Extracting...'.format(dataset))

    # prepare config structure for the test dataset
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

    # extract database and query vectors

    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images[:3], imsize=image_size, bbxs=None, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    for i, input in enumerate(loader) :
        _, _, attention = model(input)

        # todo load image manually without transform

        image = mpimg.imread(qimages[i])


        fig, ax = plt.subplots(2, 1)

        ax[0].imshow(image)

        ax[0].imshow(attention.cpu().numpy())

        # todo add i
        fig.savefig('{}image{}_{}.png'.format(args.image_path, dataset, i))

        fig.close()
