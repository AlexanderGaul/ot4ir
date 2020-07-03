import numpy as np

import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

import argparse

from tqdm import tqdm

from d2net.lib.dataset import MegaDepthDataset

from d2net.lib.exceptions import NoGradientError

from networks.networks import MaxAttNet

from loss.loss_megadepth import ot_loss

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


#################
#   Parser
#################
parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')

parser.add_argument('--dataset_path', type=str, default='', help='/storage/group/cvpr/zhouq/MegaDepth_v1')
parser.add_argument('--scene_info_path', type=str, default='', help='storage/group/cvpr/zhouq/MegaDepth_v1_undistorted')
parser.add_argument('--preprocessing', type=str, default='torch', help='image preprocessing (caffe or torch)')

parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')


def main() :
    args = parser.parse_args()

    #################
    #   DATASETS
    #################
    training_dataset = MegaDepthDataset(
        scene_list_path='megadepth_utils/train_scenes.txt',
        scene_info_path=args.scene_info_path,
        base_path=args.dataset_path,
        preprocessing=args.preprocessing
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    train_loss_history = []


    #################
    #   MODEL
    #################

    model = MaxAttNet()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(1, args.num_epochs + 1) :

        training_dataset.build_dataset()

        train_loss_history.append(
            process_epoch(
                epoch,
                model, ot_loss, optimizer, training_dataloader, device,
                args
            )
        )
# end main


def process_epoch(
        epoch_idx,
        model, loss_function, optimizer, dataloader, device,
        log_file, args, train=True ):

    epoch_losses = []

    torch.set_grad_enabled(train)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        if train:
            optimizer.zero_grad()

        batch['train'] = train
        batch['epoch_idx'] = epoch_idx
        batch['batch_idx'] = batch_idx

        batch['batch_size'] = args.batch_size
        batch['preprocessing'] = args.preprocessing
        batch['log_interval'] = args.log_interval

        try:
            loss = loss_function(model, batch, device, plot=args.plot)
        except NoGradientError:
            continue

        current_loss = loss.data.cpu().numpy()[0]
        epoch_losses.append(current_loss)

        progress_bar.set_postfix(loss=('%.4f' % np.mean(epoch_losses)))

        if batch_idx % args.log_interval == 0:
            print( '[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)) )
            # log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
            #    'train' if train else 'valid',
            #    epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)
            #))

        if train:
            loss.backward()
            optimizer.step()

    print( '[%s] epoch %d - avg_loss: %f\n' % (
            'train' if train else 'valid',
            epoch_idx,
            np.mean(epoch_losses) ))
    #log_file.write('[%s] epoch %d - avg_loss: %f\n' % (
    #    'train' if train else 'valid',
    #    epoch_idx,
    #    np.mean(epoch_losses)
    #))

    log_file.flush()

    return np.mean(epoch_losses)


if __name__ == '__main__':
    main()