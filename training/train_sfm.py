import argparse
import time
import os
import shutil
import pickle

import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim

import torchvision.transforms as transforms

from cirtorch.datasets.datahelpers import collate_tuples
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename

from cirtorch.utils.evaluate import compute_map_and_print

from cirtorch.utils.general import get_data_root, htime

from networks.attention import AttRetrievalNet, extract_vectors

from loss.ot import OTContrastiveLoss, OTMatchContrastiveLoss

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained network should be saved')


loss_names = ['otmatch', 'ot', 'contrastive']

parser.add_argument('--loss', '-l', metavar='LOSS', default='otmatch',
                    choices=loss_names,
                    help='training loss options: ' +
                        ' | '.join(loss_names) +
                        ' (default: contrastive)')

metric_names = ['euclidean', 'ot']
parser.add_argument('--eval-metric', type=str, default='euclidean')

parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: 0.7)')


# CHECKPOINTS
parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                    help='name of the latest checkpoint (default: None)')

parser.add_argument('--load-loader', default='', type=str, metavar='FILENAME',
                    help='')


def main():
    global args, min_loss
    args = parser.parse_args()

    # create export dir if it doesnt exist
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)


    model = AttRetrievalNet()

    # move network to gpu
    model.cuda()

    # define loss function (criterion) and optimizer
    if args.loss == 'otmatch':
        criterion = OTMatchContrastiveLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'ot':
        criterion = OTContrastiveLoss(margin=args.loss_margin).cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))



    # define optimizer

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    # TODO: maybe pass as argument in future implementation?
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)


    ## WRAP BACKBONE ##
    model = AttRetrievalNet(model.features, model.meta)



    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume :
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))


    # Data loading code
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    ## RESUME DATA LOADER ##
    if args.load_loader :
        if os.path.isfile(args.load_loader):
            train_loader = torch.load(args.load_loader)
    else :
        train_dataset = TuplesDataset(
            name=args.training_dataset,
            mode='train',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=args.query_size,
            poolsize=args.pool_size,
            transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            drop_last=True, collate_fn=collate_tuples
        )


    if args.val:
        val_dataset = TuplesDataset(
            name=args.training_dataset,
            mode='val',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # evaluate the network before starting
    # this might not be necessary?
    #
    for epoch in range(start_epoch, args.epochs):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        # # debug printing to check if everything ok
        # lr_feat = optimizer.param_groups[0]['lr']
        # lr_pool = optimizer.param_groups[1]['lr']
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))

        # train for one epoch on train set
        ## TRAINING ##
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        #if args.val:
        #    with torch.no_grad():
        #        loss = validate(val_loader, model, criterion, epoch)

        # evaluate on test datasets every test_freq epochs
        if (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                test(args.test_datasets, model)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.directory)

    # test
    test(args.test_datasets, model)

# end main

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)
# end save_checkpoint

def train(train_loader, model, criterion, optimizer, epoch) :
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if not args.load_loader :
        print('create epoch tuples')
        avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)
        if args.store_loader :
            torch.save(train_loader, args.store_loader)

    model.train()

    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader) :
        nq = len(input)
        ni = len(input[0])

        for q in range(nq) :
            _, q_features, q_attention = model(input[q][0].cuda())
            for ini in range(1, ni) :
                _, ini_features, ini_attention = model(input[q][ini].cuda())

                loss = criterion((q_features.flatten(-2), q_attention.flatten(-2)),
                                 (ini_features.flatten(-2), ini_attention.flatten(-2)),
                                 target[q][ini].cuda())

            input_batch = torch.cat([torch.nn.functional.interpolate(input[q][imi], (683, 1024)) for imi in range(ni)],
                                    dim=0).cuda()

            _, ni_features, ni_attention = model(input_batch)

            loss = criterion(ni_features, ni_attention, target[q].cuda())
            losses.update(loss.item())
            loss.backward()

# end train

def test(datasets, net):
    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.test_whiten:
        start = time.time()

        print('>> {}: Learning whitening...'.format(args.test_whiten))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', args.test_whiten)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.test_whiten))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        # extract whitening vectors
        print('>> {}: Extracting...'.format(args.test_whiten))

        # TODO: custom function
        wvecs = extract_vectors(net, images, image_size, transform)  # implemented with torch.no_grad

        # learning whitening
        print('>> {}: Learning...'.format(args.test_whiten))
        wvecs = wvecs.numpy()
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.test_whiten, htime(time.time() - start)))
    else:
        Lw = None

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
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, image_size, transform)  # implemented with torch.no_grad
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, image_size, transform, bbxs)  # implemented with torch.no_grad

        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        if Lw is not None:
            # whiten the vectors
            vecs_lw = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time() - start)))
# end test


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# end AverageMeter


if __name__ == '__main__':
    main()
