import torch

from ot.sinkhorn import log_optimal_transport
from utils.distance import pairwise_distances

from d2net.lib.utils import upscale_positions, downscale_positions
from d2net.lib.loss import warp
from d2net.lib.exceptions import EmptyTensorError

def ot_loss(model, batch, device, plot) :

    b = batch['image1'].size(0)


    # do these have batch dim within list??
    scores, locations, features_sel, features_dense = model( torch.cat( [batch['image1'].to(device), batch['image2'].to(device)]) )


    # check for type list
    if not isinstance(scores, list):
        # make to list different sizes of
        scores = [ scores[i, :] for i in range(scores.shape[0]) ]
        locations = [ locations[i, :] for i in range(scores.shape[0]) ]
        features_sel = [ features_sel[i, :] for i in range(features_sel.shape[0]) ]

    n_valid_samples = 0

    for idx_in_batch in range(batch['image1'].size(0)):

        depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]

        depth2 = batch['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch['bbox2'][idx_in_batch].to(device)

        c, h, w = features_dense[0, :].size()

        features_1 = features_sel[idx_in_batch]
        lcoations_1 = locations[idx_in_batch]
        scores_1 = scores[idx_in_batch]


        features_2 = features_sel[idx_in_batch + b]
        locations_2 = locations[idx_in_batch + b]
        scores_2 = scores[idx_in_batch + b]

        M = torch.zeros(len(scores_1)+1, len(scores_2)+1)
        M[:-1, :-1] = pairwise_distances(features_1.transpse(1,0), features_2.transpose(0,1))

        M[-1, :] = 1 # dustbin distance
        M[:-1, -1] = 1 # dustbin distance

        marginal_1 = torch.cat([scores_1, 0])
        marginal_2 = torch.cat([scores_2, 0])

        P = log_optimal_transport(M, scores_1, scores_2, 1000).exp()


        # CALCULATE GROUND TRUTH
        # warp the positions from image 1 to image 2
        fmap_pos1 = locations[idx_in_batch].transpose(0, 1)
        pos1 = upscale_positions(fmap_pos1, scaling_steps = 5)

        # i guess this is the ground truth
        try :
            pos1, pos2, ids = warp(pos1,
                                   depth1, intrinsics1, pose1, bbox1,
                                   depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError :
            print('empty')
            continue
        # pos1 and pos2 same format?
        # ids?

        fmap_pos2 = torch.round(downscale_positions(pos2, scaling_steps = 5)).long()

        ids_2 = -1 * torch.ones_like(ids)
        # locations_2 should be stacked
        for idx, loc in enumerate( locations_2 ) :
            # replace with small distance??
            found = (fmap_pos2.transpose(1, 0) == loc)[0][0]

            ids_2[found] = idx
        print(ids_2)
        # end GROUND TRUTH

        ids_gt = torch.cat([ids.unsqueeze(0), ids_2.unsqueeze(0)], dim=1)
        gt_matches = ids_gt[ids_gt[:, 1] != -1, :]
        gt_trash = ids_gt[ids_gt[:, 1] == -1, :]

        # SUPERGLUE LOSS
        scale = (scores_1[gt_matches[:, 0]] + scores_2[gt_matches[:, 1]]) / 2

        loss = loss - (P[gt_matches[:, 0], gt_matches[:, 1]] / scale).log().sum() \
            + 0.5 (scores_1[gt_trash[:, 0]] + scores_2[gt_trash[:, 1]])
            # - (P[-1, gt_trash[:, 1]] / scores_2[gt_trash[:, 1]]).log().sum() \
            # - (P[gt_trash[:, 0], -1] / scores_1[gt_trash[:, 0]]).log().sum()

        n_valid_samples += 1

    return loss / n_valid_samples