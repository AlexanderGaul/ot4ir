import os
from PIL import Image
import numpy as np
import torch.utils.data as data

from utils.datasets.preprocess import get_tuple_transform_ops
from utils.eval.measure import pose2fund, sampson_distance

class ImMatchDatasetMega(data.Dataset):
    '''Data wrapper for train image-matching with triplets'''
    def __init__(self, data_root, match_file, scene_list=None,
                 wt=480, ht=320, min_pt=100, item_type='triplet'):
        print('\nInitialize ImMatchDatasetMega...')        
        self.dataset = 'MegaDepth_undistort'
        self.data_root = os.path.join(data_root, self.dataset)
        self.match_file = match_file
        self.transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=True)
        self.wt, self.ht = wt, ht
        self.item_type = item_type
        self.min_pt = min_pt
        
        # Initialize data
        self.ims = {}            # {scene: {im: imsize}}
        self.pos_pair_pool = []  # [pair]
        self.load_pairs(scene_list)        
        
    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        if crop:
            dw, dh = crop  
            im = np.array(im)

            # Crop from right and buttom to keep the target aspect ratio
            h, w, _ = im.shape
            im = im[0: h - int(dh), 0: w - int(dw)]
            #print(h, w, im.shape)
            im = Image.fromarray(im)
        return im
            
    def load_pairs(self, scene_list=None):        
        match_dict = np.load(self.match_file, allow_pickle=True).item()
        self.scenes = scene_list if scene_list else match_dict.keys()
        print('Loading data from {}'.format(self.match_file))                
        
        num_ims = 0
        for sc in self.scenes:
            for pair in match_dict[sc]['pairs']:
                if len(pair.matches) < self.min_pt:
                    continue
                self.pos_pair_pool.append(pair)
            self.ims[sc] = match_dict[sc]['ims']
            num_ims += len(match_dict[sc]['ims'])    
        print('Loaded scenes {} ims: {} pos pairs:{}'.format(len(self.scenes), num_ims, len(self.pos_pair_pool)))            
    
    def get_matches(self, pair, im1, im2):        
        # Recompute camera intrinsic matrix due to the resize
        sx1, sy1 = self.wt / im1.width, self.ht / im1.height
        sx2, sy2 = self.wt / im2.width, self.ht / im2.height
        sK1 = np.array([[sx1, 0, 0], [0, sy1, 0], [0, 0, 1]])
        sK2 = np.array([[sx2, 0, 0], [0, sy2, 0], [0, 0, 1]])
        K1, K2 = sK1.dot(pair.K1), sK2.dot(pair.K2)
        
        # Calculate F
        F = pose2fund(K1, K2, pair.R, pair.t)
        rescale = np.array([[sx1, sy1, sx2, sy2]])
        matches = pair.matches * rescale
        
        # Pick random topk pts
        dists = sampson_distance(matches[:, :2], matches[:, 2:4], F)
        ids = np.argsort(dists)[:self.min_pt]
        matches = matches[ids]
        #print(np.mean(dists[ids]), np.max(dists[ids]))
        return matches, F, K1, K2
    
    def __getitem__(self, index):
        """
        Batch dict:
            - 'src_im': anchor image
            - 'pos_im': positive image sample to the anchor
            - 'neg_im': negative image sample to the anchor
            - 'im_pair_refs': path of images (src, pos, neg)
            - 'pair_data': namedtuple contains relative pose information between src and pos ims.
        """
        data_dict = {}
        
        # Load positive pair data
        pair = self.pos_pair_pool[index]
        im_src_ref = os.path.join(self.data_root, pair.im1)
        im_pos_ref = os.path.join(self.data_root, pair.im2)
        im_src = self.load_im(im_src_ref, crop=pair.crop1)
        im_pos = self.load_im(im_pos_ref, crop=pair.crop2)
        
        # Compute fundamental matrix before RESIZE
        matches, F, K1, K2 = self.get_matches(pair, im_src, im_pos)
        
        # Process images
        im_src, im_pos = self.transform_ops((im_src, im_pos))
        
       # Select a negative image from other scences        
        if self.item_type == 'triplet':
            other_scenes = list(self.scenes)
            other_scenes.remove(pair.im1.split('/')[0])
            neg_scene = np.random.choice(other_scenes)
            im_neg_data = np.random.choice(self.ims[neg_scene])
            im_neg_ref = os.path.join(self.data_root, im_neg_data.name)
            im_neg = self.load_im(im_neg_ref, crop=im_neg_data.crop)
            im_neg = self.transform_ops([im_neg])[0] 
        else:
            im_neg_ref = None
            im_neg = None            
        
        # Wrap data item
        data_dict = {'src_im': im_src, 
                     'pos_im': im_pos, 
                     'neg_im': im_neg,
                     'im_pair_refs': (im_src_ref, im_pos_ref, im_neg_ref),
                     'F': F,
                     'K1': K1,
                     'K2': K2,
                     'matches': matches
                     }

        return data_dict
    
    def __len__(self):
        return len(self.pos_pair_pool)
    
    def __repr__(self):
        fmt_str = 'ImMatchDatasetMega scenes:{} data type:{}\n'.format(len(self.scenes), self.item_type)
        fmt_str += 'Number of data pairs: {}\n'.format(self.__len__())
        fmt_str += 'Image root location: {}\n'.format(self.data_root)
        fmt_str += 'Match file: {}\n'.format(self.match_file)
        fmt_str += 'Transforms: {}\n'.format(self.transform_ops.__repr__().replace('\n', '\n    '))
        return fmt_str  

