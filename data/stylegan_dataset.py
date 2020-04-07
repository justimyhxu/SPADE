"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
from .pix2pix_dataset import Pix2pixDataset
import mmcv
import pandas as pd

class StyleGANDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        label_paths, app_paths, image_paths, instance_paths, pose_mask_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(app_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        pose_paths = label_paths[:opt.max_dataset_size]
        app_paths = app_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        pose_mask_paths = pose_mask_paths[:opt.max_dataset_size]
        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.pose_paths = pose_paths
        self.app_paths = app_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.pose_mask_paths = pose_mask_paths
        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        ### parrse annotation
        annotations = pd.read_csv(opt.ann_file)
        print('--------------{}---------- '.format(opt.ann_file))
        if opt.filter_list is not None:
            flist = mmcv.load(filter_list)
            fkeys = [int(x) for x in flist if flist[x]]
            annotations = annotations.iloc[fkeys, :]
            print('Filter image list: ', len(self.annotations))

        if opt.rm_bg:
            prefix = 'crop_Mask_Img'
            full_p = lambda x: os.path.join(opt.dataroot, prefix, x.replace('.jpg','_mask_img.jpg'))
        else:
            prefix = 'crop_Img'
            full_p = lambda x: os.path.join(opt.dataroot, prefix, x)

        label_paths = [full_p(x) for x in annotations['pose_image']]
        app_paths = [full_p(x) for x in annotations['app_image']]
        images_paths = [full_p(x) for x in annotations['target_image']]
        pose_mask_paths =[os.path.join(opt.dataroot, prefix, x.replace('.jpg','_mask.jpg')) for x in annotations['pose_image']]
        if not opt.no_instance:
            instance_paths = [os.path.join(opt.dataroot, 'crop_Img', x) for x in annotations['flow']]
        else:
            instance_paths = []
        return label_paths, app_paths, images_paths, instance_paths, pose_mask_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image, provide appearance
        pose_mask_path = self.pose_mask_paths[index]
        pose_mask = Image.open(pose_mask_path)
        params = get_params(self.opt, pose_mask.size)

        if self.opt.label_nc == 0:
            transform_pose_mask = get_transform(self.opt, params)
            pose_mask_tensor = transform_pose_mask(pose_mask.convert('RGB'))
        else:
            transform_pose_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            pose_mask_tensor = transform_pose_mask(pose_mask) * 255.0
            pose_mask_tensor[pose_mask_tensor == 255] = 1  # 'unknown' is opt.label_nc

        # provide app image
        pose_path = self.pose_paths[index]
        pose_image = Image.open(pose_path)
        transform_pose = get_transform(self.opt, params)
        pose_tensor = transform_pose(pose_image)
        # provide app image
        app_path = self.app_paths[index]
        app_image = Image.open(app_path)
        transform_app = get_transform(self.opt, params)
        app_tensor = transform_app(app_image)

        # input image (real images)
        image_path = self.image_paths[index]
        # if not self.opt.no_pairing_check:
        #     assert self.paths_match(label_path, image_path), \
        #         "The label_path %s and image_path %s don't match." % \
        #         (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_pose_mask(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_pose_mask(instance)

        input_dict = {'pose': pose_tensor,
                      'pose_mask': pose_mask_tensor,
                      'app': app_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
