# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 22:58
# @Author  : Ruichen Shao
# @File    : pascal_voc_parser.py.py
import os
import numpy as np
import pickle
import xml.dom.minidom as minidom
import utils
import logging
import logging_config

logger = logging.getLogger(__name__)

class PascalVOCParser():
    def __init__(self, image_set, year, devkit_path):
        self.name = 'voc_' + year + '_' + image_set
        self.image_set = image_set
        self.year = year
        self.devkit_path = devkit_path
        self.data_path = os.path.join(self.devkit_path, 'VOC' + self.year)
        self.classes_name = ['background', # 背景 idx 为 0
                             'aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor']
        self.classes_idx = range(len(self.classes_name))
        self.image_ext = '.jpg'
        self.cache_path = 'cache'
        # image_set 的所有图片 index
        self.image_idx = self.load_image_set_index()

    # 读取对应 image_set 的索引 txt 文件
    def load_image_set_index(self):
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        if not os.path.exists(image_set_index_file):
            logger.error('Path does not exist: {}'.format(image_set_index_file))
        else:
            with open(image_set_index_file) as f:
                image_idx = [x.strip() for x in f.readlines()]
            return image_idx

    # 载入所有图片的 ground truth
    def load_ground_truths(self):
        if not os.path.exists(self.cache_path):
            logger.info('Creating cache directory...')
            os.makedirs(self.cache_path)

        gt_cache_file = os.path.join(self.cache_path, self.name + '_gt.pkl')
        if os.path.exists(gt_cache_file):
            logger.info('Loading ground truth from {}...'.format(gt_cache_file))
            with open(gt_cache_file, 'rb') as f:
                gt_roidb = pickle.load(f)
            logger.info('Loading finished')
            return gt_roidb

        gt_roidb = []
        num_img = len(self.image_idx)
        for i, idx in enumerate(self.image_idx):
            img_info, gt_rois = self.load_ground_truth(idx)
            gt_roidb.append(
                # gt_roidb 为一个列表
                # 每个图片以及其 ground truth 组成一个字典元素
                # image： 图片路径
                # bboxes: ground truth 列表,每个元素是包含类别和坐标的字典
                {'image': img_info[0], 'width': img_info[1], 'height': img_info[2], 'bboxes': gt_rois}
            )
            utils.view_bar('Loading ground truth from {}'.format(idx + '.xml'), i, num_img)
            print('')
        logger.info('Loading finished')
        # 保存到 cache 中
        with open(gt_cache_file, 'wb') as f:
            logger.info('Writing ground truth to {}'.format(gt_cache_file))
            pickle.dump(gt_roidb, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Writing finishes')
        return gt_roidb

    # 解析 xml 文件,获取一张图片的 ground truth
    def load_ground_truth(self, idx):
        xml_path = os.path.join(self.data_path, 'Annotations', idx + '.xml')
        img_path = os.path.join(self.data_path, 'JPEGImages', idx + self.image_ext)

        with open(xml_path, 'rb') as f:
            data = minidom.parseString(f.read())

        width = int(data.getElementsByTagName('width')[0].childNodes[0].data)
        height = int(data.getElementsByTagName('height')[0].childNodes[0].data)
        objs = data.getElementsByTagName('object')

        num_objs = len(objs)
        gt_rois = []

        for obj in objs:
            x1 = int(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
            y1 = int(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
            x2 = int(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
            y2 = int(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
            class_name = str(obj.getElementsByTagName('name')[0].childNodes[0].data).lower().strip()
            gt_rois.append(
                {'class': class_name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            )

        return [img_path, width, height], gt_rois



