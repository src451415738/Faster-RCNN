# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 23:58
# @Author  : Ruichen Shao
# @File    : train.py.py
from pascal_voc_parser import PascalVOCParser
import utils
import logging
import logging_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = PascalVOCParser('train', '2007', '../RCNN_caffe/VOCdevkit')
    gt = parser.load_ground_truths()
    utils.show_rect(gt[0]['image'], gt[0]['bboxes'])