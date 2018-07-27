# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 23:47
# @Author  : Ruichen Shao
# @File    : utils.py.py
# show process bar
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, '>' * rate_num, ' ' * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

# show image with rect
def show_rect(path, bboxes):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = plt.imread(path)
    ax.imshow(img)
    for bbox in bboxes:
        w = bbox['x2'] - bbox['x1']
        h = bbox['y2'] - bbox['y1']
        x = bbox['x1']
        y = bbox['y1']
        rect = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        # add label text
        ax.text(x, y, bbox['class'], fontdict={'size': 12, 'color': 'r'})
    plt.show()