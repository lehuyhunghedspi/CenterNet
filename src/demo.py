from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np
import os
import re
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, -1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:

        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for (image_name) in image_names:
            ret = detector.run(image_name)
            img = cv2.imread(image_name)
            with open("/content/drive/My Drive/GR2/icdar_evaluate/" + re.sub(".jpg", ".txt", os.path.basename(image_name)),
                      "w") as f:
                f.write(str(bbox[0]) + ',' + str(bbox[1]) + ',' + \
                        str(bbox[2]) + ',' + str(bbox[1]) + ',' + \
                        str(bbox[2]) + ',' + str(bbox[3]) + ',' + \
                        str(bbox[0]) + ',' + str(bbox[3]) + '\n')
                for bbox in [ret['results'][1][i][:4] for i, value in enumerate(ret['results'][1][:, 4]) if
                             value > 0.3]:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    cv2.imwrite("/content/drive/My Drive/GR2/visualize/" + os.path.basename(image_name), img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            # print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
