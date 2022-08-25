# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import torch
import sys
import mmcv

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        cv2.imshow('Livewebcam',img)
        result = inference_detector(model, img)
        
        frame = model.show_result(img, result, score_thr=args.score_thr)
        
        cv2.namedWindow('video', 0)
        mmcv.imshow(frame, 'video', args.wait_time)

        #model.show_result(
        #    img, result, score_thr=args.score_thr, wait_time=1, show=True)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
           break
    cv2.destroyAllWindows()    


if __name__ == '__main__':
    main()
