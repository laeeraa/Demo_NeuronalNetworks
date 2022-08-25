# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import torch
import sys

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config', default='yolov3_mobilenetv2_320_300e_coco.py', help='test config file path')
    parser.add_argument('--checkpoint', default='yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth',help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    #initialize the webcam
    camera = cv2.VideoCapture(0)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        _, frame = camera.read()
        x,y,c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = inference_detector(model,framergb)

        model.show_result(
            framergb, result, score_thr=args.score_thr, wait_time=1, show=True)

        if cv2.waitKey(1) == ord('q'):
         break

#
#        ret_val, img = camera.read()
#        result = inference_detector(model, img)
#
#        model.show_result(
#            img, result, score_thr=args.score_thr, wait_time=1, show=True)
#
#        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q') or cv2.waitKey(1)== ord('Q'):
#            break

        
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parse_args()
    main()


