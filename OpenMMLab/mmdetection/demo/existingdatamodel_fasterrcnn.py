from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
video='demo/demo.mp4'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

# test a single image and show the results
img = 'demo/obst-und-gemuese1.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')


import sys
import video_demo
# test a video and show the results
sys.argv=[video, config_file, checkpoint_file, '--device cpu', '--show']
video_demo.main()