
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

class ImageDet(): 

    def __init__(self, parent=None):
        #defs for the model
        self.imagepath=""
        self.model=""
        self.config="./../OpenMMLab/mmdetection/yolov3_mobilenetv2_320_300e_coco.py"
        self.checkpoint="./../OpenMMLab/mmdetection/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
        self.device="cpu"
        self.palette="coco"
        self.score_thr=float(0.3)


    def changemodelconfig(self,model): 
        self.model = model
        if(self.model =="Yolov3 with OpenMMLab"): 
            self.config="./../OpenMMLab/mmdetection/yolov3_mobilenetv2_320_300e_coco.py"
            self.checkpoint="./../OpenMMLab/mmdetection/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
            self.score_thr=float(0.3)
        elif(self.model == "Faster rcnn r50 with OpenMMLab"): 
            self.config="./../OpenMMLab/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
            self.checkpoint="./../OpenMMLab/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        elif(self.model == "Yolov4"): 
            self.config="./"
            self.checkpoint=" "  
        

    def processimage_OpenMMLab(self,image):
        out_file="./../images/result.jpg"
            # build the model from a config file and a checkpoint file
        model = init_detector(self.config, self.checkpoint, self.device)
            # test a single image
        result = inference_detector(model,image)
            # show the results
        show_result_pyplot(
                model,
                image,
                result,
                self.score_thr,
                'result',
                0,
                self.palette, 
                out_file)








