# Object Detection Model API

This README provides instructions on how to set up and use the object detection model API to retrieve bounding boxes.

# How to Use

1. Import Libraries and Load the Model

        from cv_algorithm import model, detect
        import cv2
        import numpy as np

        # Load the YOLOv3 model
        model_my = model.load_model('cv_algorith/model/yolov3.cfg', 'cv_algorithm/model/yolov3_ckpt_600.pth')

2. Run Detection on an Image

        # Load your image
        img = cv2.imread('path/to/your/image.jpg')

        # Run object detection
        boxes = detect.detect_image(model_my, img, conf_thres=0.01, nms_thres=0.1)

        # `boxes` contains bounding boxes in the format [x1, y1, x2, y2, confidence, class]
        print(boxes)


