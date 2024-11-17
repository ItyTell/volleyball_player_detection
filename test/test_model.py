from cv_algorithm.model import load_model
from cv_algorithm.detect import detect_image
from cv2 import imread
from pathlib import Path


def test_model():
    model = load_model(
        "cv_algorithm/model/yolov3.cfg",
        "cv_algorithm/model/yolov3_ckpt_600.pth",
    )
    test_images = [
        imread(path)
        for path in Path("./test/").iterdir()
        if str(path).endswith(".jpg")
    ]

    assert test_images

    for image in test_images:
        boxes = detect_image(model, image, conf_thres=0.01, nms_thres=0.1)
        assert len(boxes) > 0
