from .core import YOLOv8_face
from .utils import face_coverage, download_file
import cv2
import os


class Portrait:
    def __init__(self):
        self.model = None
        self.check()
        self.load_model()

    def check(self):
        model_url = 'https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn/raw/main/weights/yolov8n-face.onnx'
        model_dir = os.path.join('face_landmark', 'weights')
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'yolov8n-face.onnx')

        if not os.path.exists(self.model_path):
            download_file(model_url, self.model_path)

    def load_model(self,
                   confThreshold=0.45,
                   nmsThreshold=0.5):

        if self.model is None:
            self.model = YOLOv8_face(
                self.model_path, conf_thres=confThreshold, iou_thres=nmsThreshold)
            # srcimg = cv2.imread(imgpath)
        return self.model

    def predict(self, img_path, face_visibility=85):
        if self.model is None:
            raise Exception("Model not loaded")

        threshold = face_coverage(face_visibility/100)
        srcimg = cv2.imread(img_path)
        boxes, scores, classids, kpts = self.model.detect(srcimg)
        # Is Portrait
        features_visibility = self.model.face_features_visible(
            boxes, kpts, threshold)

        # Preview
        # dstimg = self.model.draw_detections(srcimg, boxes, scores, kpts)
        # winName = 'Deep learning face detection use OpenCV'
        # cv2.namedWindow(winName, 0)
        # cv2.imshow(winName, dstimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return features_visibility
