import cv2
import numpy as np
from tflite_support.task import processor

_MARGIN = 10 #px
_ROW_SIZE = 10 #px
_FONT_S = 2
_FONT_B = 2
_FONT_C = (0, 153, 51) #green


def visualize(
    in_img: np.ndarray,
    pred: processor.DetectionResult,
) -> np.ndarray:

  for detection in pred.detections:
    #bounding box placement
    bounding_box = detection.bounding_box
    starting_pt = bounding_box.origin_x, bounding_box.origin_y
    end_pt = bounding_box.origin_x + bounding_box.width, bounding_box.origin_y + bounding_box.height
    cv2.rectangle(in_img, starting_pt, end_pt, _FONT_C, 3)

    #object label and confidence score
    object = detection.classes[0]
    class_name = object.class_name
    confidence = round(object.score, 2)
    result_text = class_name + ' (' + str(confidence) + ')'
    fps_loc = (_MARGIN + bounding_box.origin_x,
                     _MARGIN + _ROW_SIZE + bounding_box.origin_y)
    cv2.putText(in_img, result_text, fps_loc, cv2.FONT_HERSHEY_PLAIN,
                _FONT_S, _FONT_C, _FONT_B)

  return in_img
