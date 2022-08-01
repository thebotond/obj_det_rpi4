import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:

  #FPS calculator vars
  counter, fps = 0, 0
  start_time = time.time()

  #Visual input from camera
  capture = cv2.VideoCapture(camera_id)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20 #px
  left_margin = 24 #px
  font_c = (0, 153, 51) #green
  font_s = 1
  font_b = 1
  avg_fps = 10

  #Initialize the object detection model
  base_opt = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  det_op = processor.DetectionOptions(
      max_results=3, score_threshold=0.7)
  options = vision.ObjectDetectorOptions(
      base_options=base_opt, detection_options=det_op)
  detector = vision.ObjectDetector.create_from_options(options)

  #Continuously capture images from the camera and run inference
  while capture.isOpened():
    success, in_img = capture.read()
    if not success:
      sys.exit(
          'Camera error.'
      )

    counter += 1
    in_img = cv2.flip(in_img, 1)
    
    rgb_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    in_tensor = vision.TensorImage.create_from_array(rgb_img)
    pred = detector.detect(in_tensor)
    in_img = utils.visualize(in_img, pred)

    #FPS Calculator
    if counter % avg_fps == 0:
      end_time = time.time()
      fps = avg_fps / (end_time - start_time)
      start_time = time.time()

    #Display FPS
    fps_display = 'FPS = {:.1f}'.format(fps)
    fps_loc = (left_margin, row_size)
    cv2.putText(in_img, fps_display, fps_loc, cv2.FONT_HERSHEY_PLAIN,
                font_s, font_c, font_b)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', in_img)

  capture.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
