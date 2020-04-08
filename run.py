import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import argparse
import queue
import threading
import sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")



parser = argparse.ArgumentParser()

parser.add_argument("--input", type=int, help="Input webcam index", default=0)
parser.add_argument("--movth", type=float, help="Movement threshold", default=5.0)

parser.add_argument("--debug", action="store_true", help="Print debug messages")

parser.add_argument("--raw", action="store_true", help="Output raw video to stdout")
parser.add_argument("--bgr", action="store_true", help="Keep the raw output in the BGR format")
parser.add_argument("--window", action="store_true", help="Output video to opencv window")
parser.add_argument("--info", action="store_true", help="Print webcam dimensions and examples")


args = parser.parse_args()

if not args.window and not args.raw and not args.info:
    print("You have to specify one of --window, --raw or --info")
    sys.exit()


capture = cv2.VideoCapture(args.input)
predictor = DefaultPredictor(cfg)
tracker = cv2.TrackerMOSSE_create()
backSub = cv2.createBackgroundSubtractorKNN(history=100)


width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
green = np.zeros((height, width, 3), np.uint8)
green[:,:,1] = 255

cnn_out = queue.Queue()
cnn_in = queue.Queue()



def process_frame(frame):
    outputs = predictor(frame)
    ret = None
    maxarea = 0
    for c, mask, bbox in zip(outputs["instances"].pred_classes, outputs["instances"].pred_masks, outputs["instances"].pred_boxes):
        if c.item() == 0: #person
            x1, y1, x2, y2 = bbox.cpu().numpy()
            bbox = (x1, y1, x2-x1, y2-y1)

            mask = mask.cpu().numpy().astype(np.uint8)*255
            mask = cv2.blur(mask, (10, 10))
            
            if bbox[2]*bbox[3] > maxarea:
                maxarea = bbox[2]*bbox[3]
                ret = (mask, bbox)
    
    return ret




def main():
    global cnn_in
    global cnn_out

    lastmask = None
    lastbox = None
    
    mask = None
    box = None

    firstmask = True

    ret, frame = capture.read()
    cnn_in.put(frame)
    while True:
        ret, frame = capture.read()
        new = False
        if frame is None: break
        
        movMask = backSub.apply(frame)
        if args.debug: cv2.imshow("movMask", movMask)

        try:
            outputs = cnn_out.get(False)
            if np.mean(movMask) < args.movth or firstmask:
                if args.debug: print("New mask!")
                if outputs is not None:
                    lastmask, lastbox = outputs
                    new = True
                    firstmask = False
            else:
                if args.debug: print("New mask rejected! (movth=%f)" % (np.mean(movMask)))
            if args.debug: print("Sending frame to CNN...")
            cnn_in.put(frame)
            
        except queue.Empty:
            pass
        except Exception as e:
            raise(e)
        
        if new:
            mask = lastmask
            box = lastbox

            tracker = cv2.TrackerMOSSE_create()
            tracker.init(frame, box)
            cnn_out.task_done()

        if mask is not None:
            (success, newbox) = tracker.update(frame)
            if success:
                box = tuple([int(i) for i in newbox])
                
                x, y, w, h = lastbox
                src_pts = np.array([(x, y), (x+w, y), (x, y+h) , (x+w, y+h)], dtype=np.float32)
                x, y, w, h = newbox
                dst_pts = np.array([(x, y), (x+w, y), (x, y+h) , (x+w, y+h)], dtype=np.float32)

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                mask = cv2.warpPerspective(lastmask, M, (width, height))
        
            masked = cv2.bitwise_or(frame, frame, mask=mask)
            bg = cv2.bitwise_or(green, green, mask=cv2.bitwise_not(mask))

            result = cv2.bitwise_or(bg, masked)
            x, y, w, h = box
            if args.debug: cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

            if args.window:
                cv2.imshow("result", result)
            if args.raw:
                if not args.bgr:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                sys.stdout.buffer.write(result.tobytes())
        
        if args.window:
            key = cv2.waitKey(1)

import time
def cnn():
    global cnn_out
    global cnn_in

    while True:
        input_frame = cnn_in.get(True)
        if args.debug: print("Processing frame...")
        outputs = process_frame(input_frame)
        if args.debug: print("Frame processed...")
        cnn_out.put(outputs)
        cnn_in.task_done()
        time.sleep(2)

if __name__ == "__main__":
    if args.info:
        fps = capture.get(cv2.CAP_PROP_FPS)

        print("width: %d  height: %d  fps: %d" % (width, height, fps))
        print("--- VLC ---")
        print("python run.py --raw | vlc --demux=rawvideo --rawvid-fps=%d --rawvid-width=%d --rawvid-height=%d --rawvid-chroma=RV24 - --sout '#display'" % (fps, width, height))
        print("--- ffmpeg (stream udp) ---")
        print("python run.py --raw --bgr | ffmpeg -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -an -f mpegts udp://0.0.0.0:5555" % (width, height, fps))
        print("--- ffmpeg (v4l2-loopback) ---")
        print("python run.py --raw --bgr | ffmpeg -f rawvideo -pix_fmt bgr24 -s %dx%d -r %d -i - -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video2" % (width, height, fps))
        sys.exit()
    main_thread = threading.Thread(target=main)
    cnn_thread = threading.Thread(target=cnn)

    main_thread.start()
    cnn_thread.start()
