from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from data import *

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(frame)
        # x = Variable(x.unsqueeze(0))
        x = Variable(transform(x).unsqueeze(0))

        print(x.shape)
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    global stream
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        frame = predict(frame)
        # frame = frame
        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break

NMS_CONF_THRE = 0.03
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from data import VOCAnnotationTransform, VOCDetection, BaseTransform

    from bidet_ssd import build_bidet_ssd

    cfg = voc
    num_classes = len(labelmap) + 1  # +1 for background

    net = build_bidet_ssd('test', cfg['min_dim'], num_classes,
                              nms_conf_thre=NMS_CONF_THRE, is_rsign=False,nms_iou_thre=0.45, nms_top_k=200)
    # net = build_ssd('test', 300, 21)    # initialize SSD
    # net.load_state_dict(torch.load(args.weights))
    weight_path = "pretrain/model_95000_loc_1.0231_conf_2.3187_reg_0.0155_prior_0.1799_loss_3.5372_lr_0.0001.pth"
    try:
        net.load_state_dict(torch.load(weight_path))
    except:
        net.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu'))['weight'])

    transform = BaseTransformTesting(300, (123, 117, 104))
    # transform = BaseTransform(300, mean = (104, 117, 123))
    net.eval()
    fps = FPS().start()
    with torch.no_grad():
        cv2_demo(net, transform)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()