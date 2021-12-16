import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

import rospy
import sys
from cv_bridge import CvBridge
from odhe_ros.msg import Result
from sensor_msgs.msg import Image, RegionOfInterest

class fasterRCNN(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('scene_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('scene_camera_results', Result, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera2/color/image_raw", Image, self.callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_img(self):
        result = self.image
        return result

    def getResult(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        # if predictions.has("pred_masks"):
        #     masks = np.asarray(predictions.pred_masks)
        # else:
        #     return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # mask = np.zeros(masks[i].shape, dtype="uint8")
            # mask[masks[i, :, :]]=255
            # mask = self._bridge.cv2_to_imgmsg(mask)
            # result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    def publish(self, img, result_msg):
        self.pub.publish(img)
        self.result_pub.publish(result_msg)
        self.loop_rate.sleep()


def main():
    """ Faster RCNN Object Detection with Detectron2 """
    rospy.init_node("faster_rcnn", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0

    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/odhe_ros/scene_camera_dataset/train/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/scene_camera_dataset/train")
    train_metadata = MetadataCatalog.get("train_set")
    dataset_dicts = DatasetCatalog.get("train_set")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_set")
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 classes (plate, cup, bowl, fork, spoon)

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/odhe_ros/scene_camera_dataset/output/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("train_set", )
    predictor = DefaultPredictor(cfg)

    class_names = MetadataCatalog.get("train_set").thing_classes

    # Set up custom cv2 visualization parameters
    # Classes: [name, id]
    #               -
    #          [fork,  0]
    #          [spoon, 1]
    #          [plate, 2]
    #          [bowl,  3]
    #          [cup,   4]

    # Colors = [blue, green, red]
    color_fork = [255, 200, 0]        # blue
    color_spoon = [0, 255, 0]       # green
    color_plate = [0, 0, 255]       # red
    color_bowl = [204, 0, 150]      # purple
    color_cup = [0, 220, 255]       # yellow
    colors = list([color_fork, color_spoon, color_plate, color_bowl, color_cup])

    run = fasterRCNN()
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is not None:
            outputs = predictor(img)
            predictions = outputs["instances"].to("cpu")

            # Get results
            result = run.getResult(predictions, class_names)

            # # Visualize using detectron2 built in visualizer
            # v = Visualizer(img[:, :, ::-1],
            #             metadata=train_metadata, 
            #             scale=1.0 
            #             # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            # )
            # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # img = v.get_image()[:, :, ::-1]
            # im_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")

            # Visualize using custom cv2 code
            if result is not None:
                result_cls = result.class_names
                result_clsId = result.class_ids
                result_scores = result.scores

                for i in range(len(result_clsId)):
                    start_point = (result.boxes[i].x_offset, result.boxes[i].y_offset)
                    end_point = (result.boxes[i].x_offset + result.boxes[i].width, result.boxes[i].y_offset + result.boxes[i].height)
                    start_point2 = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 2)
                    end_point2 = (result.boxes[i].x_offset + result.boxes[i].width - 2, result.boxes[i].y_offset + 12)
                    color = colors[result_clsId[i]]
                    box_thickness =  2

                    name = result_cls[i]
                    score = result_scores[i]
                    conf = round(score.item() * 100, 1)
                    string = str(name) + ":" + str(conf) + "%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 10)
                    fontScale = .3
                    text_thickness = 1
                    img = cv2.rectangle(img, start_point, end_point, color, box_thickness)
                    img = cv2.rectangle(img, start_point2, end_point2, color, -1)     # White text box
                    img = cv2.putText(img, string, org, font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)
                
                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")


            # Display Image Counter
            image_counter = image_counter + 1
            if (image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

            run.publish(im_msg, result)    
        

    return 0

if __name__ == '__main__':
    sys.exit(main())