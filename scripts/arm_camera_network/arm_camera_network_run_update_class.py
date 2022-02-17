# This code is the testbed for developing a way for user's to select
# which class an item with low confidence is supposed to be. We will
# then use one-shot learning to improve the model based on the user's
# feedback

# Author: Jack Schultz
# Email: jschultz299@gmail.com
# Created 2/8/22

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import Xlib
import Xlib.display
from collections import namedtuple

from pymouse import PyMouseEvent
from operator import attrgetter

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

import threading

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

disp = Xlib.display.Display()
local_dpy = Xlib.display.Display()
root = disp.screen().root

NET_ACTIVE_WINDOW = disp.intern_atom('_NET_ACTIVE_WINDOW')
MyGeom = namedtuple('MyGeom', 'x y height width')

# Share the variable 'clicked' between threads
c = threading.Condition()
clicked = False

class Detection:
    def __init__(self, id, name, score, box, mask, x, y):
        self.id = id
        self.name = name
        self.score = score
        self.box = box
        self.mask = mask
        self.x = x
        self.y = y
    def __repr__(self):
        return repr([self.id, self.name, self.score, self.box, self.mask, self.x, self.y])

class updateClasses(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)

        # Subscribers
        # rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/tag_detections_image", Image, self.callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_img(self):
        result = self.image
        return result

    def getResult(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg



        if low < high: 
            pi = self.partition(target,low,high, y) 
    
            self.Sort_quick(target, low, pi-1, y) 
            self.Sort_quick(target, pi+1, high, y)

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

    def get_active_window(self):
        win_id = root.get_full_property(NET_ACTIVE_WINDOW,
                                        Xlib.X.AnyPropertyType).value[0]
        try:
            return disp.create_resource_object('window', win_id)
        except Xlib.error.XError:
            pass

    def get_absolute_geometry(self, win):
        """
        Returns the (x, y, height, width) of a window relative to the top-left
        of the screen.
        """
        geom = win.get_geometry()
        (x, y) = (geom.x, geom.y)
        while True:
            parent = win.query_tree().parent
            pgeom = parent.get_geometry()
            x += pgeom.x
            y += pgeom.y
            if parent.id == root.id:
                break
            win = parent
        return MyGeom(x, y, geom.height, geom.width)

    def get_window_bbox(self, win):
        """
        Returns (x1, y1, x2, y2) relative to the top-left of the screen.
        """
        geom = self.get_absolute_geometry(win)
        x1 = geom.x
        y1 = geom.y
        x2 = x1 + geom.width
        y2 = y1 + geom.height
        return (x1, y1, x2, y2)

    def get_rel_cursor(self, win_bbox, win_name):

        global clicked

        click = "No"

        data = Xlib.display.Display().screen().root.query_pointer()._data
        x = data["root_x"]
        y = data["root_y"]

        # if cursor is within the bounds of the arm_camera_objects window, compute its relative coordinates
        if x >= win_bbox[0] and x <= win_bbox[2] and y >= win_bbox[1] and y <= win_bbox[3] and win_name == "/arm_camera_objects":
            rel_x = x - win_bbox[0]
            rel_y = y - win_bbox[1]
            c.acquire()
            if clicked is True:
                click = "Yes"
            c.notify_all()
            c.release()
        else:
            rel_x = None
            rel_y = None
            click = "No"
        return (rel_x, rel_y), click

    def compute_updates(self, image, class_id, objects):
        # Based on clicked item, draw options

        # compute centroids
        centroids = list()
        # for i in [x for x in range(len(objects.class_ids)) if x != class_id]:
        for i in range(len(objects.class_ids)):
            x1 = objects.boxes[i].x_offset
            y1 = objects.boxes[i].y_offset
            x2 = x1 + objects.boxes[i].width
            y2 = y1 + objects.boxes[i].height
            xc = np.mean([x1,x2])
            yc = np.mean([y1,y2])
            centroids.append([xc,yc])

        centroids_to_avoid = list()
        for i in [x for x in range(len(objects.class_ids)) if x != class_id]:
            centroids_to_avoid.append(centroids[i])

        centroids_to_avoid = np.array(centroids_to_avoid)
        avoid = (int(np.mean(centroids_to_avoid[:,0])), int(np.mean(centroids_to_avoid[:,1])))
        print(avoid)
        return centroids, avoid

    def compute_masks(self, im, result, colors):
        masks = []
        masks_indices = []
        for i in range(len(result.class_ids)):
            # Obtain current object mask as a numpy array (black and white mask of single object)
            current_mask = self.br.imgmsg_to_cv2(result.masks[i])

            # Find current mask indices
            mask_indices = np.where(current_mask==255)

            # Add to mask indices list
            if len(masks_indices) > len(result.class_ids):
                masks_indices = []
            else:
                masks_indices.append(mask_indices)

            # Add to mask list
            if len(masks) > len(result.class_ids):
                masks = []
            else:
                masks.append(current_mask)

            # Select correct object color
            color = colors[result.class_ids[i]]

            # Change the color of the current mask object
            im[masks_indices[i][0], masks_indices[i][1], :] = color

        return im


        if low < high: 
            pi = self.partition(target, low, high) 
    
            self.Sort_quick(target, low, pi-1) 
            self.Sort_quick(target, pi+1, high)

    def multisort(self, msg):

        # convert detections to a list of object instances
        detections = list()
        for i in range(len(msg.class_ids)):
            detections.append(Detection(msg.class_ids[i], msg.class_names[i], msg.scores[i], msg.boxes[i], msg.masks[i], msg.boxes[i].x_offset, msg.boxes[i].y_offset))

        # sort by y, then by x
        result = sorted(detections, key=attrgetter('x', 'y'))

        return result

class Clickonacci(PyMouseEvent, threading.Thread):
    def __init__(self):
        PyMouseEvent.__init__(self)

    def click(self, x, y, button, press):

        global clicked

        if button == 1:
            if press:
                c.acquire()
                clicked = True
                c.notify_all()
                c.release()

                rospy.sleep(.1)

                c.acquire()
                clicked = False
                c.notify_all()
                c.release()
        else:  # Exit if any other mouse button used
            self.stop()

def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("mask_rcnn", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0
    
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/train")
    register_coco_instances("test_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/test/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/test")
    
    train_metadata = MetadataCatalog.get("train_set")
    print(train_metadata)
    dataset_dicts_train = DatasetCatalog.get("train_set")

    test_metadata = MetadataCatalog.get("test_set")
    print(test_metadata)
    dataset_dicts_test = DatasetCatalog.get("test_set")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_set")
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 1000 # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 classes (Plate, Carrot, Celery, Pretzel, Gripper)

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset/output/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("test_set")
    predictor = DefaultPredictor(cfg)

    class_names = MetadataCatalog.get("train_set").thing_classes

    # Set up custom cv2 visualization parameters
    # Classes: [name, id]
    #               -
    #          [Plate,   0]
    #          [Carrot,  1]
    #          [Celery,  2]
    #          [Pretzel, 3]
    #          [Gripper, 4]

    # Colors = [blue, green, red]
    color_plate = [0, 255, 0]       # green
    color_carrot = [255, 200, 0]    # blue
    color_celery = [0, 0, 255]      # red
    color_pretzel = [0, 220, 255]   # yellow
    color_gripper = [204, 0, 150]   # purple
    colors = list([color_plate, color_carrot, color_celery, color_pretzel, color_gripper])

    alpha = .4

    run = updateClasses()

    rospy.sleep(1)
    print("Running...")

    clicked = False

    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        # If no image, return to start of loop
        if img is None:
            continue

        # Get detections
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")

        # Get results
        unsorted = run.getResult(predictions, class_names)

        # Sort detections by x and y offsets
        sorted = run.multisort(unsorted)
        
        # Reorganize back into Result() object type
        # TODO: Change the rest of the code to use the above organization (by object). It works well for now, it just might speed it up.
        result = Result()
        for i in range(len(sorted)):
            result.class_ids.append(sorted[i].id)
            result.class_names.append(sorted[i].name)
            result.scores.append(sorted[i].score)
            result.boxes.append(sorted[i].box)
            result.masks.append(sorted[i].mask)

        # If no detections, return to start of loop
        if result is None:
            continue

        result_cls = result.class_names
        result_clsId = result.class_ids
        result_scores = result.scores
        result_masks = result.masks

        # Create copies of the original image
        im = img.copy()
        output = img.copy()

        # Compute Masks
        im = run.compute_masks(im, result, colors)

        # Draw object masks on image
        cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

        # Draw object bbox, class label, and score on image
        for i in range(len(result_clsId)):
            # Draw Bounding boxes
            start_point = (result.boxes[i].x_offset, result.boxes[i].y_offset)
            end_point = (result.boxes[i].x_offset + result.boxes[i].width, result.boxes[i].y_offset + result.boxes[i].height)
            start_point2 = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 2)
            end_point2 = (result.boxes[i].x_offset + result.boxes[i].width - 2, result.boxes[i].y_offset + 12)
            # color = colors[result_clsId[i]]
            color = [0, 0, 0]
            box_thickness =  1

            name = result_cls[i]
            score = result_scores[i]
            conf = round(score.item() * 100, 1)

            # Test strategy for updating model classes
            if score > .9:
                string = str(name) + ":" + str(conf) + "%"
            else:
                string = str(name) + "??? - " + str(conf) + "%"

            try:
                win = run.get_active_window()
                win_name = win.get_wm_name()
                win_bbox = run.get_window_bbox(win)
                # print(win_name, win_bbox)
            except Xlib.error.BadWindow:
                print("Window vanished")
            
            rel_cursor, click = run.get_rel_cursor(win_bbox, win_name)

            # if cursor is within object bbox and a click, do something
            if rel_cursor[0] is not None and rel_cursor[1] is not None:
                if rel_cursor[0] >= start_point[0] and rel_cursor[0] <= end_point[0] and rel_cursor[1] >= start_point[1] and rel_cursor[1] <= end_point[1]:
                    color = [0, 220, 255] # yellow
                    box_thickness = 2
                    contain_id = i
                else:
                    color = [0, 0, 0]
                    contain_id = None
                    box_thickness = 1

                if contain_id is not None and click == "Yes":
                    print("Object " + str(i) + " (" + str(name) + ") was clicked!!!")
                    rospy.sleep(.05) # prevent multiple clicks detected
                    # centroids, avoid = run.compute_updates(output, contain_id, result)
                    clicked = True
                    selected_id = contain_id

            # Draw
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 10)
            fontScale = .3
            text_thickness = 1
            output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
            output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
            output = cv2.putText(output, string, org, font, fontScale, [255, 255, 255], text_thickness, cv2.LINE_AA, False)
        
        if clicked:
            # Change selected bbox color to green
            output = cv2.rectangle(output, (result.boxes[selected_id].x_offset, result.boxes[selected_id].y_offset), \
                (result.boxes[selected_id].x_offset + result.boxes[selected_id].width, result.boxes[selected_id].y_offset + result.boxes[selected_id].height), \
                [0, 255, 0], 2)
            
            # Display options to the right of the bbox
            start_x = result.boxes[selected_id].x_offset + result.boxes[selected_id].width + 20
            start_y = result.boxes[selected_id].y_offset - 20

            x = start_x
            y = start_y + 20

            for i in range(len(class_names)):
                y += 20
                output = cv2.rectangle(output, (x, y), (x+40, y+12), [255, 255, 255], -1)
                output = cv2.putText(output, class_names[i], (x+2, y+8), font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)

            # The below is potentially for a more robust way of displaying class update options
            # for i in range(len(centroids)):
            #     if i == selected_id:
            #         circle_color = [0, 0, 255]
            #     else:
            #         circle_color = [0, 255, 0]
            #     output = cv2.circle(output, (int(centroids[i][0]), int(centroids[i][1])), 2, circle_color, 2)
            # output = cv2.circle(output, avoid, 2, [255, 0, 0], 2)
            # output = cv2.arrowedLine(output, avoid, (int(centroids[selected_id][0]), int(centroids[selected_id][1])), [255,0,0], 1)  # y-axis (tag frame)

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        # # Display Image Counter
        # # image_counter = image_counter + 1
        # # if (image_counter % 11) == 10:
        # #     rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

        run.publish(im_msg, result)    
        
    return 0

if __name__ == '__main__':

    # Run mouse click callbakc in a separate thread
    e = Clickonacci()
    e.setDaemon(True) 
    e.start()

    # Main program
    sys.exit(main())