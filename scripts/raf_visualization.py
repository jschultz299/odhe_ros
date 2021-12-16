#!/usr/bin/env python

# This program publishes arm_camera_image topic for RAF study
# Created by Jack Schultz
# 8/06/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

import rospy
import sys
from cv_bridge import CvBridge
from odhe_ros.msg import Result
from sensor_msgs.msg import Image, RegionOfInterest

from Xlib import display

class raf_visualization(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('raf_visualization', Image, queue_size=10)

        # Subscribers
        rospy.Subscriber("/arm_camera_results", Result, self.detections_callback)
        rospy.Subscriber("/camera2/color/image_raw", Image, self.image_callback)

        # Subscriber for study state
            # 0 for before start button pressed
            # 1 for after start button pressed, before food item selected
            # 2 for after food item selected, before food item picked up
            # 3 for after food item picked up, before facial keypoint detection is started
            # 4 for after facial keypoint detection is started, before food item transfer
            # 5 for during food item transfer, before food item in position in front of mouth
            # 6 for after food item in position in front of mouth, before food delivered
            # 7 for after food item has been delivered, before return to start position

        # rospy.Subscriber("/raf_state", int, self.state_callback)

    # Subscriber Callbacks
    def image_callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_image(self):
        result = self.image
        return result

    def detections_callback(self, msg):
        self.detections = msg

    def get_detections(self):
        result = self.detections
        return result

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

    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y

    def publish(self, img):
        self.pub.publish(img)
        self.loop_rate.sleep()

def main():
    """ Visualize Food Items detected by Mask RCNN """

    rospy.init_node("raf_visualization", anonymous=True)
    bridge = CvBridge()

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

    run = raf_visualization()
    rospy.sleep(1.0)

    # Placeholder for state
    # TODO: Make this dynamic
    state = 1

    verbose = False

    print("Runnning. Press Cntl+C to Quit.")

    trial = 1
    while not rospy.is_shutdown():
        if verbose:
            print("\n----------------------")
            print("Frame: ", trial)

        # Get images
        img = run.get_image()
        detections = run.get_detections()

        if detections is not None and img is not None:

            # Visualize using custom cv2 code
            detections_cls = detections.class_names
            detections_clsId = detections.class_ids

            num_detections = len(detections_cls)

            item_ids = list(detections.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]         # returns index of food items

            if verbose:
                print("\nDetections: ", detections_cls)
                print("Detection IDs: ", detections_clsId)
                # print("Food Item Scores: ", detections_scores)
                print("Idx: ", idx)

            # Randomly choose item to be selecte
            # TODO: Make this happen based on experiment state
            # num_food_items = len(idx)

            # if state == 2:
            #     new_selection_idx = detections_clsId.index(selection_Id)
            #     selection_idx = new_selection_idx

            # if state  == 1:
            #     selection_idx = random.choice(idx)
            #     selection_Id = detections_clsId[selection_idx]
            #     state = 2

            # print("Selection Index = ", selection_idx)
            # print("Selection Class Id = ", selection_Id)

            # Get Mouse Cursor Position
            data = display.Display().screen().root.query_pointer()._data
            # print("(", data["root_x"], ", ", data["root_y"], ")")

            # Create copies of the original image
            output = img.copy()

            for i in range(len(idx)):
                ul_x = detections.boxes[idx[i]].x_offset
                ul_y = detections.boxes[idx[i]].y_offset
                br_x = ul_x + detections.boxes[idx[i]].width
                br_y = ul_y + detections.boxes[idx[i]].height

                # Full screen desktop
                # X1 = run.linear_map(0, 640, 604, 1955, ul_x)
                # X2 = run.linear_map(0, 640, 604, 1955, br_x)
                # Y1 = run.linear_map(0, 480, 66, 1079, ul_y)
                # Y2 = run.linear_map(0, 480, 66, 1079, br_y)

                # GUI on desktop
                X1 = run.linear_map(0, 640, 361, 1561, ul_x)
                X2 = run.linear_map(0, 640, 361, 1561, br_x)
                Y1 = run.linear_map(0, 480, 166, 1065, ul_y)
                Y2 = run.linear_map(0, 480, 166, 1065, br_y)

                # print("(", X1, ", ", Y1, ")")
                # print("(", X2, ", ", Y2, ")")

                if data["root_x"] > X1 and data["root_x"] < X2 and data["root_y"] > Y1 and data["root_y"] < Y2:
                    color = [0, 220, 255]
                    thickness = 2
                else:
                    color =  [0,0,0]
                    thickness = 1

                # Draw Bounding boxes
                start_point = (ul_x, ul_y)
                end_point = (br_x, br_y)
                output = cv2.rectangle(output, start_point, end_point, color, thickness)
            # print("--------------")

            # Draw highlighted box
            # TODO: Current problem is it flips between items of same class
            # start_point = (detections.boxes[selection_idx].x_offset, detections.boxes[selection_idx].y_offset)
            # end_point = (detections.boxes[selection_idx].x_offset + detections.boxes[selection_idx].width, detections.boxes[selection_idx].y_offset + detections.boxes[selection_idx].height)
            # output = cv2.rectangle(output, start_point, end_point, [0, 220, 255], 2)

            im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        run.publish(im_msg)
        trial = trial + 1    
        

    return 0

if __name__ == '__main__':
    sys.exit(main())