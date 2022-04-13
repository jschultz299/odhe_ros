#!/usr/bin/env python

# This program is the main file for data collection 
# Written by Jack Schultz
# Contributed to by Jinlong Li
# Created 7/20/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

### Imports ###
import struct
import sys
import math
from xmlrpc.client import Boolean
import cv2
import random

import rospy
import numpy as np
import baxter_interface
import pickle

import threading
import signal
import os

from operator import attrgetter
from statistics import mean
from datetime import datetime
from matplotlib import path
from odhe_ros.msg import Result, DltParams
from cv_bridge import CvBridge
from Xlib import display
from sensor_msgs.msg import Image
from geometry_msgs.msg import (
    PoseStamped,
    Pose
)
from std_msgs.msg import Header, String, Bool
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

### Class Defenition Example ###
# class Detection:
#     def __init__(self, id, name, score, box, mask, x, y):
#         self.id = id
#         self.name = name
#         self.score = score
#         self.box = box
#         self.mask = mask
#         self.x = x
#         self.y = y
#     def __repr__(self):
#         return repr([self.id, self.name, self.score, self.box, self.mask, self.x, self.y])

### E-Stop Thread ###
class ExitCommand(Exception):
    pass

def signal_handler(signal, frame):
    raise ExitCommand()

def thread_job(run):
    # Trigger the kill signal
    estop_pressed = False
    myfile = open("/home/labuser/raf/participant_data/cursor_data.txt", 'w')
    while not estop_pressed:
        estop_pressed = RAF_dataCollection.estop(run, myfile)

    myfile.close()
    os.kill(os.getpid(), signal.SIGUSR1)

### Main Class Definition ###
class RAF_dataCollection():
    def __init__(self, limb, verbose=True):
        # Parameters
        self.bridge = CvBridge()
        self.timer_running = False
        self.timer = 0
        self.timer_start = 0
        self.etimer_running = False
        self.etimer = 0
        self.etimer_start = 0
        self.RAF_msg = "Loading..."
        self.selected = False
        self.cursor_angle = 0
        self.dwell_time = 2         # Time to fill cursor and select item
        self.delay_time = .5        # Time before cursor begins to fill
        self.estop_dwell_time = 1   # Time to fill cursor and select e-stop
        self.estop_delay_time = .5  # Time before cursor begins to fill on e-stop
        self.hl_id = None
        self.image_flag = 0         # 0 for raw image, 1 for item selection image, 2 for face detection
        self.mouth_open = False
        self.food_set = 0
        self.food_num = 0

        self._limb_name = limb
        self._verbose = False
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        left = baxter_interface.Gripper('left', baxter_interface.CHECK_VERSION)
        right = baxter_interface.Gripper('right', baxter_interface.CHECK_VERSION)
        if self._verbose:
            print("Getting robot state... ")
            print("Enabling robot... ")
        self._rs.enable()
        if self._verbose:
            print("Calibrating gripper...")
        left.calibrate()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Subscribers
        self.sub_detections = rospy.Subscriber('arm_camera_results', Result, self.detections_callback)
        self.sub_dlt = rospy.Subscriber('dlt_params', DltParams, self.dlt_callback)
        self.sub_depth = rospy.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.sub_image = rospy.Subscriber("/camera2/color/image_raw", Image, self.image_callback)
        self.sub_face = rospy.Subscriber("/face_detections", Image, self.face_callback)
        self.sub_mouth = rospy.Subscriber("/mouth_open", Bool, self.mouth_callback)

        # Publishers
        self.pub_vis = rospy.Publisher('raf_visualization', Image, queue_size=10)
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)
        self.pub_cursor = rospy.Publisher('raf_cursor_angle', String, queue_size=10)

    ### Subscriber Callbacks ###
    def image_callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

        output = self.image.copy()

        # Draw Current State in the corner of the image
        info_str1 = "Set: " + str(self.food_set)
        info_str2 = "Item: " + str(self.food_num)
        output = cv2.putText(output, info_str1, (20, 450), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)
        output = cv2.putText(output, info_str2, (20, 470), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        self.image_msg = self.bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        try:
            self.publish()
        except:
            if self._verbose:
                print("Getting image...")

    def face_callback(self, msg):
        self.face_image = self.convert_to_cv_image(msg)

        output = self.face_image.copy()

        # Draw Current State in the corner of the image
        info_str1 = "Set: " + str(self.food_set)
        info_str2 = "Item: " + str(self.food_num)
        output = cv2.putText(output, info_str1, (20, 450), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)
        output = cv2.putText(output, info_str2, (20, 470), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        self.face_msg = self.bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

    def mouth_callback(self, msg):
        self.mouth_open = msg.data
    
    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header
    
    def detections_callback(self, msg):
        # Receive detections and sort them according to y-offset

        # temp = self.multisort(msg)
        temp = self.sort_detections(msg)
        
        # Reorganize back into Result() object type
        # TODO: Change the rest of the code to use the above organization (by object). It works well for now, it just might speed it up.
        # self.detections = Result()
        # for i in range(len(temp)):
        #     self.detections.class_ids.append(temp[i].id)
        #     self.detections.class_names.append(temp[i].name)
        #     self.detections.scores.append(temp[i].score)
        #     self.detections.boxes.append(temp[i].box)
        #     self.detections.masks.append(temp[i].mask)

        self.detections = Result()
        for i in range(len(temp)):
            self.detections.class_ids.append(temp[i][0])
            self.detections.class_names.append(temp[i][1])
            self.detections.scores.append(temp[i][2])
            self.detections.boxes.append(temp[i][3])
            self.detections.masks.append(temp[i][4])

        # TODO: Instead of sorting, we will use a deep sort machine learning method to track specific food items
        # Sort detections based on box position in image

    def dlt_callback(self, msg):
        self.dlt = msg

    # Conversion Functions for Image Callbacks
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

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    # Get callback info in main
    def get_image(self):
        result = self.image
        return result

    def get_depth_array(self):
        result = self.depth_array
        return result

    def get_detections(self):
        result = self.detections
        return result

    ### Main Functions ###
    def sort_detections(self, msg):
        # Sort detections by y-position of upper left bbox corner
        # TODO: Sort by y-position first and then sort again by x-position
        # This will prevent object ids from flipping back and forth if they are at the same y-position

        target = self.Out_transfer(msg.class_ids, msg.class_names, msg.scores, msg.boxes, msg.masks)

        # Sort by y-offset
        self.Sort_quick(target, 0, len(target)-1)

        # Sort by x-offset
        # self.Sort_quick(target, 0, len(target)-1)

        return target

    def Out_transfer(self, class_id, class_name, score, box, mask):

        num = int(len(class_id))
        target = []

        for i in range(num):

            target.append([class_id[i], class_name[i], score[i], box[i], mask[i]])

        return target

    def partition(self, target, low, high):

        i = ( low-1 )
        arr = []
        arr = [target[w][3].y_offset for w in range(len(target))]

        pivot = arr[high]

        for j in range(low , high): 
    
            if   arr[j] <= pivot: 
            
                i = i+1 
                target[i],target[j] = target[j],target[i] 
    
        target[i+1],target[high] = target[high],target[i+1] 

        return ( i+1 )

    def Sort_quick(self, target, low, high):

        if low < high: 
            pi = self.partition(target, low, high) 
    
            self.Sort_quick(target, low, pi-1) 
            self.Sort_quick(target, pi+1, high)

    def estop(self, file):
        estop_pressed = False

        data = display.Display().screen().root.query_pointer()._data
        # p = path.Path([(120,805), (234,805), (306, 878), (306,991), (234,1064), (121,1064), (48,991), (48,878)])      # Define Octogon (left side big)
        p = path.Path([(1737,865), (1824,865), (1880,920), (1880,1008), (1824,1064), (1737,1064), (1680,1008), (1680,920)])      # Define Octogon (right side small)
        contains = p.contains_points([(data["root_x"], data["root_y"])])
        cur_time = rospy.get_time()
        line = str(cur_time) + "," + str(data["root_x"]) + "," + str(data["root_y"])
        file.write(line + "\n")

        # Write cursor positions to file

        # Check if cursor is inside the stop sign
        if contains:
            # Cursor enters box
            if not self.etimer_running:
                self.etimer_start = rospy.get_time()
                self.etimer_running = True
                self.cursor_angle = self.linear_map(self.estop_delay_time, self.estop_dwell_time, 0, 2*math.pi, np.clip(self.etimer, self.estop_delay_time, self.estop_dwell_time))
                self.pub_cursor.publish(str(self.cursor_angle))

            # Cursor remains in box
            if self.etimer_running:
                self.etimer = rospy.get_time() - self.etimer_start
                self.cursor_angle = self.linear_map(self.estop_delay_time, self.estop_dwell_time, 0, 2*math.pi, np.clip(self.etimer, self.estop_delay_time, self.estop_dwell_time))
                self.pub_cursor.publish(str(self.cursor_angle))

        else:
            # Cursor leaves box
            if self.etimer_running:
                self.etimer_running = False
                self.etimer_start = 0
                self.etimer = 0
                self.cursor_angle = self.linear_map(self.estop_delay_time, self.estop_dwell_time, 0, 2*math.pi, np.clip(self.etimer, self.estop_delay_time, self.estop_dwell_time))
                self.pub_cursor.publish(str(self.cursor_angle))

        if self.etimer > self.estop_dwell_time + self.estop_delay_time:
            # print("Object Selected!")
            estop_pressed = True
            self.etimer_running = False
            self.etimer_start = 0
            self.etimer = 0

            self.cursor_angle = self.linear_map(self.estop_delay_time, self.estop_dwell_time, 0, 2*math.pi, np.clip(self.etimer, self.estop_delay_time, self.estop_dwell_time))
            self.pub_cursor.publish(str(self.cursor_angle))

            raf_msg = "E-stop Pressed."
            self.change_raf_message(raf_msg)
            self.publish()

            print("E-stop Pressed. Exiting Program.")
            
        return estop_pressed

    def compute_position(self, x, y):
        # Divide pixels by camera resolution for better numerical stability
        x = x / 640
        y = y / 480

        # This function computes (X,Y) meters from (x,y) pixels
        X = ( (self.dlt.P2 + x*self.dlt.P8)*(self.dlt.P6 + y) - (self.dlt.P5 + y*self.dlt.P8)*(self.dlt.P3 + x) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )
        Y = ( (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P3 + x) - (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P6 + y) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )

        return X, Y

    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y

    def item_selection(self, item, item_cls):
        raf_msg = "Select Food item " + str(item + 1) + ": " + str(item_cls)
        self.change_raf_message(raf_msg)
        self.selected = False
        selected_item = None

        self.image_flag = 1

        # Move the cursor off the image to prevent unintentional selections
        d = display.Display()
        s = d.screen()
        root = s.root
        root.warp_pointer(100,540)
        d.sync()

        while not self.selected:
            if self.detections is not None and self.image is not None:

                item_ids = list(self.detections.class_ids)
                idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]         # returns index of food items

                if self._verbose:
                    print("\nDetections: ", self.detections.class_names)
                    print("Detection IDs: ", self.detections.class_ids)
                    # print("Food Item Scores: ", detections_scores)
                    print("Idx: ", idx)

                # Get Mouse Cursor Position
                data = display.Display().screen().root.query_pointer()._data
                if self._verbose:
                    print("(", data["root_x"], ", ", data["root_y"], ")")

                # Create a copy of the original image
                output = self.image.copy()

                for i in range(len(idx)):
                    ul_x = self.detections.boxes[idx[i]].x_offset
                    ul_y = self.detections.boxes[idx[i]].y_offset
                    br_x = ul_x + self.detections.boxes[idx[i]].width
                    br_y = ul_y + self.detections.boxes[idx[i]].height

                    # Full screen desktop
                    # X1 = run.linear_map(0, 640, 604, 1955, ul_x)
                    # X2 = run.linear_map(0, 640, 604, 1955, br_x)
                    # Y1 = run.linear_map(0, 480, 66, 1079, ul_y)
                    # Y2 = run.linear_map(0, 480, 66, 1079, br_y)

                    # GUI on desktop
                    X1 = self.linear_map(0, 640, 361, 1561, ul_x)
                    X2 = self.linear_map(0, 640, 361, 1561, br_x)
                    Y1 = self.linear_map(0, 480, 166, 1065, ul_y)
                    Y2 = self.linear_map(0, 480, 166, 1065, br_y)

                    # Check if cursor is inside the bounding box
                    # TODO: need to also check if cursor has been stable for a length of time
                    if data["root_x"] > X1 and data["root_x"] < X2 and data["root_y"] > Y1 and data["root_y"] < Y2:
                        color = [0, 220, 255]
                        thickness = 2

                        # return id of highlighted item
                        self.hl_id = i

                        # Cursor enters box
                        if not self.timer_running and self.hl_id == i:
                            self.timer_start = rospy.get_time()
                            self.timer_running = True
                            self.cursor_angle = self.linear_map(self.delay_time, self.dwell_time, 0, 2*math.pi, np.clip(self.timer, self.delay_time, self.dwell_time))
                            self.pub_cursor.publish(str(self.cursor_angle))

                        # Cursor remains in box
                        if self.timer_running and self.hl_id == i:
                            self.timer = rospy.get_time() - self.timer_start
                            self.cursor_angle = self.linear_map(self.delay_time, self.dwell_time, 0, 2*math.pi, np.clip(self.timer, self.delay_time, self.dwell_time))
                            self.pub_cursor.publish(str(self.cursor_angle))

                    else:
                        color =  [0,0,0]
                        thickness = 1

                        # Cursor leaves box
                        if self.timer_running and self.hl_id == i:
                            self.timer_running = False
                            self.timer_start = 0
                            self.timer = 0
                            self.cursor_angle = self.linear_map(self.delay_time, self.dwell_time, 0, 2*math.pi, np.clip(self.timer, self.delay_time, self.dwell_time))
                            self.pub_cursor.publish(str(self.cursor_angle))

                    if self.timer > self.dwell_time + self.delay_time:
                        # print("Object Selected!")
                        self.timer_running = False
                        self.timer_start = 0
                        self.timer = 0
                        self.cursor_angle = self.linear_map(self.delay_time, self.dwell_time, 0, 2*math.pi, np.clip(self.timer, self.delay_time, self.dwell_time))
                        self.pub_cursor.publish(str(self.cursor_angle))

                        self.selected = True
                        selected_item = i
                        selected_item_cls = self.detections.class_names[selected_item]
                        raf_msg = "Food item " + str(i+1) + " selected: " + str(selected_item_cls)
                        self.change_raf_message(raf_msg)
                        if selected_item == item:
                            color = [0, 255, 0]
                        else:
                            color = [0, 0, 255]
                        thickness = 2

                    # Draw Bounding boxes
                    start_point = (ul_x, ul_y)
                    end_point = (br_x, br_y)
                    output = cv2.rectangle(output, start_point, end_point, color, thickness)
                    output = cv2.putText(output, str(i+1), (ul_x + 2, br_y - 3), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,255,255), 1, cv2.LINE_AA)

                    # Draw Current State in the corner of the image
                    info_str1 = "Set: " + str(self.food_set)
                    info_str2 = "Item: " + str(self.food_num)
                    output = cv2.putText(output, info_str1, (20, 450), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)
                    output = cv2.putText(output, info_str2, (20, 470), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,255), 1)

                im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                self.im_msg = self.bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        if self.selected:
            rospy.sleep(0.5)
        self.image_flag = 0

        return selected_item, selected_item_cls

    def acquire_item_pose(self, item, item_class, det):
        # Get Transformation between camera and gripper (from calibrate_camera.py)
        with open('/home/labuser/ros_ws/src/odhe_ros/scripts/Transformation.npy', 'rb') as f:
            T = np.load(f)
        f.close()

        mask = self.bridge.imgmsg_to_cv2(det.masks[item])

        ret, thresh = cv2.threshold(mask, 0, 255, 0)

        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        rotrect = cv2.minAreaRect(cntrs[0])
        box = cv2.boxPoints(rotrect)
        box = np.int0(box)

        height = np.linalg.norm(box[0] - box[3])
        width = np.linalg.norm(box[0] - box[1])

        img = mask.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.drawContours(img, [box], 0, (0,0,255), 2)
        
        angle = rotrect[-1]

        # if angle < -45:
        #     angle = -(90 + angle)
        # else:
        #     angle = -angle

        # if height > width:
        #     # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
        #     gripper_angle = abs(angle - 180)
        #     mid = (int((box[3][0] + box[2][0]) / 2), int((box[3][1] + box[2][1]) / 2))
        # else:
        #     gripper_angle = abs(angle - 90)
        #     mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))

        # # Create a deadzone that defaults to a grip angle of 0 degrees
        # if (gripper_angle > 0 and gripper_angle <= 3):
        #     gripper_angle = 0
        # elif (gripper_angle > 177 and gripper_angle <= 180):
        #     gripper_angle = 0

        if height > width:
            # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
            gripper_angle = abs(angle)
            mid = (int((box[0][0] + box[1][0]) / 2), int((box[0][1] + box[1][1]) / 2))
        else:
            gripper_angle = abs(angle) + 90
            mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))

        centroid = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))
        grasp_point = (int((centroid[0] + mid[0]) / 2), int((centroid[1] + mid[1]) / 2))
        grasp_point2 = (int((centroid[0] + grasp_point[0]) / 2), int((centroid[1] + grasp_point[1]) / 2))

        if item_class == "pretzel":
            x = grasp_point2[0]
            y = grasp_point2[1]
        elif item_class == "carrot":
            x = grasp_point2[0]
            y = grasp_point2[1]
        elif item_class == "celery":
            x = grasp_point2[0]
            y = grasp_point2[1]
        else:
            x = grasp_point[0]
            y = grasp_point[1]
            print("Selected Item is not a Food Item.")

        ### Draw grasp point on image ###
        self.image_flag = 1
        output = self.image.copy()

        output = cv2.arrowedLine(output, mid, centroid, (0,255,0), 2)
        output = cv2.circle(output, (x, y), 2, (255, 0, 0), -1)

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        self.im_msg = self.bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        rospy.sleep(2.0)

        self.image_flag = 0

        ### ------------------------- ###

        ### Find average depth value across object mask ###
        coord = cv2.findNonZero(mask)
        depth_list = []
        for ii in range(len(coord)):
            if int(self.depth_array[int(coord[ii][0][1]), int(coord[ii][0][0])]) == 0:
                print("Zero depth for coordinate: ", [int(coord[ii][0][1]), int(coord[ii][0][0])])
            else:
                depth_list.append(int(self.depth_array[int(coord[ii][0][1]), int(coord[ii][0][0])]))

        depth_avg = np.mean(depth_list)
        depth_avg = depth_avg / 1000

        ### ------------------------- ###
        
        X, Y = self.compute_position(x, y)
        depth = int(self.depth_array[int(centroid[1]), int(centroid[0])])   # depth_array index reversed for some reason
        depth = depth / 1000
        point_camera = np.array([[X], [Y], [depth_avg], [1]])

        point_robot = np.matmul(T, point_camera)

        print("    - Robot Position: (X: " + str(point_robot[0][0]) + ", Y: " + str(point_robot[1][0]) + ", Z: " + str(point_robot[2][0]) + ")")
        print("    - Gripper Angle: " + str(gripper_angle))

        raf_msg = "Picking up Food Item " + str(item + 1)
        self.change_raf_message(raf_msg)

        return point_robot, gripper_angle

    def acquire_item(self, joint_angles, point_robot, item, table_height):

        self.move_to_angles(joint_angles)

        # The height adjust value lowers the gripper by the specified amount in meters
        if item == "pretzel":
            height_adjust = .003       # .0025
        elif item == "carrot":
            height_adjust = .01        # .010
        elif item == "celery":
            height_adjust = .0035        # .005
        else:
            "Unknown acquire_item error."
            height_adjust = 0

        acquire_x = point_robot[0][0]
        acquire_y = point_robot[1][0]
        # acquire_z = point_robot[2][0] - height_adjust
        acquire_z = point_robot[2][0]

        # Make sure we don't hit the table
        if acquire_z < table_height:
            print("Possible table collision detected. Pose adjusted.")

        while acquire_z < table_height:
            acquire_z = acquire_z + .001        # Raise z-coord by 1mm until above table height

        current_pose = self.get_current_pose()

        pose = Pose()
        pose.position.x = acquire_x
        pose.position.y = acquire_y
        pose.position.z = acquire_z + .05
        pose.orientation = current_pose['orientation']

        self._servo_to_pose(pose)

        pose.position.z = acquire_z

        self._servo_to_pose(pose)

        self.gripper_close()

        pose.position.z = point_robot[2][0] + .1

        self._servo_to_pose(pose)

    def check_grasp(self):

        # Try detections from multiple frames to get 2 grippers and at least 1 food item. If not, inconclusive
        check = False
        for i in range(30):
            if not check:
                self.loop_rate.sleep()

                det = self.get_detections()

                item_ids = list(det.class_ids)
                gripper_idx = [i for i, e in enumerate(item_ids) if e == 4 ]                # Grippers
                food_item_idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]     # Food Items

                numGrippers = len(gripper_idx)
                numFoods = len(food_item_idx)

                if numGrippers == 2 and numFoods > 0:
                    check = True
                    break
            else:
                pass
                
        if check:
            # Figure out which gripper is to the left (could be solved by proper sorting)
            if det.boxes[gripper_idx[0]].x_offset < det.boxes[gripper_idx[1]].x_offset:
                left_gripper_idx = gripper_idx[0]
                right_gripper_idx = gripper_idx[1]
            else:
                left_gripper_idx = gripper_idx[1]
                right_gripper_idx = gripper_idx[0]

            # Gripper Bounding Boxes
            left_gripper_ul_x = det.boxes[left_gripper_idx].x_offset
            left_gripper_ul_y = det.boxes[left_gripper_idx].y_offset
            left_gripper_br_x = left_gripper_ul_x + det.boxes[left_gripper_idx].width
            left_gripper_br_y = left_gripper_ul_y + det.boxes[left_gripper_idx].height

            right_gripper_ul_x = det.boxes[right_gripper_idx].x_offset
            right_gripper_ul_y = det.boxes[right_gripper_idx].y_offset
            right_gripper_br_x = right_gripper_ul_x + det.boxes[right_gripper_idx].width
            right_gripper_br_y = right_gripper_ul_y + det.boxes[right_gripper_idx].height

            # Interest Region between grippers
            region_x1 = int((left_gripper_ul_x + left_gripper_br_x) / 2)
            region_x2 = int((right_gripper_ul_x + right_gripper_br_x) / 2)
            region_ul = (region_x1, int(left_gripper_ul_y))
            region_ur = (region_x2, int(right_gripper_ul_y))
            region_br = (region_x2, int(right_gripper_br_y))
            region_bl = (region_x1, int(left_gripper_br_y))

            p = path.Path([region_ul, region_ur, region_br, region_bl])

            # Check if any food items are in this region
            for i in range(len(food_item_idx)):
                # Point based on segmentation mask
                mask = self.bridge.imgmsg_to_cv2(det.masks[i])
                ret, thresh = cv2.threshold(mask, 0, 255, 0)

                cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

                rotrect = cv2.minAreaRect(cntrs[0])
                box = cv2.boxPoints(rotrect)
                box = np.int0(box)

                height = np.linalg.norm(box[0] - box[3])
                width = np.linalg.norm(box[0] - box[1])

                if height > width:
                    # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
                    interest_point_x = int((box[0][0] + box[1][0]) / 2)
                    interest_point_y = int((box[0][1] + box[1][1]) / 2)
                else:
                    interest_point_x = int((box[3][0] + box[0][0]) / 2)
                    interest_point_y = int((box[3][1] + box[0][1]) / 2)
                
                # Point based off bounding box
                # interest_point_x = int(((2 * det.boxes[food_item_idx[i]].x_offset) + det.boxes[food_item_idx[i]].width) / 2)
                # interest_point_y = int(det.boxes[food_item_idx[i]].y_offset + (0.95 * det.boxes[food_item_idx[i]].height))
                contains = p.contains_points([(interest_point_x, interest_point_y)])
                if contains:
                    grasped_item_class = det.class_names[food_item_idx[i]]
                    grasped = True
                    break
                else:
                    grasped_item_class = None
                    grasped = False
        else:
            grasped = None
            grasped_item_class = "No Food Items Detected"

        return grasped, grasped_item_class

    def deliver_item(self, joint_angles):
        # Robot moves to face the user
        self.move_to_angles(joint_angles)

    def trigger_transfer(self):

        self.image_flag = 2

        # Robot waits for the user to open their mouth, then it releases the food item
        raf_msg = "Loading..."
        self.change_raf_message(raf_msg)

        # rospy.sleep(0.5)

        raf_msg = "Open your mouth when ready"
        self.change_raf_message(raf_msg)

        self.mouth_open = False

        timer_start = rospy.get_time()
        success = None

        while success is None:
            self.loop_rate.sleep()
            timer = rospy.get_time() - timer_start
            if self.mouth_open:
                success = 1
            elif not self.mouth_open and timer > 10:
                success = 0
            else:
                success = None

        rospy.sleep(0.5)
        self.image_flag = 0
        
        return success

    def transfer_item(self, pre_transfer_angles, transfer_angles):
        # Robot orients the gripper for easy food item hand-off, approaches the mouth, then releases the food item
        raf_msg = "Transferring Food Item"
        self.change_raf_message(raf_msg)

        self.move_to_angles(pre_transfer_angles)

        self.move_to_angles(transfer_angles)

        raf_msg = "Releasing Food Item"
        self.change_raf_message(raf_msg)

        # Release Delay
        rospy.sleep(1.0)

        self.gripper_open()

        raf_msg = "Food Item Delivered"
        self.change_raf_message(raf_msg)

    def retract(self):
        current_pose = self.get_current_pose()

        pose = Pose()
        pose.position.x = current_pose['position'][0] - .1
        pose.position.y = current_pose['position'][1] - .1
        pose.position.z = current_pose['position'][2]
        pose.orientation = current_pose['orientation']

        self._servo_to_pose(pose)

    def change_raf_message(self, msg):
        self.RAF_msg = msg
        # print("Message: " + self.RAF_msg)

    def spinOnce(self):
        r = rospy.Rate(10)
        r.sleep()

    def publish(self):
        if self.image_flag == 2:
            self.pub_vis.publish(self.face_msg)
        elif self.image_flag == 1:
            self.pub_vis.publish(self.im_msg)
        else:
            self.pub_vis.publish(self.image_msg)

        self.pub_msg.publish(self.RAF_msg)
        # self.pub_cursor.publish(str(self.cursor_angle))
        self.loop_rate.sleep()

    ### Robot Functions ###
    def move_to_angles(self, start_angles=None):
        # print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        rospy.sleep(0.1)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
    
    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(0.5)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(0.5)

    def get_current_pose(self):
        # Get the current pose of the robot
        current_pose = self._limb.endpoint_pose()
        return current_pose

    def get_current_joint_angles(self):
        current_joint_angles = self._limb.joint_angles()
        return current_joint_angles
           
def main(run):
    """ODHE Project First Experiment: Feeding food items to SCI Individuals

    This is experiment took place Summer 2021

    This code is meant to be run with the following experimental setup:
    The subject is seated at a table in a wheelchair with a monitor mounted on the chair.
    A Tobii Eyetracker 4 is  positioned on the monitor such that the subject can control
    a mouse cursor on the monitor with their eye movements. A plate of sparsely distributed 
    food items (carrots, celery, and pretzel rods) is placed on the table. The Baxter Robot
    is positioned across the table from the subject. An Intel L515 Lidar camera is fixed to
    Baxter's wrist. The output from this camera is fed through a detectron2 maskRCNN network
    which detects the food items as well as the plate and the robot's gripper. The Lidar
    camera's output is also fed through a Facial Keypoint Detection network which detects the
    subject's face for feeding.

    The following steps must be performed prior to data collection:
    1) Run calibration.py to create the transformation between the image plane and the robot frame
    2) Initialize the facial keypoint detection
    3) Set the food item transfer position
    
    The following steps detail the data collection procedure (repeated for each food item):
    4) Food Item Selection
    5) Food Item Acquisition
    6) Food Item Delivery
    7) Food Item Transfer
    8) Return to Home
    """

    # RAF Parameters
    state = 1               # determines experimental logic
    sets_till_break = 5     # Number of food item sets until participant can take a break

    num_sets = 10   # Number of trials to run. Each trial should have 6 food items.
                   # About 18 trials (108) food items should take an hour (non-stop)
    food_items_per_set = 6
    num_food_items = num_sets * food_items_per_set

    # Set debug to True to move robot to starting joint angles and then just spin
    debug = False

    bridge = CvBridge()
    limb = 'left'

    # Read home position set in Calibration.py
    with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
        home_joint_angles = pickle.load(handle)
    handle.close()

    wrist_orientation = home_joint_angles['left_w2']

    # Initialize Class
    # run = RAF_dataCollection(limb)

    # Move to home position
    run.move_to_angles(home_joint_angles)

    # Move to starting location
    run.move_to_angles(home_joint_angles)
    run.gripper_open()

    print("\n")
    participantID = input("Enter Participant ID (e.g. '04'): ")
    print("Participant ID: ", participantID)

    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y at %H:%M:%S")
    print("Data collected on", dt_string)
    print("\n")

    file_name = "participant_" + str(participantID) + ".txt"
    path_to_file = "/home/labuser/raf/participant_data/" + file_name
    data_file = open(path_to_file, "w")
    data_file.write("Participant ID: " + participantID + "\n")
    data_file.write("Data collected on " + dt_string + "\n")
    data_file.close()

    data_collection_start_time = rospy.get_time()

    for ii in range(num_sets):
        run.food_set = ii+1
        raf_msg = "Wait for researcher to arrange food items and begin selection."
        run.change_raf_message(raf_msg)
        run.publish()

        numFoodItems = None
        # numFoodItems = food_items_per_set
        while numFoodItems != food_items_per_set:
            input("Press ENTER when " + str(food_items_per_set) + " food items have been arranged on the plate.")

            detections = run.get_detections()
            item_ids = list(detections.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
            numFoodItems = len(idx)

        print("\n")
        print("########################################")
        print("########### Begin Food Set " + str(ii+1) + " ###########")
        print("########################################")

        food_set_start_time = rospy.get_time()

        for i in range(numFoodItems):
            # Selection
            if state == 1:
                if i > 0:
                    raf_msg = "Wait for researcher to arrange food items and begin selection."
                    run.change_raf_message(raf_msg)
                    run.publish()

                    intended_num_food_items = numFoodItems-i
                    current_numFoodItems = None
                    while current_numFoodItems != intended_num_food_items:
                        input("Press ENTER when " + str(intended_num_food_items) + " food items have been arranged on the plate.")

                        detections = run.get_detections()
                        item_ids = list(detections.class_ids)
                        idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
                        current_numFoodItems = len(idx)

                print("\n----- Food Set " + str(ii+1) + ", Food Item " + str(i+1) + " ----- \n")
                run.food_num = i+1
                food_item_start_time = rospy.get_time()
                overall_attempts_list.append(True)

                print("Selection:")

                # Randomly choose one of the detected items for the participant to select
                detections = run.get_detections()
                item_ids = list(detections.class_ids)
                idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
                numCurrentFoodItems = len(idx)
                item_to_select = random.randrange(0, numCurrentFoodItems)
                item_to_select_cls = detections.class_names[item_to_select]
                

                # Wait for participant to select item
                selected_item, selected_item_cls = run.item_selection(item_to_select, item_to_select_cls)

                state = 2

            rospy.sleep(0.1)

            # Acquisition
            if state == 2:

                print("Acquisition:")

                # Compute pose in robot frame of selected item
                det = run.get_detections()

                point_robot, gripper_angle = run.acquire_item_pose(selected_item, selected_item_cls, det)

                rospy.sleep(0.1)

                # Pick up the selected food item
                joint_angles = home_joint_angles
                # joint_angles['left_w2'] = math.radians(-1*(gripper_angle - 90))
                joint_angles['left_w2'] = math.radians(math.degrees(joint_angles['left_w2']) - (gripper_angle - 90))
                run.acquire_item(joint_angles, point_robot, selected_item_cls, table_height)

                acquisition_end_time = rospy.get_time()
                
                grasped, acquired_item = run.check_grasp()

            # Delivery
            if state == 3:

                print("Delivery:")

                run.deliver_item(mouth_joint_angles)

                delivery_end_time = rospy.get_time()

                grasped, delivered_item = run.check_grasp()

            # Trigger Food Item Transfer
            if state == 4:
                print("Trigger:")

                trigger = run.trigger_transfer()

                trigger_end_time = rospy.get_time()

                grasped, triggered_item = run.check_grasp()

            # Transfer
            if state == 5:
                print("Transfer:")

                run.transfer_item(pre_transfer_joint_angles, transfer_joint_angles)

                transfer_end_time = rospy.get_time()

                run.retract()

                grasped, transfered_item = run.check_grasp()

                state = 6
            
            # Return home
            if state == 6:

                home_joint_angles['left_w2'] = wrist_orientation
                run.move_to_angles(home_joint_angles)

                run.gripper_open()

                food_item_end_time = rospy.get_time()

                food_item_time = food_item_end_time - food_item_start_time

                state = 1

        # Check to see if all food items have been delivered
        detections = run.get_detections()
        item_ids = list(detections.class_ids)
        idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
        numFoodItems = len(idx)

        if numFoodItems == 0:
            raf_msg = "All food items delivered."
            run.change_raf_message(raf_msg)
            run.publish()
        else:
            raf_msg = "Not all food items delivered."
            run.change_raf_message(raf_msg)
            run.publish()
            print("Not all food items delivered.")

        food_set_end_time = rospy.get_time()

        food_set_time = food_set_end_time - food_set_start_time
        food_set_time_list.append(food_set_time)
        print("\nFood Set " + str(ii+1) + " Time Elapsed: " + str(food_set_time))
        print("########################################")
        print("\n")

        if (ii+1 % sets_till_break) == 0 and ii+1 != num_sets:
            break_num = int(ii+1 / sets_till_break)
            raf_msg = "Time for a Break. Notify researcher when ready to continue."
            run.change_raf_message(raf_msg)
            run.publish()

            break_start_time = rospy.get_time()
            input("Time for break " + str(break_num) + ". Press ENTER when ready to continue.\n")
            break_end_time = rospy.get_time()
            break_time = break_end_time - break_start_time
            break_time_list.append(break_time)
            print("Break " + str(break_num) + " Time Elapsed: " + str(break_time))

    data_collection_end_time = rospy.get_time()
    data_collection_time = data_collection_end_time - data_collection_start_time
    print("Data Collection Complete.")
    print("Data Collection Elapsed Time: ", data_collection_time)

    raf_msg = "Data Collection Finished. Thank you for your participation!"
    run.change_raf_message(raf_msg)
    run.publish()

    return 0

if __name__ == '__main__':

    # Initialize ROS Node
    rospy.init_node("RAF_dataCollection", anonymous=True)

    run = RAF_dataCollection('left')

    # Create e-stop exit signal and start e-stop thread
    signal.signal(signal.SIGUSR1, signal_handler)
    e = threading.Thread(target=thread_job, args=(run,))
    e.setDaemon(True) 
    e.start()

    try:
        sys.exit(main(run))
    except ExitCommand:
        pass
    finally:
        print('Program Terminated.')