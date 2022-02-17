#!/usr/bin/env python

# This program is the main file for data collection 
# Written by Jack Schultz
# Created 1/20/22

"""
Test for integrating object detection and GPD
"""

import struct
import sys
import math
from xmlrpc.client import Boolean
import cv2
import random
import roslaunch

from cv2 import data
from baxter_core_msgs import msg
from genpy import message
import rospy
import numpy as np
import baxter_interface
import pickle

from odhe_ros.msg import Result, DltParams
from cv_bridge import CvBridge
from Xlib import X, display
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

class gpd_launcher():
    def __init__(self, limb, verbose=True):
        # Parameters
        self.detections = None
        self.image = None
        self.bridge = CvBridge()
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
        self._rs.enable()
        left.calibrate()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Subscribers
        self.sub_detections = rospy.Subscriber('arm_camera_results', Result, self.detections_callback)
        self.sub_depth = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.sub_image = rospy.Subscriber('arm_camera_objects', Image, self.image_callback)

        # Publishers

    # Subscriber Callbacks
    def image_callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header
    
    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header
    
    def detections_callback(self, msg):
        # Receive detections and sort them according to y-offset
        temp = self.sort_detections(msg)
        
        # Reorganize back into Result() object type
        # TODO: Change the rest of the code to use the above organization (by object). It works well for now, it just might speed it up.
        self.detections = Result()
        for i in range(len(temp)):
            self.detections.class_ids.append(temp[i][0])
            self.detections.class_names.append(temp[i][1])
            self.detections.scores.append(temp[i][2])
            self.detections.boxes.append(temp[i][3])
            self.detections.masks.append(temp[i][4])

        # TODO: Instead of sorting, we will use a deep sort machine learning method to track specific food items
        # Sort detections based on box position in image

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

    # Main Functions
    def compute_grasps(self):
        node = roslaunch.core.Node("gpd_ros", "detect_grasps", name="detect_grasps", output="screen")
        rospy.set_param("cloud_type", "0")
        rospy.set_param("cloud_topic", "/camera/depth/color/points")
        rospy.set_param("samples_topic", "")
        rospy.set_param("config_file", "/home/labuser/gpd/cfg/ros_eigen_params.cfg")
        rospy.set_param("rviz_topic", "plot_grasps")

        # params = roslaunch.core.Param("cloud_type", "0")
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        script = launch.launch(node)
        print(script.is_alive())

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

    def get_item_bbox(self):
        print(type(self.detections))
        print(type(self.image))
        if self.detections is not None and self.image is not None:
            item_ids = list(self.detections.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]         # returns index of food items

            for i in range(len(idx)):
                ul_x = self.detections.boxes[idx[i]].x_offset
                ul_y = self.detections.boxes[idx[i]].y_offset
                br_x = ul_x + self.detections.boxes[idx[i]].width
                br_y = ul_y + self.detections.boxes[idx[i]].height

        return ul_x, ul_y, br_x, br_y

    def get_centroid_depth(self, ul_x, ul_y, br_x, br_y):
        centroid = (int((ul_x + br_x) / 2), int((ul_y + br_y) / 2))
        return centroid

    def convert_to_meters(self, coord, depth):
        pass

    
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

        if height > width:
            # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
            gripper_angle = abs(angle - 180)
            mid = (int((box[3][0] + box[2][0]) / 2), int((box[3][1] + box[2][1]) / 2))
        else:
            gripper_angle = abs(angle - 90)
            mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))

        # Create a deadzone that defaults to a grip angle of 0 degrees
        if (gripper_angle > 0 and gripper_angle <= 3):
            gripper_angle = 0
        elif (gripper_angle > 177 and gripper_angle <= 180):
            gripper_angle = 0

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
                    interest_point_x = int((box[3][0] + box[2][0]) / 2)
                    interest_point_y = int((box[3][1] + box[2][1]) / 2)
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

    # Robot Functions
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
           
def main():
    """
    Test for integrating object detection a GPD
    """

    rospy.init_node("gpd_launcher")
    limb = 'left'

    run = gpd_launcher(limb)

    print("Initializing...")
    rospy.sleep(1)

    # run.compute_grasps()

    ul_x, ul_y, br_x, br_y = run.get_item_bbox()
    ul = [ul_x, ul_y]
    br = [br_x, br_y]

    print("Upper Left Corner (pixels): ", str(ul))
    print("Bottom Right Corner (pixels): ", str(br))

    cen_depth = run.get_centroid_depth(ul_x, ul_y, br_x, br_y)

    ul_x_m = run.convert_to_meters(ul_x, cen_depth)
    ul_y_m = run.convert_to_meters(ul_y, cen_depth)
    br_x_m = run.convert_to_meters(br_x, cen_depth)
    br_y_m = run.convert_to_meters(br_y, cen_depth)

    ul_m = [ul_x_m, ul_y_m]
    br_m = [br_x_m, br_y_m]

    print("Upper Left Corner (meters): ", str(ul_m))
    print("Bottom Right Corner (meters): ", str(br_m))

    while not rospy.is_shutdown():
        rospy.spin()

    return 0

if __name__ == '__main__':
    sys.exit(main())