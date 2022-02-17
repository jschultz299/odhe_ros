#!/usr/bin/env python

# This program is the main file for data collection 
# Created by Jack Schultz
# Contributed to by Jinlong Li
# 7/20/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import struct
import sys
import math
from xmlrpc.client import Boolean
# from Xlib.protocol.rq import Bool
import cv2
import random
from baxter_core_msgs import msg
from genpy import message
import rospy
import numpy as np
import baxter_interface
import pickle

import threading
import queue
import signal

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

class RAF_dataCollection():
    def __init__(self, limb, verbose=True):
        # Parameters
        self.bridge = CvBridge()
        self.timer_running = False
        self.timer = 0
        self.timer_start = 0
        self.RAF_msg = "Loading..."
        self.selected = False
        self.cursor_angle = math.pi * 0.5
        self.dwell_time = 2         # Time to fill cursor and select item
        self.delay_time = .5        # Time before cursor begins to fill
        self.estop_dwell_time = 1   # Time to fill cursor and select e-stop
        self.estop_delay_time = .5 # Time before cursor begins to fill on e-stop
        self.hl_id = None
        self.image_flag = 2     # 0 for raw image, 1 for item selection image, 2 for face detection
        self.mouth_open = False

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

        # Subscriber for study state
            # 0 for before start button pressed
            # 1 for after start button pressed, before food item selected
            # 2 for after food item selected, before food item picked up
            # 3 for after food item picked up, before facial keypoint detection is started
            # 4 for after facial keypoint detection is started, before food item transfer
            # 5 for during food item transfer, before food item in position in front of mouth
            # 6 for after food item in position in front of mouth, before food delivered
            # 7 for after food item has been delivered, before return to start position

        # Publishers
        self.pub_vis = rospy.Publisher('raf_visualization', Image, queue_size=10)
        # self.pub_state = rospy.Publisher('raf_state_to_GUI', RafStateToGui, queue_size=10)
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)
        self.pub_cursor = rospy.Publisher('raf_cursor_angle', String, queue_size=10)

    # Subscriber Callbacks
    def image_callback(self, msg):
        self.image_msg = msg
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

        try:
            self.publish()
        except:
            if self._verbose:
                print("Getting image...")

    def face_callback(self, msg):
        self.face_msg = msg

    def mouth_callback(self, msg):
        self.mouth_open = msg.data
    
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

    # Signal Handlers
    def handler1(self, sig_num, curr_stack_frame):
        print("Signal: {" + signal.strsignal(sig_num) +  "} Received.")

    # Main Functions
    def sort_detections(self, msg):
        # Sort detections by y-position of upper left bbox corner
        # TODO: Sort by y-position first and then sort again by x-position
        # This will prevent object ids from flipping back and forth if they are at the same y-position

        target = self.Out_transfer(msg.class_ids, msg.class_names, msg.scores, msg.boxes, msg.masks)

        # Sort by y-offset
        self.Sort_quick(target, 0, len(target)-1)

        # Sort by x-offset
        self.Sort_quick(target, 0, len(target)-1)

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

    def estop(self):
        signal.signal(signal.SIGILL, self.handler1)

        data = display.Display().screen().root.query_pointer()._data
        p = path.Path([(120,805), (234,805), (306, 878), (306,991), (234,1064), (121,1064), (48,991), (48,878)])      # Define Octogon
        contains = p.contains_points([(data["root_x"], data["root_y"])])

        # Check if cursor is inside the stop sign
        if contains:
            # Cursor enters box
            if not self.timer_running:
                self.timer_start = rospy.get_time()
                self.timer_running = True

            # Cursor remains in box
            if self.timer_running:
                self.timer = rospy.get_time() - self.timer_start

        else:
            # Cursor leaves box
            if self.timer_running:
                self.timer_running = False
                self.timer_start = 0
                self.timer = 0

        if self.timer > self.estop_dwell_time + self.estop_delay_time:
            # print("Object Selected!")
            self.timer_running = False
            self.timer_start = 0
            self.timer = 0
            raf_msg = "E-stop Pressed."
            self.change_raf_message(raf_msg)

            self.publish()

            print("E-stop Pressed. Exiting Program.")
            signal.raise_signal(signal.SIGINT)

        self.cursor_angle = self.linear_map(self.estop_delay_time, self.estop_dwell_time, 0, 2*math.pi, np.clip(self.timer, self.estop_delay_time, self.estop_dwell_time))

    def compute_position(self, x, y):
        # This function computes (X,Y) meters from (x,y) pixels
        X = ( (self.dlt.P2 + x*self.dlt.P8)*(self.dlt.P6 + y) - (self.dlt.P5 + y*self.dlt.P8)*(self.dlt.P3 + x) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )
        Y = ( (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P3 + x) - (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P6 + y) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )

        return X, Y

    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y

    def item_selection(self, item):
        # TODO: Add class name in RAF_msg
        raf_msg = "Select Food item " + str(item + 1)
        self.change_raf_message(raf_msg)
        self.selected = False
        selected_item = None

        self.image_flag = 1

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

                        # Cursor remains in box
                        if self.timer_running and self.hl_id == i:
                            self.timer = rospy.get_time() - self.timer_start

                    else:
                        color =  [0,0,0]
                        thickness = 1

                        # Cursor leaves box
                        if self.timer_running and self.hl_id == i:
                            self.timer_running = False
                            self.timer_start = 0
                            self.timer = 0

                    if self.timer > self.dwell_time + self.delay_time:
                        # print("Object Selected!")
                        self.timer_running = False
                        self.timer_start = 0
                        self.timer = 0
                        self.selected = True
                        selected_item = i
                        raf_msg = "Food item " + str(i+1) + " selected"
                        self.change_raf_message(raf_msg)
                        color = [0, 255, 0]
                        thickness = 2

                    # Draw Bounding boxes
                    start_point = (ul_x, ul_y)
                    end_point = (br_x, br_y)
                    output = cv2.rectangle(output, start_point, end_point, color, thickness)
                    output = cv2.putText(output, str(i+1), (ul_x + 2, br_y - 3), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,255,255), 1, cv2.LINE_AA)

                self.cursor_angle = self.linear_map(self.delay_time, self.dwell_time, 0, 2*math.pi, np.clip(self.timer, self.delay_time, self.dwell_time))

                im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                self.im_msg = self.bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        return selected_item

    def acquire_item_pose(self, item_idx, item, det):
        # Get Transformation between camera and gripper (from calibrate_camera.py)
        with open('/home/labuser/ros_ws/src/odhe_ros/scripts/Transformation.npy', 'rb') as f:
            T = np.load(f)
        f.close()

        mask = self.bridge.imgmsg_to_cv2(det.masks[item_idx])

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

        # x = centroid[0]
        # y = centroid[1]

        x = grasp_point[0]
        y = grasp_point[1]

        X, Y = self.compute_position(x, y)
        depth = int(self.depth_array[int(centroid[1]), int(centroid[0])])   # depth_array index reversed for some reason
        depth = depth / 1000
        point_camera = np.array([[X], [Y], [depth], [1]])

        point_robot = np.matmul(T, point_camera)

        print("Robot Position: (X: " + str(point_robot[0][0]) + ", Y: " + str(point_robot[1][0]) + ", Z: " + str(point_robot[2][0]) + ")")
        print("Gripper Angle: " + str(gripper_angle))

        raf_msg = "Picking up Food Item " + str(item + 1)
        self.change_raf_message(raf_msg)

        return point_robot, gripper_angle

    def acquire_item(self, joint_angles, point_robot):

        self.image_flag = 2

        self.move_to_angles(joint_angles)

        current_pose = self.get_current_pose()

        pose = Pose()
        pose.position.x = point_robot[0][0]
        pose.position.y = point_robot[1][0]
        pose.position.z = point_robot[2][0] + .1
        pose.orientation = current_pose['orientation']

        self._servo_to_pose(pose)

        pose.position.z = pose.position.z  - .08

        self._servo_to_pose(pose)

        pose.position.z = pose.position.z  - .025

        self._servo_to_pose(pose)

        self.gripper_close()

        current_pose = self.get_current_pose()

        pose = Pose()
        pose.position.x = point_robot[0][0]
        pose.position.y = point_robot[1][0]
        pose.position.z = point_robot[2][0] + .4
        pose.orientation = current_pose['orientation']

        self._servo_to_pose(pose)

    def deliver_item(self, joint_angles):
        # Robot moves to face the user
        # TODO: Add in a way to set this position on startup. It should save it to a file and load it in. Different for each participant

        self.move_to_angles(joint_angles)

    def item_transfer(self, pre_transfer_angles, transfer_angles):
        # Robot waits for the user to open their mouth, then it releases the food item
        raf_msg = "Loading..."
        self.change_raf_message(raf_msg)

        rospy.sleep(1.0)

        raf_msg = "Open your mouth when ready"
        self.change_raf_message(raf_msg)

        self.mouth_open = False

        while not self.mouth_open:
            self.spinOnce()

        raf_msg = "Transferring Food Item"
        self.change_raf_message(raf_msg)

        self.move_to_angles(pre_transfer_angles)

        self.move_to_angles(transfer_angles)

        raf_msg = "Releasing Food Item"
        self.change_raf_message(raf_msg)

        rospy.sleep(2.0)

        self.gripper_open()

        current_pose = self.get_current_pose()

        pose = Pose()
        pose.position.x = current_pose['position'][0] - .1
        pose.position.y = current_pose['position'][1] - .1
        pose.position.z = current_pose['position'][2]
        pose.orientation = current_pose['orientation']

        self._servo_to_pose(pose)

        raf_msg = "Food Item Delivered"
        self.change_raf_message(raf_msg)

    def change_raf_message(self, msg):
        self.RAF_msg = msg
        print("Message: " + self.RAF_msg)

    def spinOnce(self):
        r = rospy.Rate(10)
        r.sleep()

    def publish(self):
        if self.image_flag == 0:
            self.pub_vis.publish(self.image_msg)
        elif self.image_flag == 1:
            self.pub_vis.publish(self.im_msg)
        else:
            self.pub_vis.publish(self.face_msg)

        self.pub_msg.publish(self.RAF_msg)
        self.pub_cursor.publish(str(self.cursor_angle))
        self.loop_rate.sleep()

    # Robot Functions
    def move_to_angles(self, start_angles=None):
        # print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        rospy.sleep(1.0)

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
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def get_current_pose(self):
        # Get the current pose of the robot
        current_pose = self._limb.endpoint_pose()
        return current_pose

    def get_current_joint_angles(self):
        current_joint_angles = self._limb.joint_angles()
        return current_joint_angles

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)
    
        
def main():
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

    # TODO List:
    #       1) Change code to use object form returned from Sort_Quick()
    #       2) Deep Sort to track specific food items
    #       3) Sort by y-position and then again by x-position
    #       4) Add food item class type to RAF_message
    #       5) Add cursor stability into dwell time trigger
    #       6) Set robot deliver position for each participant when code starts
    #       7) Instead of food item centroid, pick up toward one end

    # RAF Parameters
    state = 1               # determines experimental logic

    # Set debug to True to move robot to starting joint angles and then just spin
    debug = True

    # Initialize ROS Node
    rospy.init_node("RAF_dataCollection", anonymous=True)

    bridge = CvBridge()
    limb = 'left'

    # Read home position set in Calibration.py
    with open('home_position.pkl', 'rb') as handle:
        home_joint_angles = pickle.load(handle)
    handle.close()

    wrist_orientation = home_joint_angles['left_w2']

    # Initialize Class
    run = RAF_dataCollection(limb)

    # Move to home position
    run.move_to_angles(home_joint_angles)

    if debug:

        while not rospy.is_shutdown():
            rospy.loginfo_once("Running in debug mode. Press ctrl+C to quit.")

            run.estop()
            run.publish()

    else:
        # 1) Calibrate.py should have already been run
        # 2) Initialize facial keypoint detection. Set the mouth detection joint angles.

        with open('mouth_position.pkl', 'rb') as handle:
            mouth_joint_angles = pickle.load(handle)
        handle.close()

        # Comment out to skip saving the mouth position
        run.move_to_angles(mouth_joint_angles)

        input("\nPress ENTER to save Mouth Detection Position.")
        mouth_joint_angles = run.get_current_joint_angles()

        f = open("mouth_position.pkl","wb")
        pickle.dump(mouth_joint_angles,f)
        f.close()

        # 3) Set food item pre-transfer position

        with open('pre_transfer_position.pkl', 'rb') as handle:
            pre_transfer_joint_angles = pickle.load(handle)
        handle.close()

        # # Comment out to skip saving the transfer position
        run.move_to_angles(pre_transfer_joint_angles)

        input("\nPress ENTER to save Food Pre-Transfer Position.")
        pre_transfer_joint_angles = run.get_current_joint_angles()

        f = open("pre_transfer_position.pkl","wb")
        pickle.dump(pre_transfer_joint_angles,f)
        f.close()

        # 4) Set food item transfer position

        with open('transfer_position.pkl', 'rb') as handle:
            transfer_joint_angles = pickle.load(handle)
        handle.close()

        # Comment out to skip saving the transfer position
        run.move_to_angles(transfer_joint_angles)

        input("\nPress ENTER to save Food Transfer Position.")
        transfer_joint_angles = run.get_current_joint_angles()

        f = open("transfer_position.pkl","wb")
        pickle.dump(transfer_joint_angles,f)
        f.close()

        # Move to starting location
        run.move_to_angles(home_joint_angles)
        run.gripper_open()

        detections = run.get_detections()
        item_ids = list(detections.class_ids)
        idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
        numFoodItems = len(idx)

        for i in range(numFoodItems):
            if state == 1:
                print("\n### State 1: Food Item Selection ###")
                # Randomly choose one of the detected items for the participant to select
                detections = run.get_detections()
                if detections is not None:
                    item_ids = list(detections.class_ids)
                    idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]
                    item_to_select = random.randrange(0, len(idx))

                if run._verbose:
                        print("item_ids:", item_ids)
                        print("idx: ", idx)
                        print("item_to_select: ", item_to_select)
                        print("Waiting for selection...")

                # Wait for participant to select item
                selected_item = run.item_selection(item_to_select)

                if run._verbose:
                    print("Selection Made.")
                    print("selected_item_idx: ", selected_item)
                    print("selected item index of sorted list: ", idx[selected_item])

                state = 2

            rospy.sleep(1.0)

            if state == 2:
                print("\n### State 2: Food Item Acquisition ###")

                # Compute pose in robot frame of selected item
                # TODO: change position to be near the end of the food item instead of the centroid

                det = run.get_detections()

                if det is not None:
                    item_ids = list(det.class_ids)
                    # idx = [i for i, e in enumerate(item_ids) if e == 3 ]            # Only for pretzels
                    idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 4 ]   # For all food items

                point_robot, gripper_angle = run.acquire_item_pose(selected_item, selected_item, det)

                rospy.sleep(1.0)

                # Pick up the selected food item
                joint_angles = home_joint_angles
                joint_angles['left_w2'] = math.radians(-1*(gripper_angle - 90))
                run.acquire_item(joint_angles, point_robot)

                state = 3

            if state == 3:
                print("\n### State 3: Food Item Delivery ###")

                run.deliver_item(mouth_joint_angles)

                run.item_transfer(pre_transfer_joint_angles, transfer_joint_angles)

                home_joint_angles['left_w2'] = wrist_orientation
                run.move_to_angles(home_joint_angles)

                state = 1

            raf_msg = "All food items delivered."
            run.change_raf_message(raf_msg)
            run.publish()

    return 0

if __name__ == '__main__':
    sys.exit(main())
