#!/usr/bin/env python

# This program is the main file for data collection 
# Created by Jack Schultz
# 7/20/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import argparse
import struct
import sys
import copy
import math
import cv2

import rospy
import numpy as np
import baxter_interface
from baxter_core_msgs.msg import EndpointState
from odhe_ros.msg import Result
from cv_bridge import CvBridge
# from odhe_ros.msg import EyeTracker, Object
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header, String

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

class RAF_dataCollection():
    def __init__(self, limb, verbose=True):
        self.sub_food = rospy.Subscriber('arm_camera_results', Result, self.food_callback)
        
        # self.selection = rospy.Subscriber('selected_object', Object, self.selection_callback)
        # rospy.Subscriber('speech_command', String, self.speech_callback)
        self._limb_name = limb # string
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        left = baxter_interface.Gripper('left', baxter_interface.CHECK_VERSION)
        right = baxter_interface.Gripper('right', baxter_interface.CHECK_VERSION)
        print("Enabling robot... ")
        self._rs.enable()
        print("Calibrating gripper...")
        left.calibrate()

    def food_callback(self, msg):
        # Callback for receiving food item classes w/ masks from maskRCNN
        # self.food_header = msg.header
        # self.food_boxes = msg.boxes
        # self.food_class_ids = msg.class_ids
        # self.food_class_names = msg.class_names
        # self.food_scores = msg.scores
        # self.food_masks = msg.masks
        self.food_items = msg
        # print(self.food_items.class_ids)

    def get_food(self):
        # result = Result()
        try:
            result = self.food_items
        except:
            result = None
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

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        # self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

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

    # def _approach(self, pose):
    #     approach = copy.deepcopy(pose)
    #     # approach slightly above and behind the cup
    #     approach.position.x = approach.position.x - self._behind_distance
    #     approach.position.z = approach.position.z + self._above_distance
    #     joint_angles = self.ik_request(approach)
    #     self._guarded_move_to_joint_position(joint_angles)

    #     # Lower to just behind the cup
    #     print("Lowering...\n")
    #     approach.position.z = approach.position.z - self._above_distance
    #     joint_angles = self.ik_request(approach)
    #     self._guarded_move_to_joint_position(joint_angles)

    # def _hover(self, pose):
    #     hover = copy.deepcopy(pose)
    #     # hover the arm just over the place point
    #     hover.position.z = hover.position.z + self._above_distance
    #     joint_angles = self.ik_request(hover)
    #     self._guarded_move_to_joint_position(joint_angles)

    # def _retract(self):
    #     # retrieve current pose from endpoint
    #     current_pose = self._limb.endpoint_pose()
    #     ik_pose = Pose()
        # ik_pose.position.x = current_pose['position'].x 
        # ik_pose.position.y = current_pose['position'].y
        # ik_pose.position.z = current_pose['position'].z + self._above_distance
        # ik_pose.orientation.x = current_pose['orientation'].x 
        # ik_pose.orientation.y = current_pose['orientation'].y 
        # ik_pose.orientation.z = current_pose['orientation'].z 
        # ik_pose.orientation.w = current_pose['orientation'].w
    #     joint_angles = self.ik_request(ik_pose)
    #     # servo up from current pose
    #     self._guarded_move_to_joint_position(joint_angles)

    def get_current_pose(self):
        # Get the current pose of the robot
        current_pose = self._limb.endpoint_pose()
        return current_pose

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    # def pick_up_cup(self, poseCup, poseApproach, poseLower, poseHover):
    #     # open the gripper
    #     print("Opening the gripper...\n")
    #     self.gripper_open()
    #     # servo behind pose
    #     print("Approaching the cup...\n")
    #     self._servo_to_pose(poseApproach)
    #     self._servo_to_pose(poseLower)
    #     # servo to pose
    #     print("Moving to the cup...\n")
    #     self._servo_to_pose(poseCup)
    #     # close gripper
    #     print("Closing the gripper...\n")
    #     self.gripper_close()
    #     # retract to clear object
    #     print("Retracting the gripper...\n")
    #     self._servo_to_pose(poseHover)

    # def move_to_user(self, pose):
    #     # servo to pose
    #     print("Moving to the user...\n")
    #     self._servo_to_pose(pose)

    # def place_cup(self, poseCup, poseHover):
    #     #servo above pose
    #     print("Approaching the place point...\n")
    #     self._servo_to_pose(poseHover)
    #     # servo to pose
    #     print("Lowering the cup...\n")
    #     self._servo_to_pose(poseCup)
    #     # open gripper
    #     print("Opening the gripper...\n")
    #     self.gripper_open()
    #     # retract to clear object
    #     print("Retracting the gripper...\n")
    #     self._servo_to_pose(poseHover)

    def spinOnce(self):
        self.rate = 15
        r = rospy.Rate(self.rate)
        r.sleep()
    
        
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

    The following detail the specific steps of this data collection procedure:
    1) Pre-Selection / Idle
    2) Food Item Selection
    3) Food Item Acquisition
    4) Food Item Delivery
    5) Food Item Transfer
    6) Return to Home
    """

    bridge = CvBridge()
    rospy.init_node("RAF_dataCollection")
    limb = 'left'
    rate = rospy.Rate(100)
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_e0': 0,
                             'left_e1': 0.7735098122912198,
                             'left_s0': 0,
                             'left_s1': -0.966791391564782,
                             'left_w0': 0,
                             'left_w1': math.pi / 2,
                             'left_w2': 0}

    run = RAF_dataCollection(limb)

    # Move to starting location
    run.move_to_start(starting_joint_angles)
    run.gripper_open()

    # while not rospy.is_shutdown():

    # Print food item information
    food_items = run.get_food()
    if food_items is not None:
        numItems = len(food_items.class_ids)
        print("Number of objects detected: " + str(numItems))

        item_ids = list(food_items.class_ids)
        print(item_ids)
        food_ids = [x for x in item_ids if x > 1 and x < 4]
        print(food_ids)


    # Loop through all the food items
        
    mask = bridge.imgmsg_to_cv2(food_items.masks[0])

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

    # print("Gripper Angle: ", gripper_angle, "deg")

    centroid = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))
    cv2.circle(img, centroid, 2, (0,255,0), 2)
    cv2.arrowedLine(img, mid, centroid, (255,0,0), 2)
    org = (centroid[0] + 20, centroid[1] - 20)
    cv2.putText(img, "Theta: " + "{:.2f}".format(gripper_angle) + " deg", org, cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)

    # Visualize Food Item and Gripper Orientation
    # cv2.imshow("THRESH", thresh)
    # cv2.imshow("image", img)
    # cv2.waitKey(33)

    starting_joint_angles['left_w2'] = math.radians(-1*(gripper_angle - 90))
    run.move_to_start(starting_joint_angles)

    pose = run.get_current_pose()
    print(pose['orientation'])

    test_pose = Pose()
    test_pose.position.x = 0.5657568329227736
    test_pose.position.y = 0.751227429855027
    test_pose.position.z = -0.24783635116629155 + .10
    test_pose.orientation = pose['orientation']

    run._servo_to_pose(test_pose)

    test_pose.position.z = test_pose.position.z  - .05

    run._servo_to_pose(test_pose)

    test_pose.position.z = test_pose.position.z  - .05

    run._servo_to_pose(test_pose)

    run.gripper_close()

    starting_joint_angles['left_w2'] = 0
    run.move_to_start(starting_joint_angles)
    pose = run.get_current_pose()

    test_pose = Pose()
    test_pose.position.x = 0.5657568329227736
    test_pose.position.y = 0.751227429855027
    test_pose.position.z = -0.24783635116629155 + .10
    test_pose.orientation = pose['orientation']

    run._servo_to_pose(test_pose)

    test_pose.position.z = test_pose.position.z  - .05

    run._servo_to_pose(test_pose)

    test_pose.position.z = test_pose.position.z  - .05

    run._servo_to_pose(test_pose)

    run.gripper_open()

    run.move_to_start(starting_joint_angles)

    return 0

if __name__ == '__main__':
    sys.exit(main())
