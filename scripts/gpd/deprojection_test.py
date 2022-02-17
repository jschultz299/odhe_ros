#!/usr/bin/env python

# This program is the main file for data collection 
# Created by Jack Schultz
# 7/20/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import struct
import sys
import math
import cv2
import pyrealsense2
import pickle

import tf
import tf.transformations as transform

import image_geometry

import rospy
import numpy as np
import baxter_interface
from baxter_core_msgs.msg import EndpointState
from odhe_ros.msg import Result, DltParams
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
# from odhe_ros.msg import EyeTracker, Object
from geometry_msgs.msg import (
    PoseStamped,
    Pose
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

class RAF_dataCollection():
    def __init__(self, limb, verbose=True):
        # Subscribers
        self.sub_food = rospy.Subscriber('arm_camera_results', Result, self.food_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camInfo_callback)
        
        # Initialize parameters
        self.bridge = CvBridge()
        self._limb_name = limb # string
        self._verbose = False # bool
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
        # right = baxter_interface.Gripper('right', baxter_interface.CHECK_VERSION)
        # print("Enabling robot... ")
        # self._rs.enable()
        # print("Calibrating gripper...")
        # left.calibrate()

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header

    def get_depth_array(self):
        result = self.depth_array
        return result

    def food_callback(self, msg):
        self.food_items = msg

    def get_food_items(self):
        result = self.food_items
        return result

    def dlt_callback(self, msg):
        self.dlt = msg

    def camInfo_callback(self, msg):
        self.cam_info = msg

    def get_cam_info(self):
        result = self.cam_info
        return result

    def compute_position(self, x, y):
        # This function computes (X,Y) meters from (x,y) pixels
        X = ( (self.dlt.P2 + x*self.dlt.P8)*(self.dlt.P6 + y) - (self.dlt.P5 + y*self.dlt.P8)*(self.dlt.P3 + x) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )
        Y = ( (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P3 + x) - (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P6 + y) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )

        return X, Y

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

    def compute_grasp(self):
        # This function computes the centroid of the detected food item in pixels (u, v)
        #      You must only have one pretzel on the plate at a time

        # Check and make sure there is only one food item in the camera frame
        food_items = self.food_items
        if food_items is not None:
            # numItems = len(food_items.class_ids)
            # print("Number of objects detected: " + str(numItems))

            item_ids = list(food_items.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e == 3]         # returns index of pretzels
            # TODO: make above line include all food items, not just pretzels

            if len(idx) > 1:
                rospy.logwarn_once("Only have one food item in view of the camera!")
                centroid = None
            elif len(idx) < 1:
                rospy.logwarn_once("Only have one food item in view of the camera!")
                centroid = None
            else:

                # Compute food item centroid
                mask = self.bridge.imgmsg_to_cv2(food_items.masks[idx[0]])

                ret, thresh = cv2.threshold(mask, 0, 255, 0)

                cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

                rotrect = cv2.minAreaRect(cntrs[0])
                box = cv2.boxPoints(rotrect)
                box = np.int0(box)

                height = np.linalg.norm(box[0] - box[3])
                width = np.linalg.norm(box[0] - box[1])

                angle = rotrect[-1]

                if height > width:
                    # Midpoint of bottom center of food item
                    mid = (int((box[3][0] + box[2][0]) / 2), int((box[3][1] + box[2][1]) / 2))
                    # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
                    gripper_angle = abs(angle - 180)
                else:
                    mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))
                    gripper_angle = abs(angle - 90)

                # Create a deadzone that defaults to a grip angle of 0 degrees
                if (gripper_angle > 0 and gripper_angle <= 3):
                    gripper_angle = 0
                elif (gripper_angle > 177 and gripper_angle <= 180):
                    gripper_angle = 0

                # Centroid of food item
                centroid = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))
            return centroid, gripper_angle

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, cameraInfo):
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        # result[0]: right, result[1]: down, result[2]: forward
        return result[0], result[1], result[2]

    def move_to_angles(self, joint_angles=None):
        # print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not joint_angles:
            joint_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(joint_angles)
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
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format((seed_str)))
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

    def tf_to_matrix(self, t, q):
        """ROS transform to 4x4 matrix"""
        t_matrix = transform.translation_matrix(t)
        r_matrix = transform.quaternion_matrix(q)
        return np.dot(t_matrix, r_matrix)

    def acquire_item(self, joint_angles, point_robot, table_height):

        self.move_to_angles(joint_angles)

        acquire_x = point_robot[0][0]
        acquire_y = point_robot[1][0]
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

        return acquire_z
    
        
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
    rospy.init_node("temp")
    limb = 'left'
    rate = rospy.Rate(15)

    run = RAF_dataCollection(limb)

    # Read home position set in Calibration.py
    with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
        home_joint_angles = pickle.load(handle)
    handle.close()

    # Read table height set in Calibration.py
    with open('/home/labuser/raf/set_positions/table_height.pkl', 'rb') as handle:
        table_height = pickle.load(handle)
    handle.close()

    # Move to starting location
    run.move_to_angles(home_joint_angles)
    run.gripper_open()

    food_items = run.get_food_items()

    if food_items is not None:
        # numItems = len(food_items.class_ids)
        # print("Number of objects detected: " + str(numItems))

        item_ids = list(food_items.class_ids)
        idx = [i for i, e in enumerate(item_ids) if e == 3]         # returns index of pretzels
        # TODO: make above line include all food items, not just pretzels

        if len(idx) > 1:
            rospy.logwarn_once("Only have one food item in view of the camera!")
            centroid = None
        elif len(idx) < 1:
            rospy.logwarn_once("Only have one food item in view of the camera!")
            centroid = None
        else:

            # Compute food item centroid
            mask = bridge.imgmsg_to_cv2(food_items.masks[idx[0]])

            ret, thresh = cv2.threshold(mask, 0, 255, 0)

            cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            rotrect = cv2.minAreaRect(cntrs[0])
            box = cv2.boxPoints(rotrect)
            box = np.int0(box)

            height = np.linalg.norm(box[0] - box[3])
            width = np.linalg.norm(box[0] - box[1])

            angle = rotrect[-1]

            if height > width:
                # Midpoint of bottom center of food item
                mid = (int((box[3][0] + box[2][0]) / 2), int((box[3][1] + box[2][1]) / 2))
                # Angle measured from horizontal axis (x-axis), counter-clockwise to line connecting box vertices 0 and 1
                gripper_angle = abs(angle - 180)
            else:
                mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))
                gripper_angle = abs(angle - 90)

            # Create a deadzone that defaults to a grip angle of 0 degrees
            if (gripper_angle > 0 and gripper_angle <= 3):
                gripper_angle = 0
            elif (gripper_angle > 177 and gripper_angle <= 180):
                gripper_angle = 0

            # Centroid of food item
            centroid = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))

            depth_array = run.get_depth_array()
            depth = int(depth_array[int(centroid[1]), int(centroid[0])])   # depth_array index reversed for some reason
            depth = depth / 1000

            # Test value
            # depth = .595

            print("Centroid: ", centroid)
            print("Depth: ", depth)
            print("Gripper angle: ", gripper_angle)

            x = centroid[0]
            y = centroid[1]
            cameraInfo = run.get_cam_info()

            X, Y, Z = run.convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo)

            # Transform point in camera from to robot frame
            listener = tf.TransformListener()
            count = 0
            while count < 10:
                rate.sleep()
                count += 1
                try:
                    (trans_cam_world, rot_cam_world) = listener.lookupTransform('/world', '/_color_optical_frame', rospy.Time(0))
                    # (trans_hand_world, rot_hand_world) = listener.lookupTransform('/world', '/left_gripper', rospy.Time(0))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("Error in transform lookup.")
                    continue

            # T_cam_point = np.array([1, 0, 0, X],[0, 1, 0, Y],[0, 0, 1, Z],[0, 0, 0, 1])
            P_cam_point = np.array([[X],[Y],[Z],[1]])
            T_cam_world = run.tf_to_matrix(trans_cam_world, rot_cam_world)
            P_cam_world = np.matmul(T_cam_world, P_cam_point)

            print("P_cam_point: ", P_cam_point)
            print("T_cam_world: ", T_cam_world)
            print("P_cam_world: ", P_cam_world)

            print("Pixel - (" + str(x) + ", " + str(y) + ")")
            print("Point camera frame - [" + str(X) + ", " + str(Y) + ", " + str(Z) + "]")
            print("Point robot frame - [" + str(P_cam_world[0]) + ", " + str(P_cam_world[1]) + ", " + str(P_cam_world[2]) + "]")

            point_robot = P_cam_world

            # Test values
            table_height = -.123

            # Pick up the selected food item
            joint_angles = home_joint_angles
            # joint_angles['left_w2'] = math.radians(-1*(gripper_angle - 90))
            joint_angles['left_w2'] = math.radians(math.degrees(joint_angles['left_w2']) - (gripper_angle - 90))
            new_z = run.acquire_item(joint_angles, point_robot, table_height)

            print("Point robot frame (height adjusted) - [" + str(P_cam_world[0]) + ", " + str(P_cam_world[1]) + ", " + str(new_z) + "]")

    return 0

if __name__ == '__main__':
    sys.exit(main())
