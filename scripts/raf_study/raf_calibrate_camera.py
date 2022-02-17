#!/usr/bin/env python

# This program creates a transformation matrix between the camera and the robot's gripper
# Created by Jack Schultz
# 7/26/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import sys
from numpy.lib.scimath import sqrt
import rospy
import numpy as np
import baxter_interface
import cv2
import math
import pickle
import tf.transformations
import time

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

from odhe_ros.msg import Result, DltParams
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tempfile import TemporaryFile
from statistics import mean

class RAF_calibrate_camera():
    def __init__(self, limb, verbose=True):
        # Subscribers
        # rospy.Subscriber("/camera2/depth/image_rect_raw", Image, self.callback)
        rospy.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.sub_food = rospy.Subscriber('arm_camera_results', Result, self.food_callback)
        self.sub_dlt = rospy.Subscriber('dlt_params', DltParams, self.dlt_callback)

        # Publishers
        # self.pub_tf = rospy.Publisher('dlt_tf', Float64MultiArray, queue_size=10)

        # Initialize parameters
        self.bridge = CvBridge()
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

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header

    def get_depth_array(self):
        result = self.depth_array
        return result

    def food_callback(self, msg):
        self.food_items = msg

    def get_food(self):
        try:
            result = self.food_items
        except:
            result = None
        return result

    def dlt_callback(self, msg):
        # Callback for receiving dlt parameters from dlt.py
        self.dlt = msg

    def compute_position(self, x, y):
        # Divide pixels by camera resolution for better numerical stability
        x = x / 640
        y = y / 480

        # This function computes (X,Y) meters from (x,y) pixels
        X = ( (self.dlt.P2 + x*self.dlt.P8)*(self.dlt.P6 + y) - (self.dlt.P5 + y*self.dlt.P8)*(self.dlt.P3 + x) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )
        Y = ( (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P3 + x) - (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P6 + y) ) / ( (self.dlt.P1 + x*self.dlt.P7)*(self.dlt.P5 + y*self.dlt.P8) - (self.dlt.P4 + y*self.dlt.P7)*(self.dlt.P2 + x*self.dlt.P8) )

        return X, Y

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        rospy.sleep(1.0)
        # print("Running. Ctrl-c to quit")

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

    def compute_centroid(self):
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
                print("Only have one food item in view of the camera!")
                return

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

        if height > width:
            # Midpoint of bottom center of food item
            mid = (int((box[3][0] + box[2][0]) / 2), int((box[3][1] + box[2][1]) / 2))
        else:
            mid = (int((box[3][0] + box[0][0]) / 2), int((box[3][1] + box[0][1]) / 2))

        # Centroid of food item
        centroid = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))
        return centroid, mask

    def soder(self, P1, P2):
        # Inputs:
            # P1 (Nx3) coordinates of food item centroids in camera frame
            # P2 (Nx3) coordinates of food item centroids in robot frame
        # Output:
            # T (4x4) transformation matrix that transforms P1 to P2 coordinates (P2 = T*P1)

        [nmarkers, ndim1] = P1.shape
        [nmarkers2, ndim2] =  P2.shape

        m1 = np.mean(P1, axis=0)        # (1x3)
        m2 = np.mean(P2, axis=0)

        A = np.zeros((nmarkers,3))      # (nx3)
        B = np.zeros((nmarkers,3))
        for i in range(nmarkers):
            A[i,:] = P1[i,:] - m1
            B[i,:] = P2[i,:] - m2

        A = np.transpose(A)
        B = np.transpose(B)

        C = np.matmul(B, np.transpose(A))

        [P,T,Q] = np.linalg.svd(C)

        temp = np.linalg.det(np.matmul(P, Q))
        R = P @ np.diag([1, 1, temp]) @ Q

        d = np.transpose(m2) - R @ np.transpose(m1)
        d = np.array([[d[0]],[d[1]],[d[2]]])

        T = np.concatenate((R, d), axis=1)
        temp = np.array([0,0,0,1])
        T = np.vstack((T, temp))

        sumsq = 0
        for i in range(nmarkers):
            P2model = (R @ np.transpose(P1[i][:])) + d.T
            sumsq = sumsq + math.pow(np.linalg.norm(P2model - np.transpose(P2[i][:])), 2)

        rms = sqrt(sumsq/3/nmarkers)

        return T, rms

    def soder_alt(self, P1, P2):
        # This might work the same as soder using built in tf function
        P1 = np.transpose(P1)
        P2 = np.transpose(P2)
        T = tf.transformations.superimposition_matrix(P1, P2)

        return T

    def get_current_joint_angles(self):
        current_joint_angles = self._limb.joint_angles()
        return current_joint_angles


def main():
    """ODHE Project First Experiment: Feeding food items to SCI Individuals

    This is experiment took place Summer 2021

    This code creates a transformation matrix between the arm-mounted depth camera
    and the Baxter robot's gripper. This should be performed if the camera position
    has been changed.

    The calibration works by placing food items on the plate and  obtaining the 3D 
    position of the centroid of the food item in the camera's frame, and associating
    it with the 3D position of the centroid of the food item in Baxter's global frame.
    Then, a transformation matrix can be obtained.

    For each food item:
        1) Obtain 3D position of centroid of food item in camera's frame
        2) Obtain 3D position of centroid of food item in Baxter's frame
    
    3) Compute transformation matrix

    """
    rospy.init_node("calibrate_camera")
    limb = 'left'

    # Read home position set in Calibration.py
    with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
        joint_angles = pickle.load(handle)
    handle.close

    run = RAF_calibrate_camera(limb)
    
    # Move to starting location
    run.move_to_start(joint_angles)

    # Here you can update the home position if you want to.
    input("Press ENTER to save Home Position.")

    current_joint_angles = run.get_current_joint_angles()
    f = open("/home/labuser/raf/set_positions/home_position.pkl","wb")
    pickle.dump(current_joint_angles,f)
    f.close()

    joint_angles = current_joint_angles

    run.move_to_start(joint_angles)

    P1 = list()
    P2 = list()
    Z = list()

    calibration_points = input("Enter number of calibration points.\n")

    print("\n######### Starting Calibration Sequence #########")
    print("Points to calibrate: ", str(calibration_points))

    for i in range(int(calibration_points)):
        # Save centroid of food item in camera frame
        input("\nPress Enter to save Point " + str(i+1) + " in camera frame...")
        # Compute centroid in pixels
        # centroid = run.compute_centroid()
        
        ### BEGIN ADDED CODE ###
        centroid, mask = run.compute_centroid()

        coord = cv2.findNonZero(mask)

        # Convert centroid to meters
        X, Y = run.compute_position(centroid[0],centroid[1])
        depth_array = run.get_depth_array()
        depth = int(depth_array[int(centroid[1]), int(centroid[0])])   # depth_array index reversed for some reason

        depth_list = []
        for ii in range(len(coord)):
            if int(depth_array[int(coord[ii][0][1]), int(coord[ii][0][0])]) == 0:
                print("Zero depth for coordinate: ", [int(coord[ii][0][1]), int(coord[ii][0][0])])
            else:
                depth_list.append(int(depth_array[int(coord[ii][0][1]), int(coord[ii][0][0])]))

        depth_avg = np.mean(depth_list)

        depth = depth / 1000
        depth_avg = depth_avg / 1000
        print("Centroid Depth: ", depth)
        print("Avg Mask Depth: ", depth_avg)
        point_camera = [X, Y, depth_avg]
        print("Point " + str(i+1) + " in camera frame: " + str(point_camera))

        ##### ---> Physically move robot gripper to food item now <--- #####

        # Save centroid of food item in robot frame
        input("Press Enter to save Point " + str(i+1) + " in robot frame...")
        current_pose = run.get_current_pose()
        point_robot = [current_pose['position'].x, current_pose['position'].y, current_pose['position'].z]
        print("Point " + str(i+1) + " in robot frame: " + str(point_robot))

        # Save points in both coordinate frames in a list
        P1.append(point_camera)
        P2.append(point_robot)
        Z.append(current_pose['position'].z)

        # Move to starting location
        run.move_to_start(joint_angles)

    print("\nTotal of " + str(i+1) + " points collected.")

    #  Compute transformation matrix between camera frame and robot frame (P2 = T*P1)
    P1 = np.array(P1)
    P2 = np.array(P2)
    T, rms = run.soder(np.asarray(P1), np.asarray(P2))

    # This built in tf method is slightly faster
    # T = run.soder_alt(np.asarray(P1), np.asarray(P2))

    # Test T
    # T = np.array([[ -0.6592,  0.7224,  -0.2086,  0.9492],
    #                 [ -0.7023, -0.6907,  -0.1725,  0.7792],
    #                 [ -0.2687,  0.0328,   0.9627, -0.9762],
    #                 [       0,       0,        0,       1]], dtype=float)

    print("Calibration Complete!")

    print("\nTransformation Matrix: ")
    print(T)

    print("\nRMS Error: ")
    print(rms)

    # Write to a text file
    input("\nPress ENTER to save Transformation Matrix...")

    with open('/home/labuser/ros_ws/src/odhe_ros/scripts/Transformation.npy', 'wb') as f:
        np.save(f, T)
    f.close()
    print("Transformation saved.")

    max_z = max(Z)      # Highest table coordinate. This is the value we should not go lower than
    avg_z = mean(Z)
    min_z = min(Z)
    print("\nZ Coordinate array: ", Z)
    print("Maximum Z coordinate: ", max_z)
    print("Average Z coordinate: ", avg_z)
    print("Minimum Z coordinate: ", min_z)
    
    # Write to a text file
    input("\nPress ENTER to save table height...")

    with open("/home/labuser/raf/set_positions/table_height.pkl","wb") as file:
        pickle.dump(min_z, file)    # Although intuitively should use the max, the min tends to yield the best results
    file.close()
    print("Table height saved.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

