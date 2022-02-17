#!/usr/bin/env python

# This program creates a transformation matrix between the camera and the robot's gripper
# Created by Jack Schultz
# 7/26/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import sys
import rospy
import baxter_interface
import pickle
import tf
import tf.transformations as transform
import numpy as np

class RAF_calibrate_table():
    def __init__(self, limb, verbose=True):
        # Subscribers

        # Publishers

        # Initialize parameters
        self._limb_name = limb # string
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        left = baxter_interface.Gripper('left', baxter_interface.CHECK_VERSION)
        print("Enabling robot... ")
        self._rs.enable()
        print("Calibrating gripper...")
        left.calibrate()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

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

    def get_current_pose(self):
        # Get the current pose of the robot
        current_pose = self._limb.endpoint_pose()
        return current_pose

    def get_current_joint_angles(self):
        current_joint_angles = self._limb.joint_angles()
        return current_joint_angles

    def soder_alt(self, P1, P2):
        P1 = np.transpose(P1)
        P2 = np.transpose(P2)
        T = transform.superimposition_matrix(P1, P2)
        return T

    def tf_to_matrix(self, t, q):
        """ROS transform to 4x4 matrix"""
        t_matrix = transform.translation_matrix(t)
        r_matrix = transform.quaternion_matrix(q)
        return np.dot(t_matrix, r_matrix)

def main():
    """
    Record AR tag positions in robot frame

    """
    rospy.init_node("test_camera_register")
    limb = 'left'

    # Read home position set in Calibration.py
    with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
        joint_angles = pickle.load(handle)
    handle.close

    run = RAF_calibrate_table(limb)
    
    # Move to starting location
    run.move_to_start(joint_angles)

    rate = rospy.Rate(10)
    listener = tf.TransformListener()
    
    count = 0
    while count < 10:
        rate.sleep()
        count += 1
        try:
            (trans_cam_world, rot_cam_world) = listener.lookupTransform('/world', '/_link', rospy.Time(0))
            (trans_hand_world, rot_hand_world) = listener.lookupTransform('/world', '/left_gripper', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Error in transform lookup.")
            continue


    T_cam_world = run.tf_to_matrix(trans_cam_world, rot_cam_world)
    T_hand_world = run.tf_to_matrix(trans_hand_world, rot_hand_world)

    print("T_cam_world: ", T_cam_world)
    print("T_hand_world: ", T_hand_world)

    T_cam_hand = np.matmul(np.linalg.inv(T_hand_world), T_cam_world)
    # T_cam_hand = np.linalg.inv(T_cam_hand)
    print("T_cam_hand: ", T_cam_hand)

    trans = transform.translation_from_matrix(T_cam_hand)
    quat = transform.quaternion_from_matrix(T_cam_hand)

    print("\nTranslation from Camera to Left Gripper (X, Y, Z):")
    print(trans)
    print("\nQuaternion Rotation from Camera to Left Gripper (x, y, z, w):")
    print(quat)

    # input("\nPress ENTER to save transformation...")
    # T = run.soder_alt(P2, P1)
    # translation = transform.translation_from_matrix(T)
    # euler = transform.euler_from_matrix(T)
    # quat = transform.quaternion_from_matrix(T)

    # print("\n #### Summary ####")
    # print("\nTag Positions in Robot Frame:")
    # print(P1)
    # print("\nTag Positions in Camera Frame:")
    # print(P2)
    # print("\nTransformation Matrix from Robot to Camera Frame:")
    # print(T)
    # print("\nTranslation from Robot to Camera Frame (X, Y, Z):")
    # print(trans)
    # print("\nEuler Rotation from Robot to Camera Frame (Roll, Pitch, Yaw):")
    # print(rot)
    # print("\nQuaternion Rotation from Robot to Camera Frame (x, y, z, w):")
    # print(quat)

    return 0


if __name__ == '__main__':
    sys.exit(main())

