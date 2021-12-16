#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#    
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Example
"""
import argparse
import struct
import sys
import copy
import math

import rospy
import numpy as np
import baxter_interface
from baxter_core_msgs.msg import EndpointState
from odhe_ros.msg import EyeTracker, Object
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

class odhe_cup_demo():
    def __init__(self, limb, above_distance = 0.15, behind_distance = 0.1, verbose=True):
        self.sub = rospy.Subscriber('dlt_ik', PoseStamped, self.dlt_callback)
        self.sub_ht = rospy.Subscriber('eyetracker', EyeTracker, self.ht_callback)
        self.selection = rospy.Subscriber('selected_object', Object, self.selection_callback)
        rospy.Subscriber('speech_command', String, self.speech_callback)
        self._limb_name = limb # string
        self._above_distance = above_distance # in meters
        self._behind_distance = behind_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        self.objX = None
        self.objY = None
        self.objZ = None
        self.ht_X = None
        self.ht_Y = None
        self.ht_Z = None
        self.last_speech_cmd = None
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

    def dlt_callback(self, msg):
        self.objX = msg.pose.position.x
        self.objY = msg.pose.position.y
        self.objZ = msg.pose.position.z

    def selection_callback(self, msg):
        self.selected_class = msg.class_index

    def ht_callback(self, msg):
        self.ht_X = msg.x / 39.37 # convert to meters
        self.ht_Y = msg.y / 39.37 # convert to meters
        self.ht_Z = msg.z / 39.37 # convert to meters

    def get_dlt(self):
        result = np.array([[self.objX], [self.objY], [self.objZ], [1]])
        return result

    def get_ht(self):
        result = np.array([[self.ht_X], [self.ht_Y], [self.ht_Z], [1]])
        return result

    def get_selection(self):
        result = self.selected_class
        return result

    def speech_callback(self, msg):
        # if what you say makes sense
        self.last_speech_cmd = msg.data

    def transform(self, p):
        # Transformation matrix from table frame to Baxter frame
        # From soder.m in matlab
        # T = np.array([[-0.6917, 0.7220, -0.0143, 0.5638], [-0.7221, -0.6918, -0.0028, 1.0928], [-0.0119, 0.0084, 0.9999, -0.2511], [0, 0, 0, 1]])
        T = np.array([[-0.7208, 0.6926, -0.0285, 0.6048], [-0.6930, -0.7208, -0.0097, 1.0702], [-0.0138, 0.0267, 0.9995, -0.2574], [0, 0, 0, 1]])
        result = np.matmul(T, p)
        return result

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
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

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach slightly above and behind the cup
        approach.position.x = approach.position.x - self._behind_distance
        approach.position.z = approach.position.z + self._above_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

        # Lower to just behind the cup
        print("Lowering...\n")
        approach.position.z = approach.position.z - self._above_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _hover(self, pose):
        hover = copy.deepcopy(pose)
        # hover the arm just over the place point
        hover.position.z = hover.position.z + self._above_distance
        joint_angles = self.ik_request(hover)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x 
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._above_distance
        ik_pose.orientation.x = current_pose['orientation'].x 
        ik_pose.orientation.y = current_pose['orientation'].y 
        ik_pose.orientation.z = current_pose['orientation'].z 
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick_up_cup(self, poseCup, poseApproach, poseLower, poseHover):
        # open the gripper
        print("Opening the gripper...\n")
        self.gripper_open()
        # servo behind pose
        print("Approaching the cup...\n")
        self._servo_to_pose(poseApproach)
        self._servo_to_pose(poseLower)
        # servo to pose
        print("Moving to the cup...\n")
        self._servo_to_pose(poseCup)
        # close gripper
        print("Closing the gripper...\n")
        self.gripper_close()
        # retract to clear object
        print("Retracting the gripper...\n")
        self._servo_to_pose(poseHover)

    def move_to_user(self, pose):
        # servo to pose
        print("Moving to the user...\n")
        self._servo_to_pose(pose)

    def place_cup(self, poseCup, poseHover):
        #servo above pose
        print("Approaching the place point...\n")
        self._servo_to_pose(poseHover)
        # servo to pose
        print("Lowering the cup...\n")
        self._servo_to_pose(poseCup)
        # open gripper
        print("Opening the gripper...\n")
        self.gripper_open()
        # retract to clear object
        print("Retracting the gripper...\n")
        self._servo_to_pose(poseHover)

    def spinOnce(self):
        self.rate = 10
        r = rospy.Rate(self.rate)
        r.sleep()
    
        
def main():
    """ODHE Project First Demo: Interaction with a cup

    This is the first demo for the advisory board meeting in January 2021.

    This code is meant to be run with any number of objects. Baxter will
    only interact with a cup. The interaction is described in the following
    steps. 1) Move to starting location. 2) Move to hover over cup. 3) Orient 
    gripper. 4) Descend to just behind cup, maintaining gripper orientation.
    5) Approach cup with gripper open. 6) Close gripper. 7) Move cup to 
    user's face. 8) Wait for 10s or user input. 9) Return cup to position. 
    10) Open gripper. 11) Return to starting location.
    """
    rospy.init_node("odhe_cup_demo")
    limb = 'left'
    above_distance = 0.1 # meters
    behind_distance = 0.1 # meters
    speech_command = 0
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_e0': 0.04295146206079158,
                             'left_e1': 2.1142090209030715,
                             'left_s0': -0.2109223583342444,
                             'left_s1': -0.5188690015022412,
                             'left_w0': -0.18791264651596318,
                             'left_w1': -1.570029336400721,
                             'left_w2': -0.104310693576208132}

    demo = odhe_cup_demo(limb, above_distance, behind_distance)

    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.333405000183,
                             y=0.734164167355,
                             z=0.204318832542,
                             w=0.555065668091)

    # Mag source box position in the table frame (meters)
    #msb_x = -.72
    #msb_y = 0
    #msb_z = .485
    #msb = np.array([[msb_x], [msb_y], [msb_z], [0]])
    # Head tracker position relative to mag source box in the table frame (meters)
    #htCoord_table = demo.get_ht()
    #arr = np.append([msb], [htCoord_table], axis=0)
    # userCoord position relative to head tracker position in the table frame (meters)
    #userCoord_ht = np.array([[.35], [.11], [-.18], [0]])
    #arr = np.append(arr, [userCoord_ht], axis=0)
    #userCoord_table = sum(arr)
    # User position
    #userCoord_baxter = demo.transform(userCoord_table)

    # This is how you might do multiple objects
    #object_poses = list()
    #object_poses.append(Pose(
    #    position=Point(x=float(cupCoord_baxter[0]), y=float(cupCoord_baxter[1]), z=float(cupCoord_baxter[2])),
    #    orientation=overhead_orientation))

    for i in range(3):
        print("Reach Number: " + str(i+1))
        # 1) Move to starting location
        demo.move_to_start(starting_joint_angles)
        selection = demo.get_selection()

        # Only continue if a cup is selected
        print("Look at object to select it and issue voice command.\n")
        while not rospy.is_shutdown():
            demo.spinOnce()
            speech_command = demo.last_speech_cmd
            selection = demo.get_selection()
            if(speech_command is not None and "grab" in speech_command and selection == 2.0):
                break
        print("Selection Made!\n")
        if selection == 2.0:
            obj_class = "cup"
        print("Looking at: " + obj_class)

        # 2) Acquire poses based on current cup location
        #raw_input("Press ENTER to aqcuire poses.")
        print("Acquiring poses...\n") 

        # Cup position from dlt node
        cupCoord_table = demo.get_dlt()
        cupCoord_baxter = demo.transform(cupCoord_table)

        cupPose = Pose(
            position=Point(x=float(cupCoord_baxter[0]), y=float(cupCoord_baxter[1]), z=float(cupCoord_baxter[2])),
            orientation=overhead_orientation)

        # Gripper above and behind cup
        approachCoord_table = copy.deepcopy(cupCoord_table)
        approachCoord_table[0] = approachCoord_table[0] + behind_distance
        approachCoord_table[2] = approachCoord_table[2] + above_distance
        approachCoord_baxter = demo.transform(approachCoord_table)

        approachPose = Pose(
            position=Point(x=float(approachCoord_baxter[0]), y=float(approachCoord_baxter[1]), z=float(approachCoord_baxter[2])),
            orientation=overhead_orientation)

        # Gripper behind cup
        lowerCoord_table = copy.deepcopy(cupCoord_table)
        lowerCoord_table[0] = lowerCoord_table[0] + behind_distance
        lowerCoord_baxter = demo.transform(lowerCoord_table)

        lowerPose = Pose(
            position=Point(x=float(lowerCoord_baxter[0]), y=float(lowerCoord_baxter[1]), z=float(lowerCoord_baxter[2])),
            orientation=overhead_orientation)

        # Gripper above cup
        hoverCoord_table = copy.deepcopy(cupCoord_table)
        hoverCoord_table[2] = hoverCoord_table[2] + above_distance
        hoverCoord_baxter = demo.transform(hoverCoord_table)

        hoverPose = Pose(
            position=Point(x=float(hoverCoord_baxter[0]), y=float(hoverCoord_baxter[1]), z=float(hoverCoord_baxter[2])),
            orientation=overhead_orientation)

        # Gripper to user
        #userPose = Pose(
        #    position=Point(x=float(userCoord_baxter[0]), y=float(userCoord_baxter[1]), z=float(userCoord_baxter[2])),
        #    orientation=Quaternion(x=-0.31271493701, y=0.723633471488, z=0.226066494234, w=0.572239379367))
    
        # 3) Pick up the cup
        #raw_input("Press ENTER to pick up the cup.")
        demo.pick_up_cup(cupPose, approachPose, lowerPose, hoverPose)

        # 4) Move cup to user's face
        #raw_input("Press ENTER to move to the user.")
        #pose4 = Pose( position=Point(x=0.869335903943, y=1.1087532564, z=0.1241853977),
        #              orientation=Quaternion(x=-0.31271493701, y=0.723633471488, z=0.226066494234, w=0.572239379367))

        pose4 = Pose( position=Point(x=0.9334824956, y=1.00887567066, z=0.042478492701),
                    orientation=Quaternion(x=-0.317722427941, y=0.724208319059, z=0.178304977476, w=0.585475963978))
        demo.move_to_user(pose4)
        #demo.move_to_user(userPose)

        rospy.sleep(5.0)

        # 5) Place the cup
        #raw_input("Press ENTER to place the cup")
        demo._guarded_move_to_joint_position(starting_joint_angles)
        demo.place_cup(cupPose, hoverPose)
        demo.last_speech_cmd = None   # Resets the speech command

    return 0

if __name__ == '__main__':
    sys.exit(main())
