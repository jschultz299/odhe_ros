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

def main():
    """
    Record AR tag positions in robot frame

    """
    rospy.init_node("calibrate_table")
    limb = 'left'

    # Read home position set in Calibration.py
    with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
        joint_angles = pickle.load(handle)
    handle.close

    run = RAF_calibrate_table(limb)
    
    # Move to starting location
    run.move_to_start(joint_angles)

    # Here you can update the home position if you want to.
    input("Press ENTER to save Home Position.")

    current_joint_angles = run.get_current_joint_angles()
    with open("/home/labuser/raf/set_positions/home_position.pkl","wb") as file:
        pickle.dump(current_joint_angles, file)
    file.close()

    joint_angles = current_joint_angles

    run.move_to_start(joint_angles)

    P = list()
    Z = list()

    tag_name_list = ["Top Right", "Bottom Right", "Bottom Left", "Top Left"]

    num_points = 4

    for i in range(num_points):
        # Move robot arm to tag centers in order
        input("\nMove robot arm to "  + str(tag_name_list[i]) + " tag center. Press ENTER to record.")

        # Save centroid of food item in robot frame
        current_pose = run.get_current_pose()
        point_robot = [current_pose['position'].x, current_pose['position'].y, current_pose['position'].z]
        print("Point " + str(i+1) + " in robot frame: " + str(point_robot))

        # Save tag positions in a list
        P.append(point_robot)

        # Move to starting location
        run.move_to_start(joint_angles)

    print("\nAll tag positions recorded.")

    # Write to a text file
    input("\nPress ENTER to save tag positions...")

    with open("/home/labuser/raf/set_positions/tag_positions.pkl","wb") as file:
        pickle.dump(P, file)    
    file.close()
    print("Tag positions saved.")

    return 0


if __name__ == '__main__':
    sys.exit(main())

