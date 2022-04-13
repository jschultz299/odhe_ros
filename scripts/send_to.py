#!/usr/bin/env python

# This program sends Baxter to specified position
# Written by Jack Schultz
# Created 1/11/22

import rospy
import baxter_interface
import sys
import pickle
import sys, getopt

class control_baxter():
    def __init__(self, limb, verbose=True):
        # Initialize parameters
        self._limb_name = limb
        self._verbose = False
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        left = baxter_interface.Gripper('left', baxter_interface.CHECK_VERSION)
        if self._verbose:
            print("Getting robot state... ")
            print("Enabling robot... ")
        self._rs.enable()
        if self._verbose:
            print("Calibrating gripper...")
        left.calibrate()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

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

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

def main(argv):

    rospy.init_node("send_baxter_home")
    limb = 'left'

    run = control_baxter(limb)

    target = ''

    try:
        opts, args = getopt.getopt(argv,"ht:",["target=",])
    except getopt.GetoptError:
        print('Usage: send_to.py -t <target>')
        print('Target Options: ')
        print('   - home')
        print('   - GPD')
        sys.exit(2)
    
    if len(opts) > 0:
        for opt, arg in opts:
            if opt == '-h':
                print('send_to.py -t <target>')
                sys.exit()
            elif opt in ("-t", "--target"):
                target = arg
            else:
                target = "home"
    else:
        target = "home"
            
    print('Target:', target)

    if target == "GPD":
        home_joint_angles = dict()
        home_joint_angles['left_e0'] = -0.4126408319411763
        home_joint_angles['left_e1'] = 1.6766410011587571
        home_joint_angles['left_s0'] = 0.3056456719861687
        home_joint_angles['left_s1'] = -1.358339987672534
        home_joint_angles['left_w0'] = 0.18331070415230694
        home_joint_angles['left_w1'] = 0.5495486172599495
        home_joint_angles['left_w2'] = -0.30833013836496814
    elif target == "home":
        # Read home position set in Calibration.py
        with open('/home/labuser/raf/set_positions/home_position.pkl', 'rb') as handle:
            home_joint_angles = pickle.load(handle)
        handle.close()
    else:
        print('Usage: send_to.py -t <target>')
        print('Target Options: ')
        print('   - home')
        print('   - GPD')
        sys.exit(2)

    run.move_to_angles(home_joint_angles)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))