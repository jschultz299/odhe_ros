#!/usr/bin/env python

# This program publishes DLT parameters from detected tags
# Created by Jack Schultz
# 7/29/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import rospy, sys, pickle
import numpy as np
# from std_msgs.msg import String
from odhe_ros.msg import DltParams, TagDetectionArray
# from apriltag2_ros.msg import AprilTagDetectionArray

class DLT(object):
    def __init__(self):
        # Initialize parameters to some values.
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.x3 = 0
        self.y3 = 0
        self.x4 = 0
        self.y4 = 0
        # self.P = [0,0,0,0,0,0,0,0]

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Subscribers
        rospy.Subscriber("tag_detections", TagDetectionArray, self.callback)

        # Pubishers
        self.pub = rospy.Publisher('dlt_params', DltParams, queue_size=10)
        
    def callback(self, tag_array):

        # Tag Order: Start at top right and continue clockwise (tags 0-3)
        for i in range(len(tag_array.detections)):
            id = tag_array.detections[i].id
            if id == 0:
                self.last_top_right = np.array(tag_array.detections[i].center)
            elif id == 1:
                self.last_bottom_right = np.array(tag_array.detections[i].center)
            elif id == 2:
                self.last_bottom_left = np.array(tag_array.detections[i].center)
            elif id == 3:
                self.last_top_left = np.array(tag_array.detections[i].center)
            else:
                # print("Exactly 4 tags should be visible!")
                pass

        self.x1 = self.last_top_right[0] / 640
        self.y1 = self.last_top_right[1] / 480
        self.x2 = self.last_bottom_right[0] / 640
        self.y2 = self.last_bottom_right[1] / 480
        self.x3 = self.last_bottom_left[0] / 640
        self.y3 = self.last_bottom_left[1] / 480
        self.x4 = self.last_top_left[0] / 640
        self.y4 = self.last_top_left[1] / 480

    def compute_params(self,X1,Y1,X2,Y2,X3,Y3,X4,Y4):
        temp1 = [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
        temp2 = [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]
        print("Robot frame: ", temp1)
        print("Camera frame (normalized): ", temp2)
        A = np.array([[X1, Y1, 1, 0, 0, 0, self.x1*X1, self.x1*Y1], [0, 0, 0, X1, Y1, 1, self.y1*X1, self.y1*Y1], [X2, Y2, 1, 0, 0, 0, self.x2*X2, self.x2*Y2], [0, 0, 0, X2, Y2, 1, self.y2*X2, self.y2*Y2], [X3, Y3, 1, 0, 0, 0, self.x3*X3, self.x3*Y3], [0, 0, 0, X3, Y3, 1, self.y3*X3, self.y3*Y3], [X4, Y4, 1, 0, 0, 0, self.x4*X4, self.x4*Y4], [0, 0, 0, X4, Y4, 1, self.y4*X4, self.y4*Y4]])
        b = np.array([-self.x1, -self.y1, -self.x2, -self.y2, -self.x3, -self.y3, -self.x4, -self.y4])
        P = np.linalg.solve(A, b)
        return P

    def publish(self,P):
        msg = DltParams()
        msg.P1 = P[0]
        msg.P2 = P[1]
        msg.P3 = P[2]
        msg.P4 = P[3]
        msg.P5 = P[4]
        msg.P6 = P[5]
        msg.P7 = P[6]
        msg.P8 = P[7]

        self.pub.publish(msg)
        self.loop_rate.sleep()

def main():
    rospy.init_node("DLT", anonymous=True)
    run = DLT()
    print("Initializing...")
    rospy.sleep(2)

    # Read tag positions in robot frame from file (set in calibrate_table.py)
    with open('/home/labuser/raf/set_positions/tag_positions.pkl', 'rb') as handle:
        tag_positions = pickle.load(handle)
    handle.close

    # This is the order tag positions should be recorded in calibrate_table.py
    X1 = tag_positions[0][0]        # Top right corner marker X coordinate (m)
    Y1 = tag_positions[0][1]        # Top right corner marker Y coordinate (m)
    X2 = tag_positions[1][0]        # Bottom right corner marker X coordinate (m)
    Y2 = tag_positions[2][1]        # Bottom right corner marker Y coordinate (m)
    X3 = tag_positions[2][0]        # Bottom left corner marker X coordinate (m)
    Y3 = tag_positions[2][1]        # Bottom left corner marker Y coordinate (m)
    X4 = tag_positions[3][0]        # Top left corner marker X coordinate (m)
    Y4 = tag_positions[3][1]        # Top left corner marker Y coordinate (m)

    while not rospy.is_shutdown():
        params = run.compute_params(X1,Y1,X2,Y2,X3,Y3,X4,Y4)
        print("DLT Parameters: ", params)
        print("------------------------------------\n")
        run.publish(params)

if __name__ == '__main__':
    sys.exit(main())