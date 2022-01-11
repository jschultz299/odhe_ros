#!/usr/bin/env python

# This program publishes DLT parameters from detected tags
# Created by Jack Schultz
# 7/29/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import rospy
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
        self.P = [0,0,0,0,0,0,0,0]

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


        # print("Top Left: " + str(self.last_top_left))
        # print("Top Right: " + str(self.last_top_right))
        # print("Bottom Right: " + str(self.last_bottom_right))
        # print("Bottom Left: " + str(self.last_bottom_left))  
        # print("\n-------------------------------\n")      

        # DLT with 4 tags (center of each tag)
        # These must be updated if the marker's positions are changed!
        # Top left tag is the origin to stay consistent with image frame coordinates
        # +X is to the right, +Y is down
        X1 = 0          # Bottom left corner marker X coordinate (m)
        Y1 = .25       # Bottom left corner marker Y coordinate (m)
        X2 = .40       # Bottom right corner marker X coordinate (m)
        Y2 = .25       # Bottom right corner marker Y coordinate (m)
        X3 = .40       # Top right corner marker X coordinate (m)
        Y3 = 0          # Top right corner marker Y coordinate (m)
        X4 = 0          # Top left corner marker X coordinate (m)
        Y4 = 0          # Top left corner marker Y coordinate (m)
    
        # Divide pixel by camera resolution to normalize between 0 and 1 for numerical stability
        x1 = self.last_bottom_left[0] / 640
        y1 = self.last_bottom_left[1] / 480
        x2 = self.last_bottom_right[0] / 640
        y2 = self.last_bottom_right[1] / 480
        x3 = self.last_top_right[0] / 640
        y3 = self.last_top_right[1] / 480
        x4 = self.last_top_left[0] / 640
        y4 = self.last_top_left[1] / 480
        
        A = np.array([[X1, Y1, 1, 0, 0, 0, x1*X1, x1*Y1], [0, 0, 0, X1, Y1, 1, y1*X1, y1*Y1], [X2, Y2, 1, 0, 0, 0, x2*X2, x2*Y2], [0, 0, 0, X2, Y2, 1, y2*X2, y2*Y2], [X3, Y3, 1, 0, 0, 0, x3*X3, x3*Y3], [0, 0, 0, X3, Y3, 1, y3*X3, y3*Y3], [X4, Y4, 1, 0, 0, 0, x4*X4, x4*Y4], [0, 0, 0, X4, Y4, 1, y4*X4, y4*Y4]])
        b = np.array([-x1, -y1, -x2, -y2, -x3, -y3, -x4, -y4])
        self.P = np.linalg.solve(A, b)

        self.publish()

    def publish(self):
        msg = DltParams()
        msg.P1 = self.P[0]
        msg.P2 = self.P[1]
        msg.P3 = self.P[2]
        msg.P4 = self.P[3]
        msg.P5 = self.P[4]
        msg.P6 = self.P[5]
        msg.P7 = self.P[6]
        msg.P8 = self.P[7]

        self.pub.publish(msg)
        self.loop_rate.sleep()
        
    def start(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    rospy.init_node("DLT", anonymous=True)
    my_node = DLT()
    my_node.start()