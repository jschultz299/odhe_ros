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
from odhe_ros.msg import TagDetection, DltParams
# from rospy.numpy_msg import numpy_msg
# from geometry_msgs.msg import Point, PoseStamped
# import tf
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
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("tag_detections", TagDetection, self.callback)

        # Pubishers
        self.pub = rospy.Publisher('dlt_params', DltParams, queue_size=10)
        
    def callback(self, tag):
        self.last_center = np.array([tag.center.x, tag.center.y])
        self.last_bottom_left = np.array([tag.corner_bl.x, tag.corner_bl.y])
        self.last_bottom_right = np.array([tag.corner_br.x, tag.corner_br.y])
        self.last_top_right = np.array([tag.corner_tr.x, tag.corner_tr.y])
        self.last_top_left = np.array([tag.corner_tl.x, tag.corner_tl.y])        

        # These must be updated if the marker's positions are changed!
        l = 0.063       # Tag length (m)    
        X1 = -(l/2)   # Bottom left corner marker X coordinate (m)
        Y1 = l/2      # Bottom left corner marker Y coordinate (m)
        X2 = -(l/2)   # Bottom right corner marker X coordinate (m)
        Y2 = -(l/2)   # Bottom right corner marker Y coordinate (m)
        X3 = l/2      # Top right corner marker X coordinate (m)
        Y3 = -(l/2)   # Top right corner marker Y coordinate (m)
        X4 = l/2      # Top left corner marker X coordinate (m)
        Y4 = l/2      # Top left corner marker Y coordinate (m)

        if self.last_center[0] == 0 and self.last_center[1] == 0 and self.last_top_left[1] == 0 and self.last_top_left[1] == 0:
            rospy.logwarn_once("Tag not visible")
        else: 
            x1 = self.last_bottom_left[0]
            y1 = self.last_bottom_left[1]
            x2 = self.last_bottom_right[0]
            y2 = self.last_bottom_right[1]
            x3 = self.last_top_right[0]
            y3 = self.last_top_right[1]
            x4 = self.last_top_left[0]
            y4 = self.last_top_left[1]
            
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