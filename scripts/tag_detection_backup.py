#!/usr/bin/env python

# This program publishes detected tag corners in pixels
# Created by Jack Schultz
# 7/29/21

"""
Robot-Assisted Feeding for Individuals with Spinal Cord Injury
"""

import cv2
import rospy
import sys
import numpy as np
from apriltag import apriltag
from cv_bridge import CvBridge
from odhe_ros.msg import TagDetection
from sensor_msgs.msg import Image

class tag_detection(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.detector = apriltag("tag36h11")
        self.tag = TagDetection()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.image_pub = rospy.Publisher('tag_detections_image', Image, queue_size=10)
        self.result_pub = rospy.Publisher('tag_detections', TagDetection, queue_size=10)

        # Subscribers
        # rospy.Subscriber("/camera2/color/image_raw", Image, self.callback)
        rospy.Subscriber("/arm_camera_objects", Image, self.callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_img(self):
        result = self.image
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

    def detect_tags(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.detections = self.detector.detect(self.img)

        if len((self.detections)) > 1:
            rospy.logerr_once("Only one tag can be in the frame!")
        elif len((self.detections)) < 1:
            rospy.logerr_once("No tags detected!")
            self.tag = TagDetection()
        else:
            self.tag.id = self.detections[0]['id']
            self.tag.center.x = self.detections[0]['center'][0]
            self.tag.center.y = self.detections[0]['center'][1]
            temp = self.detections[0]['lb-rb-rt-lt']
            self.tag.corner_bl.x = temp[0][0]
            self.tag.corner_bl.y = temp[0][1]
            self.tag.corner_br.x = temp[1][0]
            self.tag.corner_br.y = temp[1][1]
            self.tag.corner_tr.x = temp[2][0]
            self.tag.corner_tr.y = temp[2][1]
            self.tag.corner_tl.x = temp[3][0]
            self.tag.corner_tl.y = temp[3][1]
        
        return self.tag

    def publish(self, img, result):
        self.image_pub.publish(img)
        self.result_pub.publish(result)
        self.loop_rate.sleep()

def main():
    """Detect Tags

        This code detects AR tags and publishes the tag id, center coordinate, and 
        corner coordinates in pixels.

        TODO: Make this code work with multiple tags detected
    """

    bridge = CvBridge()
    rospy.init_node("Tag_Detection")
    run = tag_detection()

    count = 0
    while not rospy.is_shutdown():

        img = run.get_img()

        if img is not None:
            detections = run.detect_tags(img)

            if detections.center.x != 0 and detections.center.y != 0 and detections.corner_tl.x != 0 and detections.corner_tl.y != 0:
                bl = (int(detections.corner_bl.x), int(detections.corner_bl.y))
                br = (int(detections.corner_br.x), int(detections.corner_br.y))
                tr = (int(detections.corner_tr.x), int(detections.corner_tr.y))
                tl = (int(detections.corner_tl.x), int(detections.corner_tl.y))
                center = (int(detections.center.x), int(detections.center.y))

                text = str(detections.id)
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.line(img, bl, tl, (255,0,0), 1)  # edge opposite x-axis (tag frame)
                cv2.line(img, tl, tr, (255,0,0), 1)  # edge opposite x-axis (tag frame)
                cv2.arrowedLine(img, br, tr, (0,0,255), 1)  # x-axis (tag frame)
                cv2.arrowedLine(img, br, bl, (0,255,0), 1)  # y-axis (tag frame)

                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                org = (int(center[0] - (textsize[0] / 4)), int(center[1] + (textsize[1] / 4)))

                cv2.putText(img, text, org, font, .6, (40,215,255), 2, cv2.LINE_4)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            run.publish(img_msg, detections)
            # cv2.imshow("image", img)
            # cv2.waitKey(33)

        count = count + 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
