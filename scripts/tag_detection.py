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
from odhe_ros.msg import TagDetection, TagDetectionArray
from sensor_msgs.msg import Image

class tag_detection(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.detector = apriltag("tag36h11")
        self.tag = TagDetection()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.image_pub = rospy.Publisher('tag_detections_image', Image, queue_size=10)
        self.result_pub = rospy.Publisher('tag_detections', TagDetectionArray, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera2/color/image_raw", Image, self.callback)
        # rospy.Subscriber("/arm_camera_objects", Image, self.callback)

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
        detections = self.detector.detect(self.img)
        
        return detections

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

    # Static parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    while not rospy.is_shutdown():

        img = run.get_img()

        if img is not None:
            detections = run.detect_tags(img)

            # print(detections)

            # Create the tag array to be published (pixels)
            tag_array = TagDetectionArray()
            for i in range(len(detections)):
                tag = TagDetection()
                tag.id = int(detections[i]["id"])
                tag.center = detections[i]["center"]
                tag.corner_bl = detections[i]["lb-rb-rt-lt"][0]
                tag.corner_br = detections[i]["lb-rb-rt-lt"][1]
                tag.corner_tr = detections[i]["lb-rb-rt-lt"][2]
                tag.corner_tl = detections[i]["lb-rb-rt-lt"][3]
                tag_array.detections.append(tag)

                # Draw detections on the image
                bl = (int(tag.corner_bl[0]), int(tag.corner_bl[1]))
                br = (int(tag.corner_br[0]), int(tag.corner_br[1]))
                tl = (int(tag.corner_tl[0]), int(tag.corner_tl[1]))
                tr = (int(tag.corner_tr[0]), int(tag.corner_tr[1]))

                text = str(tag.id)
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                cv2.line(img, bl, tl, (255,0,0), 1)  # edge opposite x-axis (tag frame)
                cv2.line(img, tl, tr, (255,0,0), 1)  # edge opposite x-axis (tag frame)
                cv2.arrowedLine(img, br, tr, (0,0,255), 1)  # x-axis (tag frame)
                cv2.arrowedLine(img, br, bl, (0,255,0), 1)  # y-axis (tag frame)
                org = (int(tag.center[0] - (textsize[0] / 4)), int(tag.center[1] + (textsize[1] / 4)))
                cv2.putText(img, text, org, font, .6, (40,215,255), 2, cv2.LINE_4)

            # print(tag_array)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            run.publish(img_msg, tag_array)
            # cv2.imshow("image", img)
            # cv2.waitKey(33)

    return 0

if __name__ == '__main__':
    sys.exit(main())
