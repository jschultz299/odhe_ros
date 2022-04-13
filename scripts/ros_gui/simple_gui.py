#!/usr/bin env python

import rospy
from kivymd.app import MDApp
from kivy.lang import Builder
from std_msgs.msg import Bool, Int8

class TutorialApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.screen = Builder.load_file('/home/labuser/ros_ws/src/odhe_ros/scripts/ros_gui/simple_gui.kv')

    def build(self):
        return self.screen

    def my_function(self, *args):
        print("Button Pressed")

        self.screen.ids.my_label.text = 'Button pressed!'

        button_pressed = True
        button_pub.publish(button_pressed)

    def slider_function(self, slider_value):
        # print(int(slider_value))
        slider_pub.publish(int(slider_value))

if __name__ == '__main__':

    button_pub = rospy.Publisher('/button', Bool, queue_size=1)
    slider_pub = rospy.Publisher('/slider', Int8, queue_size=1)

    rospy.init_node('simple_gui', anonymous=True)

    TutorialApp().run()