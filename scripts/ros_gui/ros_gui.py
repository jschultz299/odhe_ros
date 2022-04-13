#!/usr/bin env python

import rospy
from kivy.lang import Builder
from std_msgs.msg import Bool, Int8
from kivymd.app import MDApp
from kivy.utils import get_color_from_hex

class TutorialApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.start_screen = Builder.load_file('/home/labuser/ros_ws/src/odhe_ros/scripts/ros_gui/start_screen.kv')
        self.navigation_screen = Builder.load_file('/home/labuser/ros_ws/src/odhe_ros/scripts/ros_gui/navigation_screen.kv')
        # self.selection_screen = Builder.load_file('/home/labuser/ros_ws/src/odhe_ros/scripts/ros_gui/selection_screen.kv')

        # Initial screen
        self.screen = self.start_screen
        self.screen = self.navigation_screen
        # self.screen = self.selection_screen

    def build(self):
        # self.theme_cls.material_style = "M3"
        screen = self.screen
        return screen

    def start_button_function(self, *args):
        print("Button Pressed!")
        button_pressed = True
        button_pub.publish(button_pressed)
        self.screen = self.navigation_screen

if __name__ == '__main__':

    button_pub = rospy.Publisher('/button', Bool, queue_size=1)

    rospy.init_node('ros_gui', anonymous=True)

    TutorialApp().run()