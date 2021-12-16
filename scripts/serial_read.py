#!/usr/bin/env python3
import rospy, serial
from std_msgs.msg import String, Header
from odhe_ros.msg import EyeTracker

def main():
    pub = rospy.Publisher('eyetracker', EyeTracker, queue_size=10)
    rospy.init_node('eyetrack', anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        with serial.Serial('/dev/ttyUSB0', 115200, timeout=1) as ser:
            try:
                line = ser.readline().decode('ascii').split('\t')[0:-1]
                if len(line) != 8:
                    continue
                m = EyeTracker()
                h = Header()
                for i  in range(len(line)):
                    line[i] = float(line[i])
                m.x, m.y, m.z, m.azim, m.elev, m.roll, m.cursor_x, m.cursor_y = line
                #if not (m.cursor_x == 0 and m.cursor_y == 0):
                h.stamp =  rospy.Time.now()
                m.header = h
                pub.publish(m)
            except Exception as e:
                print(e)
                continue
            rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
