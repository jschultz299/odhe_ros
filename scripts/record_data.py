#!/usr/bin/env python3
import rospy, serial
import sys
from std_msgs.msg import String, Header
from odhe_ros.msg import EyeTracker2, EndpointState

class record_data():
    def __init__(self, verbose=True):
        rospy.Subscriber("robot/limb/left/endpoint_state", EndpointState, self.endpoint_callback)

    def endpoint_callback(self, msg):
        self.robotX = msg.pose.position.x
        self.robotY = msg.pose.position.y
        self.robotZ = msg.pose.position.z

    def get_endpoint_state(self):
        result = [self.robotX, self.robotY, self.robotZ]
        return result

def main():
    rospy.init_node("record_data", anonymous=True)
    record = record_data()


    pub = rospy.Publisher('eyetracker', EyeTracker2, queue_size=10)
    publish_rate = 5
    rate = rospy.Rate(publish_rate) # 5 Hz
    frame_id = 0
    file1 = open("/home/labuser/ros_ws/src/odhe_ros/scripts/calibration_data/MyFile.txt","w+")
    file1.write("Time" + "\t" + "HeadX" + "\t" + "HeadY" + "\t" + "HeadZ" + "\t" + "HeadAzim" + "\t" + "HeadElev" + "\t" + "HeadRoll" + "\t" + "PupilH1" + "\t" + "PupilV1" + "\t" + "PupilA1" + "\t" + "PupilH2" + "\t" + "PupilV2" + "\t" + "PupilA2" + "\t" + "robotX" + "\t" + "robotY" + "\t" + "robotZ" + "\n")
    file1.close()
    while not rospy.is_shutdown():
        frame_id = frame_id + 1
        t = rospy.Time.now()
        if frame_id == 1:
            tstart = t.to_sec()     # start time in seconds
        with serial.Serial('/dev/ttyUSB0', 115200, timeout = 0.01) as ser:
            try:
                line = ser.readline().decode('UTF-8').split('\t')[0:-1]
                #line = ser.readline().split('\t')[0:-1]
                if len(line) != 12:
                    continue
            except Exception as e:
                print(e)
                continue

        m = EyeTracker2()
        for i in range(len(line)):
            line[i] = float(line[i])
        m.head_x, m.head_y, m.head_z, m.head_azim, m.head_elev, m.head_roll, m.pupil_H1, m.pupil_V1, m.pupil_A1, m.pupil_H2, m.pupil_V2, m.pupil_A2 = line
        m.header.stamp = t
        m.header.frame_id = str(frame_id)
        current_time = t.to_sec() - tstart

        endpoint_state = record.get_endpoint_state()
        robotX = endpoint_state[0]
        robotY = endpoint_state[1]
        robotZ = endpoint_state[2]
        data = (str(current_time) + "\t" + str(m.head_x) + "\t" + str(m.head_y) + "\t" + str(m.head_z) + "\t" + str(m.head_azim) + "\t" + str(m.head_elev) + "\t" + str(m.head_roll) + "\t" + str(m.pupil_H1) + "\t" +  str(m.pupil_V1) + "\t" + str(m.pupil_A1) + "\t" + str(m.pupil_H2) + "\t" + str(m.pupil_V2) + "\t" + str(m.pupil_A2) + "\t" + str(robotX) + "\t" + str(robotY) + "\t" + str(robotZ) + "\n")
        print((str(current_time) + "\t" + str(m.head_x) + "\t" + str(m.head_y) + "\t" + str(m.head_z) + "\t" + str(m.head_azim) + "\t" + str(m.head_elev) + "\t" + str(m.head_roll) + "\t" + str(m.pupil_H1) + "\t" +  str(m.pupil_V1) + "\t" + str(m.pupil_A1) + "\t" + str(m.pupil_H2) + "\t" + str(m.pupil_V2) + "\t" + str(m.pupil_A2) + "\t" + str(robotX) + "\t" + str(robotY) + "\t" + str(robotZ)))
        file1 = open("/home/labuser/ros_ws/src/odhe_ros/scripts/calibration_data/MyFile.txt","a+")
        file1.write(data)
        file1.close()
        pub.publish(m)
        rate.sleep()

if __name__ == '__main__':
    sys.exit(main())

