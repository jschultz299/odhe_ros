#!/usr/bin/env python

import numpy as np
import tf.transformations as tf
import random, math

# angle = (random.random() - 0.5) * (2*math.pi)
# direc = np.random.random(3) - 0.5
# point = np.random.random(3) - 0.5
# R0 = tf.rotation_matrix(angle, direc, point)
# angle, direc, point = tf.rotation_from_matrix(R0)
# R1 = tf.rotation_matrix(angle, direc, point)
# euler = tf.euler_from_matrix(R0)
# quat = tf.quaternion_from_euler(euler[0], euler[1], euler[2])
# quat2 = tf.quaternion_from_matrix(R0)
# print("#### Test ####")
# print("Angle: ", angle)
# print("Direction: ", direc)
# print("Point: ", point)
# print("R0: ", R0)
# print("R1: ", R1)
# print("Euler: ", euler)
# print("Quaternion: ", quat)
# print("Quaternion2: ", quat2)
# print(tf.is_same_transform(R0, R1))
# print("#### End Test ####")

# Get Transformation between camera and gripper (from calibrate_camera.py)
with open('/home/labuser/ros_ws/src/odhe_ros/scripts/Transformation.npy', 'rb') as f:
    T = np.load(f)
f.close()

trans = tf.translation_from_matrix(T)
rot = tf.rotation_from_matrix(T)
euler = tf.euler_from_matrix(T)
quat = tf.quaternion_from_matrix(T)

print("T: ", T)
print("Translation: ", trans)
print("Rotation: ", rot)
print("Euler: ", euler)
print("Quaternion: ", quat)