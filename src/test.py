#!/usr/bin/env python2

# @author glauberrleite

import rospy

import numpy as np

from trajectory_msgs.msg import JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates

rospy.init_node('test')

pub = rospy.Publisher('/trajectory', JointTrajectoryPoint, queue_size=10)

sub_once = rospy.wait_for_message('/gazebo/link_states', LinkStates)

x_e = np.array([sub_once.pose[-1].position.x, sub_once.pose[-1].position.y, sub_once.pose[-1].position.z])

time = 20.0
Ts = 0.01
r = 0.7

for i in range(0, int(time/Ts)):

    # Computing desired position for current iteraction (polar coords)
    k = i/(time/Ts) # Normalization
    theta = k * np.pi

    x = np.zeros(3)
    x[0] = r * np.cos(theta)
    x[1] = 0
    x[2] = r * np.sin(theta)

    x[0] += (x_e[0] - r) # Offset
    x[2] += x_e[2]

    # Computing derivative of position
    omega = np.pi/time

    d_x = np.zeros(3)
    d_x[0] = - omega * r * np.sin(theta)
    d_x[1] = 0
    d_x[2] = omega * r * np.cos(theta)

    # Computing devirative of velocity
    dd_x = np.zeros(3)
    dd_x[0] = - (omega ** 2) * r * np.cos(theta)
    dd_x[1] = 0
    dd_x[2] = - (omega ** 2) * r * np.sin(theta)

    msg = JointTrajectoryPoint()
    msg.positions = [x[0], x[1], x[2]]
    msg.velocities = [d_x[0], d_x[1], d_x[2]]
    msg.accelerations = [dd_x[0], dd_x[1], dd_x[2]]

    pub.publish(msg)

    rospy.sleep(Ts)

# Last trajectory point means the system is stable, i.e. no velocity nor acceleration
x = np.zeros(3)
x[0] = x_e[0] - 2*r
x[1] = 0
x[2] = x_e[2]

d_x = np.zeros(3)
dd_x = np.zeros(3)

msg = JointTrajectoryPoint()
msg.positions = [x[0], x[1], x[2]]
msg.velocities = [d_x[0], d_x[1], d_x[2]]
msg.accelerations = [dd_x[0], dd_x[1], dd_x[2]]

pub.publish(msg)

print("Last trajectory point is:")
print(msg)
