#!/usr/bin/env python2

# @author glauberrleite

import rospy

import numpy as np

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

class Controller:
    def __init__(self):
        self.L1 = 0.5
        self.L2 = 0.35
        self.L3 = 0.4
        self.L4 = 0.05

        # Controller gains
        self.Kp = 1 * np.eye(3)
        self.Kd = 1 * np.eye(3)

        self.k = 0.2 # Damped pseudo-inverse factor
        self.q = np.zeros((4, 1))

        self.x_d = self.fkm(self.q)
        self.d_x_d = np.zeros((3, 1))
        self.dd_x_d = np.zeros((3, 1))

        rospy.init_node('motion_control_accelaration')

        self.pub = rospy.Publisher('/arm_controller/command', Float64MultiArray, queue_size=10)

        rospy.Subscriber('/joint_states', JointState, self.set_q)
        rospy.Subscriber('/trajectory', JointTrajectoryPoint, self.set_x_d)

    def set_q(self, state):
        self.q = np.array([[state.position[0]],[state.position[1]],[state.position[2]],[state.position[3]]])

    def set_x_d(self, point):
        self.x_d = np.array([[point.positions[0]],[point.positions[1]],[point.positions[2]]])
        self.d_x_d = np.array([[point.velocities[0]],[point.velocities[1]],[point.velocities[2]]])
        self.dd_x_d = np.array([[point.accelerations[0]],[point.accelerations[1]],[point.accelerations[2]]])

    def fkm(self, q):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4

        x_e = np.zeros((3, 1))

        x_e[0] = np.cos(q[0])*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        x_e[1] = np.sin(q[0])*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        x_e[2] = L1 - L2*np.sin(q[1]) - L3*np.sin(q[1] + q[2]) - L4*np.sin(q[1] + q[2] + q[3])

        return x_e

    def jacobian(self, q):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4

        jacob = np.zeros((3, 4))

        jacob[0,0] = - np.sin(q[0])*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        jacob[1,0] = np.cos(q[0])*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        jacob[2,0] = 0

        jacob[0,1] = - np.cos(q[0])*(L2*np.sin(q[1]) + L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        jacob[1,1] = - np.sin(q[0])*(L2*np.sin(q[1]) + L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        jacob[2,1] = - L2*np.cos(q[1]) - L3*np.cos(q[1] + q[2]) - L4*np.cos(q[1] + q[2] + q[3])

        jacob[0,2] = - np.cos(q[0])*(L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        jacob[1,2] = - np.sin(q[0])*(L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        jacob[2,2] = - L3*np.cos(q[1] + q[2]) - L4*np.cos(q[1] + q[2] + q[3])

        jacob[0,3] = - np.cos(q[0])*L4*np.sin(q[1] + q[2] + q[3])
        jacob[1,3] = - np.sin(q[0])*L4*np.sin(q[1] + q[2] + q[3])
        jacob[2,3] = - L4*np.cos(q[1] + q[2] + q[3])

        return jacob

    def d_jacobian(self, q, d_q):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4

        d_jacob = np.zeros((3, 4))

        d_jacob[0,0] = np.sin(q[0])*(L2*np.sin(q[1])*d_q[1] + L3*np.sin(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.sin(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    - np.cos(q[0])*d_q[0]*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        d_jacob[1,0] = - np.cos(q[0])*(L2*np.sin(q[1])*d_q[1] + L3*np.sin(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.sin(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    - np.sin(q[0])*d_q[0]*(L2*np.cos(q[1]) + L3*np.cos(q[1] + q[2]) + L4*np.cos(q[1] + q[2] + q[3]))
        d_jacob[2,0] = 0

        d_jacob[0,1] = - np.cos(q[0])*(L2*np.cos(q[1])*d_q[1] + L3*np.cos(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    + np.sin(q[0])*d_q[0]*(L2*np.sin(q[1]) + L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        d_jacob[1,1] = - np.sin(q[0])*(L2*np.cos(q[1])*d_q[1] + L3*np.cos(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    - np.cos(q[0])*d_q[0]*(L2*np.sin(q[1]) + L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        d_jacob[2,1] = L2*np.sin(q[1])*d_q[1] + L3*np.sin(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.sin(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])

        d_jacob[0,2] = - np.cos(q[0])*(L3*np.cos(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    + np.sin(q[0])*d_q[0]*(L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        d_jacob[1,2] = - np.sin(q[0])*(L3*np.cos(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])) \
                    - np.cos(q[0])*d_q[0]*(L3*np.sin(q[1] + q[2]) + L4*np.sin(q[1] + q[2] + q[3]))
        d_jacob[2,2] = L3*np.sin(q[1] + q[2])*(d_q[1] + d_q[2]) + L4*np.sin(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])

        d_jacob[0,3] = - np.cos(q[0])*L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3]) \
                    + np.sin(q[0])*d_q[0]*L4*np.sin(q[1] + q[2] + q[3])
        d_jacob[1,3] = - np.sin(q[0])*L4*np.cos(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3]) \
                    - np.cos(q[0])*d_q[0]*L4*np.sin(q[1] + q[2] + q[3])
        d_jacob[2,3] = L4*np.sin(q[1] + q[2] + q[3])*(d_q[1] + d_q[2] + d_q[3])

        return d_jacob

    def pinv(self, jacobian):
        pinv_jacobian = np.dot(np.transpose(jacobian), np.linalg.inv(np.dot(jacobian,np.transpose(jacobian)) + (self.k**2)*np.eye(3)))
        return pinv_jacobian

    def computeSecondaryTaskTerm(self):
        # In general, all joint can move in the interval [-pi, pi]
        q_min = -np.pi * np.ones(4)
        q_max = np.pi * np.ones(4)
        # Except for the third Joint which will work in [-pi/2, pi/2]
        q_min[2] = -np.pi/2
        q_max[2] = np.pi/2
        n = 4
        gain = 1

        result = np.zeros((4,1))

        for i in range(0, len(result)):
            result[i] = - gain * (1.0/(n * (q_max[i] - q_min[i])**2))

        return result

controller = Controller()
Ts = 0.01
d_q = np.zeros((4, 1))
dd_q = np.zeros((4, 1))

while not rospy.is_shutdown():

    #print(controller.fkm(controller.q))

    error = controller.x_d - controller.fkm(controller.q)
    d_error = controller.d_x_d - np.dot(controller.jacobian(controller.q), d_q)

    jacobian = controller.jacobian(controller.q)
    jacobian_pinv = controller.pinv(jacobian)

    dd_q_0 = controller.computeSecondaryTaskTerm()

    dd_q = np.dot(jacobian_pinv, \
                    controller.dd_x_d + np.dot(controller.Kd,d_error) + np.dot(controller.Kp,error) \
                    - np.dot(controller.d_jacobian(controller.q,d_q), d_q)) \
                    +  np.dot((np.eye(4) - np.dot(jacobian_pinv, jacobian)), dd_q_0)
    d_q += dd_q*Ts
    q = controller.q + d_q*Ts

    # publish q
    msg = Float64MultiArray()
    msg.data = q
    controller.pub.publish(msg)

    rospy.sleep(Ts)
