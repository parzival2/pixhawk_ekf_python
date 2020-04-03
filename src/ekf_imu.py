#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
from filterpy.kalman import ExtendedKalmanFilter
from sympy import symbols, Matrix, pprint, init_printing, zeros
import numpy as np
from pyquaternion.quaternion import Quaternion
from geometry_msgs.msg import PoseStamped
import scipy

init_printing(True)

class PixhawkEKF(ExtendedKalmanFilter):
    def __init__(self):
        super(PixhawkEKF, self).__init__(dim_x=7, dim_z=3)
        # Initialize as a unit quaternion.
        self.x[0, 0] = 1.
        self.Q = self.Q
        self.P[0:4, 0:4] = self.P[0:4, 0:4] * 0.01
        self.P[4:, 4:] = self.P[4:, 4:] * 0.001
        self.R = self.R * (0.05)
        # Timestamp to calculate the time difference between each messages
        self.timeStamp = None
        # Time difference between successive gyro measurements
        self.deltaT = 0
        # Generate equations using sympy.
        self.generateEquations()
        # IMU rotation
        self.imuRotation = Quaternion(axis=[1., 0., 0.], degrees=180.0).rotation_matrix
        # Initialize node
        self.ekfNodeHandle = rospy.init_node('ekf_estimator')
        # Subscribers
        # We will be listensing on IMU topic
        self.imuSubscriber = rospy.Subscriber('imu', Imu, self.omImuMessageReceived)
        # Publisher
        self.cubePublisher = rospy.Publisher('cubevis', Marker, queue_size=10)
        self.posePublisher = rospy.Publisher('pose', PoseStamped, queue_size=10)
        rospy.spin()
    
    def generateEquations(self):
        ### xdot
        # Symbols for quaternion
        self.qw, self.qx, self.qy, self.qz = symbols('qw qx qy qz')
        # Symbols for gyro input
        self.gx, self.gy, self.gz = symbols('gx, gy, gz')
        # Symbols for accelerometer measurement
        self.ax, self.ay, self.az = symbols('ax ay az')
        # Symbols for gyroscope bias
        self.bx, self.by, self.bz = symbols('bwx bwy bwz')
        # Matrices
        # Gyro measurement
        gyro = Matrix([[self.gx], [self.gy], [self.gz]])
        # Accel measurement
        accel = Matrix([[self.ax], [self.ay], [self.az]])
        # Bias
        bias = Matrix([[self.bx], [self.by], [self.bz]])
        # State
        state = Matrix([[self.qw], [self.qx], [self.qy], [self.qz], 
                        [self.bx], [self.by], [self.bz]])
        omega2qdot = 0.5 * Matrix([[-self.qx, -self.qy, -self.qz],
                                    [self.qw, self.qz, -self.qy],
                                    [-self.qz, self.qw, self.qx],
                                    [self.qy, -self.qx, self.qw]])
        angVelJacobian = omega2qdot * (gyro - bias)
        self.xdot = zeros(7, 1)
        self.xdot[0, 0] = angVelJacobian[0]
        self.xdot[1, 0] = angVelJacobian[1]
        self.xdot[2, 0] = angVelJacobian[2]
        self.xdot[3, 0] = angVelJacobian[3]
        ### F
        self.fSympy = self.xdot.jacobian(state)
        print(self.fSympy)
        ### zhat
        # Calculate the estimated gravity from our state
        CNed2Body =  Matrix([[1-2*(self.qy**2+self.qz**2),2*(self.qx*self.qy+self.qz*self.qw),2*(self.qx*self.qz-self.qy*self.qw)],
                            [2*(self.qx*self.qy-self.qz*self.qw),1-2*(self.qx**2+self.qz**2),2*(self.qy*self.qz+self.qx*self.qw)],
                            [2*(self.qx*self.qz+self.qy*self.qw),2*(self.qy*self.qz-self.qx*self.qw),1-2*(self.qx**2+self.qy**2)]])
        self.gmps = symbols('g')
        trueGravity = Matrix([[0.0], [0.0], [self.gmps]])
        # self.zhat = Matrix([[2.0 * self.qx * self.qz - 2.0 * self.qw * self.qy],
        #                     [2.0 * self.qy * self.qz + 2.0 * self.qx * self.qw],
        #                     [self.qz * self.qz + self.qw * self.qw - self.qx * self.qx - self.qy * self.qy]])
        self.zhat = CNed2Body * trueGravity
        ### H
        self.hSympy = self.zhat.jacobian(state)
    
    def getPredictSubs(self, gyro):
        gx = gyro[0, 0]
        gy = gyro[1, 0]
        gz = gyro[2, 0]
        # Integrate
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        subs = {
            self.gx : gx,
            self.gy : gy,
            self.gz : gz,
            self.qw : qw,
            self.qx : qx,
            self.qy : qy,
            self.qz : qz,
            self.bx : self.x[4, 0],
            self.by : self.x[5, 0],
            self.bz : self.x[6, 0]
            }
        return subs
    
    def getUpdateSubs(self, state):
        # Integrate
        qw = state[0, 0]
        qx = state[1, 0]
        qy = state[2, 0]
        qz = state[3, 0]
        subs = {
            self.gmps : 1.0,
            self.qw : qw,
            self.qx : qx,
            self.qy : qy,
            self.qz : qz
        }
        return subs
    
    def predict(self, u=0):
        # Predict the state using euler integration.
        self.predict_x(u)
        # subs = self.getPredictSubs(gyro=u)
        # FSympy = self.fSympy.evalf(subs=subs)
        # F = np.array(FSympy).astype(float)
        # For simplicity
        gx = u[0, 0]
        gy = u[1, 0]
        gz = u[2, 0]
        # Biases
        bwx = self.x[0, 0]
        bwy = self.x[1, 0]
        bwz = self.x[2, 0]
        # Quaternions
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        F = np.array(
        [[0, 0.5*bwx - 0.5*gx, 0.5*bwy - 0.5*gy, 0.5*bwz - 0.5*gz, 0.5*qx, 0.5*qy, 0.5*qz],
        [-0.5*bwx + 0.5*gx, 0, 0.5*bwz - 0.5*gz, -0.5*bwy + 0.5*gy, -0.5*qw, -0.5*qz, 0.5*qy],
        [-0.5*bwy + 0.5*gy, -0.5*bwz + 0.5*gz, 0, 0.5*bwx - 0.5*gx, 0.5*qz, -0.5*qw, -0.5*qx],
        [-0.5*bwz + 0.5*gz, 0.5*bwy - 0.5*gy, -0.5*bwx + 0.5*gx, 0, -0.5*qy, 0.5*qx, -0.5*qw],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
        Q = np.eye(7, 7)
        Q[4:, 4:] = Q[4:, 4:] * 0.001
        Q[0:4, 0:4] = Q[0:4, 0:4] * 0.05
        # Discretize
        AA = np.zeros(shape=(14, 14))
        AA[0:7, 0:7] = -F
        AA[0:7, 7:] = Q
        AA[7:, 7:] = F.T
        # Exponential
        AA_dt = AA * self.deltaT
        AA_dt_sq = np.dot(AA_dt, AA_dt)
        AA_dt_cu = np.dot(AA_dt_sq, AA_dt)
        AA_dt_qu = np.dot(AA_dt_cu, AA_dt)
        BB = np.eye(14) + AA_dt + 0.5 * AA_dt_sq + (1/6.0) * AA_dt_cu + (1/np.math.factorial(4)) * AA_dt_qu
        self.F = BB[7:, 7:].T
        self.Q = np.dot(self.F, BB[0:7, 7:])
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
    
    def predict_x(self, u=0):
        # subs = self.getPredictSubs(gyro=u)
        # evaluated = self.xdot.evalf(subs=subs)
        # xdot_numpy = np.array(evaluated).astype(np.float)
        # This is inaccurate for any small value of deltaT as quaternions are not 
        # closed under elementwise-addition like vectors.
        # Solve the differential equation using exponential
        # self.x = self.x + xdot_numpy * self.deltaT
        # Quaternion integration
        # Integrate quaternion using the axis angle representation
        unbiasedGyro = u - self.x[4:]
        rotationVec = unbiasedGyro * self.deltaT
        angle = np.linalg.norm(rotationVec)
        if not np.isclose(angle, 0):
            quat = Quaternion(axis=rotationVec, angle=angle)
        else:
            quat = Quaternion([1., 0., 0., 0.])
        result = Quaternion(self.x[0:4, 0]) * quat
        if(self.x[0, 0] < 0):
            self.x[0, 0] = -self.x[0, 0]
            self.x[1, 0] = -self.x[1, 0]
            self.x[2, 0] = -self.x[2, 0]
            self.x[3, 0] = -self.x[3, 0]
        self.x[0:4, 0] = result.normalised.elements

    def rotateImuToPoseCoordSystem(self, imuMsg:Imu):
        # Rotate gyro measurements
        gx = imuMsg.angular_velocity.x
        gy = imuMsg.angular_velocity.y
        gz = imuMsg.angular_velocity.z
        # Gyro matrix
        gyroMat = np.array([[gx], [gy], [gz]])
        rotatedGyro = np.dot(self.imuRotation, gyroMat)
        # rotatedGyro = gyroMat
        # Accelerometer measurements
        ax = imuMsg.linear_acceleration.x
        ay = imuMsg.linear_acceleration.y
        az = imuMsg.linear_acceleration.z
        accelMat = np.array([[ax], [ay], [az]])
        rotatedAccel = np.dot(self.imuRotation, accelMat/np.linalg.norm(accelMat))
        # rotatedAccel = accelMat
        return rotatedGyro, rotatedAccel
    
    def measurementEstimate(self, state):
        # zhatSympy = self.zhat.evalf(subs=self.getUpdateSubs(state))
        # zhatNumpy = np.array(zhatSympy).astype(np.float)
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        estimatedAccel = np.array([[2*(qx * qz - qy * qw)],
                                   [2*(qy * qz + qx * qw)],
                                   [1 - 2 *(qx**2 + qy**2)]])
        return estimatedAccel

    
    def measurementJacobian(self, state):
        # hSympy = self.hSympy.evalf(subs=self.getUpdateSubs(state))
        # hNumpy = np.array(hSympy).astype(np.float)
        # Quaternions
        qw = self.x[0, 0]
        qx = self.x[1, 0]
        qy = self.x[2, 0]
        qz = self.x[3, 0]
        H = np.array([[-2*qy, 2*qz, -2*qw, 2*qx, 0, 0, 0],
                            [2*qx, 2*qw, 2*qz, 2*qy, 0, 0, 0], 
                            [0, -4*qx, -4*qy, 0, 0, 0, 0]])
        return H

    def ekfUpdate(self, accelData):
        self.update(accelData, HJacobian=self.measurementJacobian, 
                    Hx=self.measurementEstimate)
        self.x[0:4, 0] = Quaternion(self.x[0:4, 0]).normalised.elements
    
    def omImuMessageReceived(self, imuMsg:Imu):
        if(self.timeStamp):
            # print(imuMsg.header.stamp.to_sec())
            currentTimestamp = imuMsg.header.stamp.to_sec()
            self.deltaT = currentTimestamp - self.timeStamp
            gyro, accel = self.rotateImuToPoseCoordSystem(imuMsg)
            self.predict(u=gyro)
            self.timeStamp = currentTimestamp
            self.ekfUpdate(accel)
        else:
            # Initialize our state with the first incoming message of accelerometer
            self.timeStamp = imuMsg.header.stamp.to_sec()
        self.publishVisualizationMarker()
    
    def publishVisualizationMarker(self):
        # Visualization marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # Position
        marker.pose.position.x = 1.
        marker.pose.position.y = 1.
        marker.pose.position.z = 1.
        # Orientation
        flip = Quaternion([0., 0., 0., 1.])
        markerOrient = flip * Quaternion(self.x[0:4, 0])
        marker.pose.orientation.w = markerOrient[0]
        marker.pose.orientation.x = markerOrient[1]
        marker.pose.orientation.y = markerOrient[2]
        marker.pose.orientation.z = markerOrient[3]
        marker.scale.x = 0.7
        marker.scale.y = 1.
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # Publish the axes marker
        self.cubePublisher.publish(marker)
        # Pose 
        poseMsg = PoseStamped()
        poseMsg.header.frame_id = "map"
        poseMsg.pose.position.x = 1.
        poseMsg.pose.position.y = 1.
        poseMsg.pose.position.z = 1.
        # Orientation
        poseMsg.pose.orientation.w = markerOrient[0]
        poseMsg.pose.orientation.x = markerOrient[1]
        poseMsg.pose.orientation.y = markerOrient[2]
        poseMsg.pose.orientation.z = markerOrient[3]
        self.posePublisher.publish(poseMsg)
        
if __name__ == "__main__":
    ekf = PixhawkEKF()
