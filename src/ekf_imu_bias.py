#! /usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
import numpy as np
from pyquaternion.quaternion import Quaternion
from geometry_msgs.msg import PoseStamped
from enum import IntFlag

class State(IntFlag):
    # Quaternion
    QW = 0
    QX = 1
    QY = 2
    QZ = 3
    # Gyro bias
    BX = 4
    BY = 5
    BZ = 6
    Size = 7

class Meas(IntFlag):
    AX = 0
    AY = 1
    AZ = 2
    Size = 3

class Input(IntFlag):
    GX = 0
    GY = 1
    GZ = 2
    Size = 3

class PixhawkEKFBias(object):
    """
    An EKF filter that also estimates the bias of Gyroscope in its state vector.
    For equations take a look at equations.ipynb notebook.
    """
    def __init__(self):
        # State matrix
        # The first four are quaternions and the last three are gyro biases.
        self.x = np.zeros(shape=(State.Size, 1))
        # Uncertainitiy covariance matrix
        self.P = np.eye(State.Size)
        # State transition matrix
        self.F = np.zeros(shape=(State.Size, State.Size))
        # Measurement jacobian matrix
        self.H = np.zeros(shape=(Meas.Size, State.Size))
        # Measurement covariance matrix
        self.R = np.eye(Meas.Size)
        # Process noise covariance matrix
        self.Q = np.eye(State.Size)
        self.initializeMatrices()
        # IMU rotation
        self.imuRotation = Quaternion(axis=[1., 0., 0.], degrees=180.0).rotation_matrix
        # Initialize node
        self.ekfNodeHandle = rospy.init_node('ekf_estimator')
        # Subscribers
        # We will be listensing on IMU topic
        self.imuSubscriber = rospy.Subscriber('imu/data_raw', Imu, self.omImuMessageReceived)
        # Sort of treated as a ground truth for comparison of accuracy. 
        # You would need imu_tools package.
        self.magdwickFilterSub = rospy.Subscriber('imu/data', Imu, self.onMagdwickOutputReceived)
        # Publisher
        self.cubePublisher = rospy.Publisher('cubevis', Marker, queue_size=10)
        self.posePublisher = rospy.Publisher('pose', PoseStamped, queue_size=10)
        # Magdwick
        self.magdwickOutPublisher = rospy.Publisher('magdwick', PoseStamped, queue_size=10)
        self.cubeMagdwickPublisher = rospy.Publisher('cubeMadgwick', Marker, queue_size=10)
        self.timeStamp = None
        rospy.spin()

    def initializeMatrices(self):
        # Initialize the Quaternion to be Identity. Later when we get the first
        # accelerometer measurement we will initialize it using roll and pitch
        self.x[State.QW, 0] = 1.
        # Initialize the Uncertainity covariance matrix to a little bit higher value
        # as we are un-certain about the initial state.
        self.P[State.QW : State.BX, State.QW : State.BX] *= 0.001
        self.P[State.BX:, State.BX:] *= 0.001**2
        # Process noise
        self.Q[State.BX:, State.BX:] *= 0.001 **2
        self.Q[State.QW:State.BX, State.QW:State.BX] *= 1E-3
        # Initialize the noise covariance matrix 
        self.R *= 1E-3
    
    def predict(self, gyro, dt):
        # Using euler integration
        # If you want to use the more inaccurate euler integration instead of integrating
        # quaternion using the axis angle representation uncomment this line. Make sure that
        # the exponential integration is deactivated.
        # self.useEulerIntegration(gyro, dt)
        unbiasedGyro = np.subtract(gyro, self.x[State.BX:])
        # Quaternion integration
        # Integrate quaternion using the axis angle representation
        rotationVec = unbiasedGyro * dt
        angle = np.linalg.norm(rotationVec)
        if not np.isclose(angle, 0):
            quat = Quaternion(axis=rotationVec, angle=angle)
        else:
            quat = Quaternion([1., 0., 0., 0.])
        result = Quaternion(self.x[State.QW:State.BX, 0]) * quat
        self.x[State.QW:State.BX, 0] = result.normalised.elements
        if(self.x[0, 0] < 0):
            self.x[0, 0] = -self.x[0, 0]
            self.x[1, 0] = -self.x[1, 0]
            self.x[2, 0] = -self.x[2, 0]
            self.x[3, 0] = -self.x[3, 0]
        self.computeStateTransitionMatrix(gyro, dt)
    
    def useEulerIntegration(self, gyro, dt):
        """
        A less accurate version of Quaternion integration.
        x_hat = x_hat + xdot * dt
        """
        qw = self.x[State.QW, 0]
        qx = self.x[State.QX, 0]
        qy = self.x[State.QY, 0]
        qz = self.x[State.QZ, 0]
        # Biases
        wbx = self.x[State.BX, 0]
        wby = self.x[State.BY, 0]
        wbz = self.x[State.BZ, 0]
        # For simplicity
        gx = gyro[Input.GX, 0]
        gy = gyro[Input.GY, 0]
        gz = gyro[Input.GZ, 0]
        xdot = np.array([[-0.5*qx*(gx - wbx) - 0.5*qy*(gy - wby) - 0.5*qz*(gz - wbz)],
                         [0.5*qw*(gx - wbx) - 0.5*qy*(gz - wbz) + 0.5*qz*(gy - wby)],
                         [0.5*qw*(gy - wby) + 0.5*qx*(gz - wbz) - 0.5*qz*(gx - wbx)],
                         [0.5*qw*(gz - wbz) - 0.5*qx*(gy - wby) + 0.5*qy*(gx - wbx)],
                         [0],
                         [0],
                         [0]])
        self.x = self.x + xdot * dt
        self.normalizeQuaternion()
    
    def normalizeQuaternion(self):
        """
        Normalize the quaternion in the state vector
        """
        self.x[State.QW:State.BX, 0] = Quaternion(self.x[State.QW:State.BX, 0]).normalised.elements
    
    def update(self, accel):
        """
        Correct the state vector using the Accelerometer measurement
        """
        H = self.computeMeasurementJacobianMatrix()
        # Calculate the update loop
        # Kalman gain
        PHT = np.dot(self.P, H.T)
        S = np.dot(H, PHT) + self.R
        K = np.dot(PHT, np.linalg.inv(S))
        # Residual
        # This will be the last column of the rotation matrix when the 
        # quaternion is converted into one.
        # Quaternions
        qw = self.x[State.QW, 0]
        qx = self.x[State.QX, 0]
        qy = self.x[State.QY, 0]
        qz = self.x[State.QZ, 0]
        estimatedAccel = np.array([[2*(qx * qz - qy * qw)],
                                   [2*(qy * qz + qx * qw)],
                                   [1 - 2 *(qx**2 + qy**2)]])
        # estimatedAccel = np.array([[2.0 * qx * qz - 2.0 * qw * qy],
        #                            [2.0 * qy * qz + 2.0 * qx * qw],
        #                            [qz * qz + qw * qw - qx * qx - qy * qy]])
        residual = np.subtract(accel, estimatedAccel)
        correction = np.dot(K, residual)
        self.x = self.x + correction
        # Update covariance
        I_KH = np.eye(State.Size) - np.dot(K, H)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(K, self.R).dot(K.T)
        self.normalizeQuaternion()
    
    def computeMeasurementJacobianMatrix(self):
        """
        Compute jacobian matrix for the measurement model
        N2B * [0, 0, 1]
        The gravity is considered to be 1g as the Accelerometer measurement is already
        normalized.
        """
        # Quaternions
        qw = self.x[State.QW, 0]
        qx = self.x[State.QX, 0]
        qy = self.x[State.QY, 0]
        qz = self.x[State.QZ, 0]
        return np.array([[-2*qy, 2*qz, -2*qw, 2*qx, 0, 0, 0],
                            [2*qx, 2*qw, 2*qz, 2*qy, 0, 0, 0], 
                            [0, -4*qx, -4*qy, 0, 0, 0, 0]])
    
    def computeStateTransitionMatrix(self, gyro, dt):
        """
        Compute the state transition matrix F. Here the matrix is discretized
        using the Van-Loan method.
        """
        # Calculate the discretized state transition matrix
        # For simplicity
        gx = gyro[Input.GX, 0]
        gy = gyro[Input.GY, 0]
        gz = gyro[Input.GZ, 0]
        # Biases
        bwx = self.x[State.BX, 0]
        bwy = self.x[State.BY, 0]
        bwz = self.x[State.BZ, 0]
        # Quaternions
        qw = self.x[State.QW, 0]
        qx = self.x[State.QX, 0]
        qy = self.x[State.QY, 0]
        qz = self.x[State.QZ, 0]
        # Fill up the transition matrix
        F = np.array(
        [[0, 0.5*bwx - 0.5*gx, 0.5*bwy - 0.5*gy, 0.5*bwz - 0.5*gz, 0.5*qx, 0.5*qy, 0.5*qz],
        [-0.5*bwx + 0.5*gx, 0, 0.5*bwz - 0.5*gz, -0.5*bwy + 0.5*gy, -0.5*qw, -0.5*qz, 0.5*qy],
        [-0.5*bwy + 0.5*gy, -0.5*bwz + 0.5*gz, 0, 0.5*bwx - 0.5*gx, 0.5*qz, -0.5*qw, -0.5*qx],
        [-0.5*bwz + 0.5*gz, 0.5*bwy - 0.5*gy, -0.5*bwx + 0.5*gx, 0, -0.5*qy, 0.5*qx, -0.5*qw],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
        # Calculate the discretized transition matrix from this.
        # Discretize
        AA = np.zeros(shape=(State.Size*2, State.Size*2))
        AA[0:State.Size, 0:State.Size] = -F
        AA[0:State.Size, State.Size:] = self.Q
        AA[State.Size:, State.Size:] = F.T
        # Exponential
        AA_dt = AA * dt
        AA_dt_sq = np.dot(AA_dt, AA_dt)
        AA_dt_cu = np.dot(AA_dt_sq, AA_dt)
        AA_dt_qu = np.dot(AA_dt_cu, AA_dt)
        BB = np.eye(14) + AA_dt + 0.5 * AA_dt_sq + (1/6.0) * AA_dt_cu 
        + (1/np.math.factorial(4)) * AA_dt_qu
        self.F = BB[State.Size:, State.Size:].T
        Q = np.dot(self.F, BB[0:State.Size, State.Size:])
        self.P = np.dot(self.F, self.P).dot(self.F.T) + Q
    
    def rotateImuToPoseCoordSystem(self, imuMsg:Imu):
        # Rotate gyro measurements
        gx = imuMsg.angular_velocity.x
        gy = imuMsg.angular_velocity.y
        gz = imuMsg.angular_velocity.z
        # Gyro matrix
        gyroMat = np.array([[gx], [gy], [gz]])
        # rotatedGyro = np.dot(self.imuRotation, gyroMat)
        rotatedGyro = gyroMat
        # Accelerometer measurements
        ax = imuMsg.linear_acceleration.x
        ay = imuMsg.linear_acceleration.y
        az = imuMsg.linear_acceleration.z
        accelMat = np.array([[ax], [ay], [az]])
        # rotatedAccel = np.dot(self.imuRotation, accelMat/np.linalg.norm(accelMat))
        rotatedAccel = accelMat / np.linalg.norm(accelMat)
        return rotatedGyro, rotatedAccel
    
    def omImuMessageReceived(self, imuMsg:Imu):
        if(self.timeStamp):
            # print(imuMsg.header.stamp.to_sec())
            currentTimestamp = imuMsg.header.stamp.to_sec()
            deltat = currentTimestamp - self.timeStamp
            gyro, accel = self.rotateImuToPoseCoordSystem(imuMsg)
            self.predict(gyro=gyro, dt=deltat)
            self.timeStamp = currentTimestamp
            self.update(accel)
        else:
            # Initialize our state with the first incoming message of accelerometer
            self.timeStamp = imuMsg.header.stamp.to_sec()
        self.publishVisualizationMarker()
    
    def onMagdwickOutputReceived(self, imuData:Imu):
        """
        Publish a cube and pose marker to visualize the magdwick algorithm
        result
        """
        # Visualization marker
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # Position
        marker.pose.position.x = 0.
        marker.pose.position.y = 0.
        marker.pose.position.z = 0.
        # Orientation
        flip = Quaternion([0., 0., 0., 1.])
        # markerOrient = flip * Quaternion(self.x[0:4, 0])
        marker.pose.orientation.w = imuData.orientation.w
        marker.pose.orientation.x = imuData.orientation.x
        marker.pose.orientation.y = imuData.orientation.y
        marker.pose.orientation.z = imuData.orientation.z
        marker.scale.x = 0.7
        marker.scale.y = 1.
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # Publish the axes marker
        self.cubeMagdwickPublisher.publish(marker)
        # Pose 
        poseMsg = PoseStamped()
        poseMsg.header.frame_id = "odom"
        poseMsg.pose.position.x = 0.
        poseMsg.pose.position.y = 0.
        poseMsg.pose.position.z = 0.
        # Orientation
        poseMsg.pose.orientation.w = imuData.orientation.w
        poseMsg.pose.orientation.x = imuData.orientation.x
        poseMsg.pose.orientation.y = imuData.orientation.y
        poseMsg.pose.orientation.z = imuData.orientation.z
        self.magdwickOutPublisher.publish(poseMsg)
    
    def publishVisualizationMarker(self):
        # Visualization marker
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # Position
        marker.pose.position.x = 1.
        marker.pose.position.y = 1.
        marker.pose.position.z = 1.
        # Orientation
        flip = Quaternion([0., 0., 0., 1.])
        # markerOrient = flip * Quaternion(self.x[0:4, 0])
        markerOrient = Quaternion(self.x[0:4, 0])
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
        poseMsg.header.frame_id = "odom"
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
    ekf = PixhawkEKFBias()
