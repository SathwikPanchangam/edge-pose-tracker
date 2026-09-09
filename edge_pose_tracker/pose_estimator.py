import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import tf2_ros
# pyrefly: ignore [missing-import]
import cv2
import numpy as np


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Parameters
        self.declare_parameter('marker_size', 0.05)
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('target_frame', 'target_object_frame')

        self.marker_size = self.get_parameter('marker_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value

        self.bridge = CvBridge()

        # Initialize dynamic TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Topic Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/tracker/pose_raw', 10)
        self.debug_pub = self.create_publisher(Image, '/tracker/image_debug', 10)

        # Version-adaptive ArUco dictionary initialization
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        # Version-adaptive detector parameters
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_legacy_detector = True
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_legacy_detector = False

        # Camera Intrinsics
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 3D coordinates of marker corners in object frame
        half_l = self.marker_size / 2.0
        self.obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float64)

        self.get_logger().info('Pose Estimator with TF2 Broadcaster active.')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_legacy_detector:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            img_points = corners[0].reshape((4, 2))
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Visual debug axes
                if hasattr(cv2, 'drawFrameAxes'):
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                elif hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

                rmat, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rmat)
                roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)

                # 1. Publish standard PoseStamped topic
                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = self.camera_frame
                pose_msg.pose.position.x = float(tvec[0][0])
                pose_msg.pose.position.y = float(tvec[1][0])
                pose_msg.pose.position.z = float(tvec[2][0])
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw
                self.pose_pub.publish(pose_msg)

                # 2. Broadcast Dynamic TF Transform
                t = TransformStamped()
                t.header.stamp = msg.header.stamp
                t.header.frame_id = self.camera_frame
                t.child_frame_id = self.target_frame

                t.transform.translation.x = float(tvec[0][0])
                t.transform.translation.y = float(tvec[1][0])
                t.transform.translation.z = float(tvec[2][0])
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw

                self.tf_broadcaster.sendTransform(t)

                self.get_logger().info(
                    f'TF Broadcast -> Pos [X:{tvec[0][0]:.2f} Y:{tvec[1][0]:.2f} Z:{tvec[2][0]:.2f}] | '
                    f'Rot [R:{roll:.0f}° P:{pitch:.0f}° Y:{yaw:.0f}°]'
                )

        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def rotation_matrix_to_quaternion(self, R):
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        return qx / norm, qy / norm, qz / norm, qw / norm

    def quaternion_to_euler(self, x, y, z, w):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()