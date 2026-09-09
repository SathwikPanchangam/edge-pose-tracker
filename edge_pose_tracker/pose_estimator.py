import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
import tf2_ros
# pyrefly: ignore [missing-import]
import cv2
import numpy as np


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Declare parameters with fallbacks
        self.declare_parameter('marker_size', 0.05)
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('target_frame', 'target_object_frame')
        self.declare_parameter('fx', 600.0)
        self.declare_parameter('fy', 600.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # Ingest parameters
        self.marker_size = self.get_parameter('marker_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value

        # Construct intrinsic matrix dynamically
        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/tracker/pose_raw', 10)
        self.cov_pub = self.create_publisher(PoseWithCovarianceStamped, '/tracker/pose_with_covariance', 10)
        self.debug_pub = self.create_publisher(Image, '/tracker/image_debug', 10)

        # ArUco dictionary initialization
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_legacy_detector = True
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_legacy_detector = False

        half_l = self.marker_size / 2.0
        self.obj_points = np.array([
            [-half_l,  half_l, 0.0],
            [ half_l,  half_l, 0.0],
            [ half_l, -half_l, 0.0],
            [-half_l, -half_l, 0.0]
        ], dtype=np.float64)

        self.get_logger().info(
            f'Pose Estimator active | Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}'
        )

    def compute_reprojection_error(self, rvec, tvec, img_points):
        proj_points, _ = cv2.projectPoints(
            self.obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        proj_points = proj_points.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(img_points - proj_points, axis=1)))

    def generate_covariance(self, z_depth, reproj_error):
        cov = [0.0] * 36
        residual_scale = max(1.0, reproj_error)

        var_xy = ((0.005 * z_depth) * residual_scale) ** 2
        var_z = ((0.02 * (z_depth ** 2)) * residual_scale) ** 2
        var_rot = (np.radians(2.0) * residual_scale) ** 2

        cov[0] = float(var_xy)
        cov[7] = float(var_xy)
        cov[14] = float(var_z)
        cov[21] = float(var_rot)
        cov[28] = float(var_rot)
        cov[35] = float(var_rot)
        return cov

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
                if hasattr(cv2, 'drawFrameAxes'):
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                elif hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

                rmat, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rmat)

                z_depth = float(tvec[2][0])
                reproj_error = self.compute_reprojection_error(rvec, tvec, img_points)
                covariance = self.generate_covariance(z_depth, reproj_error)

                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = self.camera_frame
                pose_msg.pose.position.x = float(tvec[0][0])
                pose_msg.pose.position.y = float(tvec[1][0])
                pose_msg.pose.position.z = z_depth
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw
                self.pose_pub.publish(pose_msg)

                cov_msg = PoseWithCovarianceStamped()
                cov_msg.header = pose_msg.header
                cov_msg.pose.pose = pose_msg.pose
                cov_msg.pose.covariance = covariance
                self.cov_pub.publish(cov_msg)

                t = TransformStamped()
                t.header.stamp = msg.header.stamp
                t.header.frame_id = self.camera_frame
                t.child_frame_id = self.target_frame
                t.transform.translation.x = float(tvec[0][0])
                t.transform.translation.y = float(tvec[1][0])
                t.transform.translation.z = z_depth
                t.transform.rotation = pose_msg.pose.orientation
                self.tf_broadcaster.sendTransform(t)

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
