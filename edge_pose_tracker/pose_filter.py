import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TransformStamped
import tf2_ros
import numpy as np


class KalmanAxisFilter:
    """1D discrete Kalman Filter tracking position and velocity."""
    def __init__(self, process_noise_q=1e-3):
        # State vector: [position, velocity]^T
        self.x = np.zeros((2, 1), dtype=np.float64)
        # State estimation error covariance
        self.P = np.eye(2, dtype=np.float64)
        # Process noise covariance
        self.Q = np.array([
            [process_noise_q * 0.25, process_noise_q * 0.5],
            [process_noise_q * 0.5,  process_noise_q]
        ], dtype=np.float64)
        # Measurement matrix: we only observe position
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.initialized = False

    def predict(self, dt):
        if not self.initialized:
            return
        # State transition matrix F
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, measurement_var):
        z = float(measurement)
        R = np.array([[max(1e-6, float(measurement_var))]], dtype=np.float64)

        if not self.initialized:
            self.x = np.array([[z], [0.0]], dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * R[0, 0]
            self.initialized = True
            return float(self.x[0, 0])

        # Innovation (residual)
        y = z - (self.H @ self.x)[0, 0]
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K * y
        # Covariance update
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return float(self.x[0, 0])


class PoseFilterNode(Node):
    def __init__(self):
        super().__init__('pose_filter')

        self.declare_parameter('process_noise_q', 0.05)
        self.declare_parameter('filtered_frame', 'target_object_filtered')
        
        q_param = self.get_parameter('process_noise_q').value
        self.filtered_frame = self.get_parameter('filtered_frame').value

        # Independent filters for X, Y, Z translation axes
        self.kf_x = KalmanAxisFilter(process_noise_q=q_param)
        self.kf_y = KalmanAxisFilter(process_noise_q=q_param)
        self.kf_z = KalmanAxisFilter(process_noise_q=q_param)

        self.last_timestamp = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.sub_cov_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/tracker/pose_with_covariance',
            self.filter_callback,
            10
        )
        self.pub_filtered_pose = self.create_publisher(
            PoseStamped,
            '/tracker/pose_filtered',
            10
        )

        self.get_logger().info('Adaptive Kalman Pose Filter initialized.')

    def filter_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Calculate time step dt
        if self.last_timestamp is None:
            dt = 0.033  # default ~30 Hz delta on first frame
        else:
            dt = max(1e-4, current_time - self.last_timestamp)
        self.last_timestamp = current_time

        # 1. Prediction step
        self.kf_x.predict(dt)
        self.kf_y.predict(dt)
        self.kf_z.predict(dt)

        # 2. Update step using dynamic measurement variances from Day 8
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_z = msg.pose.pose.position.z

        var_x = msg.pose.covariance[0]   # Index 0: Var(X)
        var_y = msg.pose.covariance[7]   # Index 7: Var(Y)
        var_z = msg.pose.covariance[14]  # Index 14: Var(Z)

        filtered_x = self.kf_x.update(raw_x, var_x)
        filtered_y = self.kf_y.update(raw_y, var_y)
        filtered_z = self.kf_z.update(raw_z, var_z)

        # 3. Publish smoothed PoseStamped
        filtered_msg = PoseStamped()
        filtered_msg.header = msg.header
        filtered_msg.pose.position.x = filtered_x
        filtered_msg.pose.position.y = filtered_y
        filtered_msg.pose.position.z = filtered_z
        # Pass orientation through
        filtered_msg.pose.orientation = msg.pose.pose.orientation

        self.pub_filtered_pose.publish(filtered_msg)

        # 4. Broadcast smoothed TF frame
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = self.filtered_frame
        t.transform.translation.x = filtered_x
        t.transform.translation.y = filtered_y
        t.transform.translation.z = filtered_z
        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PoseFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
