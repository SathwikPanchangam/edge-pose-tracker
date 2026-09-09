import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np


class PerformanceEvaluatorNode(Node):
    def __init__(self):
        super().__init__('performance_evaluator')

        self.sub_raw = self.create_subscription(
            PoseStamped,
            '/tracker/pose_raw',
            self.raw_callback,
            10
        )
        self.sub_filtered = self.create_subscription(
            PoseStamped,
            '/tracker/pose_filtered',
            self.filtered_callback,
            10
        )

        self.raw_positions = []
        self.filtered_positions = []
        self.latencies_ms = []

        # Periodic statistics report every 5 seconds
        self.report_timer = self.create_timer(5.0, self.publish_report)
        self.get_logger().info('Performance Evaluator active. Collecting metrics...')

    def raw_callback(self, msg):
        now = self.get_clock().now()
        # Compute transport/inference latency from sensor timestamp to evaluator
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        latency = (now - msg_time).nanoseconds * 1e-6
        if latency > 0:
            self.latencies_ms.append(latency)

        self.raw_positions.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def filtered_callback(self, msg):
        self.filtered_positions.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def publish_report(self):
        if len(self.raw_positions) < 30 or len(self.filtered_positions) < 30:
            self.get_logger().info('Collecting telemetry samples...')
            return

        raw_arr = np.array(self.raw_positions[-100:])
        filt_arr = np.array(self.filtered_positions[-100:])

        # Positional variance along 3D Euclidean space
        var_raw = np.sum(np.var(raw_arr, axis=0))
        var_filt = np.sum(np.var(filt_arr, axis=0))

        variance_reduction = 0.0
        if var_raw > 1e-9:
            variance_reduction = max(0.0, (1.0 - (var_filt / var_raw)) * 100.0)

        mean_latency = np.mean(self.latencies_ms[-100:]) if self.latencies_ms else 0.0
        max_latency = np.max(self.latencies_ms[-100:]) if self.latencies_ms else 0.0

        self.get_logger().info(
            f'\n========== BENCHMARK METRICS (Last 100 Frames) ==========\n'
            f'• Samples Evaluated   : {len(raw_arr)}\n'
            f'• Raw 3D Variance     : {var_raw * 1e6:.2f} mm²\n'
            f'• Filtered 3D Variance: {var_filt * 1e6:.2f} mm²\n'
            f'• Jitter Reduction    : {variance_reduction:.1f} %\n'
            f'• Pipeline Latency    : Mean = {mean_latency:.2f} ms | Max = {max_latency:.2f} ms\n'
            f'========================================================'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()