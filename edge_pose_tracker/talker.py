from typing import final
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PipelineHeartbeatPublisher(Node):
    def __init__(self):
        super().__init__('heartbeat_publisher')
        # Create a publisher: message type Sting, topic name 'pipeline_status' queue size 10
        self.publisher_ = self.create_publisher(String, 'pipeline_status', 10)

        # Timer: Fires every 1.0 second (1Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.sequence_id = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Pipeline ACTIVE | Seq: {self.sequence_id}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Broadcasting: "{msg.data}"')
        self.sequence_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = PipelineHeartbeatPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

