import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PipelineHearbeatSubscriber(Node):
    def __init__(self):
        super().__init__('heartbeat_subscriber')

        # Subscribe to the same topic name pipeline_status
        self.subscription = self.create_subscription(String,'pipeline_status',self.listener_callback,10)
        self.subscription # Prevent Unused variable warning

    def listener_callback(self,msg):
        self.get_logger().info(f'Received telemetry: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node=PipelineHearbeatSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()