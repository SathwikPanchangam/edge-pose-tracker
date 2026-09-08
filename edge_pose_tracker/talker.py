import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from std_srvs.srv import SetBool


class ConfigurableHeartbeatPublisher(Node):
    def __init__(self):
        super().__init__('heartbeat_publisher')
        
        # 1. Declare parameters with default values
        self.declare_parameter('publish_rate',1.0) 
        self.declare_parameter('frame_id','camera_link')
        self.declare_parameter('device_id',0)

        # 2. Read delared parameter values
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.device_id = self.get_parameter('device_id').value

        # Internal operational state
        self.is_active = True

        self.get_logger().info(f'Initilized with Frame ID: "{self.frame_id}", Device ID: {self.device_id}, Rate: {self.publish_rate} Hz')

        # 3. Create publisher and timer based on parameter rate
        self.publisher_ = self.create_publisher(String,'pipeline_status',10)
        self.timer_period = 1.0/self.publish_rate
        self.timer = self.create_timer(self.timer_period,self.timer_callback)
        self.sequence_id = 0

        # 4. Register a dynamic callback to handle runtime changes
        self.add_on_set_parameters_callback(self.parameters_callback)

        # 5. Service Server: toggles telemetry on/off
        self.srv = self.create_service(
            SetBool,
            'toggle_pipeline',
            self.toggle_pipeline_callback
        )

    def toggle_pipeline_callback(self, request, response):
        self.is_active = request.data
        response.success = True
        if self.is_active:
            response.message = "Pipeline broadcast resumed (Active)"
            self.get_logger().info("Service Invoked: Resuming telemetry broadcast.")
        else:
            response.message = "Pipeline broadcast paused (Standby)"
            self.get_logger().warn("Service Invoked: Pausing telemetry broadcast.")
        return response

    def parameters_callback(self,params):
        for param in params:
            if param.name == 'publish_rate':
                if param.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    new_rate = float(param.value)
                    if new_rate <= 0.0:
                        self.get_logger().error('Publish rate must be greater than 0!')
                        return SetParametersResult(sucessful=False, reason='Rate must be > 0')
                    
                    self.publish_rate = new_rate
                    self.timer_period = 1.0 / self.publish_rate
                    
                    # Reset the running timer with the new duration
                    self.timer.cancel()
                    self.timer = self.create_timer(self.timer_period, self.timer_callback)
                    self.get_logger().warn(f'Dynamic update: Publish rate adjusted to {self.publish_rate} Hz')

            elif param.name == 'frame_id':
                self.frame_id = param.value
                self.get_logger().warn(f'Dynamic update: Target frame set to "{self.frame_id}"')

        return SetParametersResult(successful=True)


    def timer_callback(self):
        # Only publish if operational state is active
        if not self.is_active:
            return

        msg = String()
        msg.data = f'[{self.frame_id}] Heartbeat Seq: {self.sequence_id} (Rate: {self.publish_rate}Hz)'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.sequence_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableHeartbeatPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

