import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# pyrefly: ignore [missing-import]
import cv2
import numpy as np


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Declare configurable parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        self.device_id = self.get_parameter('device_id').value
        self.framerate = self.get_parameter('framerate').value
        self.frame_id = self.get_parameter('frame_id').value

        # Initialize bridge and publisher
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        # Attempt hardware camera connection
        self.cap = cv2.VideoCapture(self.device_id)
        self.use_synthetic = False

        if not self.cap.isOpened():
            self.get_logger().warn(
                f'Hardware camera at /dev/video{self.device_id} unavailable. '
                f'Switching to synthetic pattern generator.'
            )
            self.use_synthetic = True
        else:
            self.get_logger().info(f'Hardware camera connected on index {self.device_id}')

        # Setup frame capture timer
        timer_period = 1.0 / self.framerate
        self.timer = self.create_timer(timer_period, self.capture_and_publish)
        self.frame_counter = 0

    def generate_synthetic_frame(self):
        """Generates a dynamic 640x480 test grid if no camera hardware is present."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background gradient
        img[:, :] = (30, 30, 30)

        # Draw coordinate crosshair
        cv2.line(img, (320, 0), (320, 480), (70, 70, 70), 1)
        cv2.line(img, (0, 240), (640, 240), (70, 70, 70), 1)

        # Dynamic circle to visually demonstrate frame changes
        x = int(320 + 200 * np.sin(self.frame_counter * 0.05))
        y = int(240 + 100 * np.cos(self.frame_counter * 0.05))
        cv2.circle(img, (x, y), 25, (0, 255, 0), -1)

        # Diagnostic metadata overlay
        text = f"Synthetic Stream | Frame: {self.frame_counter}"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    def capture_and_publish(self):
        if self.use_synthetic:
            frame = self.generate_synthetic_frame()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('Failed to grab frame from hardware camera.')
                return

        # Convert OpenCV BGR image to ROS 2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        self.image_pub.publish(msg)
        self.frame_counter += 1

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main