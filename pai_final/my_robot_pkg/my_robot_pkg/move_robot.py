import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        topic = self.get_parameter('cmd_vel_topic').value

        self.pub = self.create_publisher(Twist, topic, 10)
        self.timer = self.create_timer(1.0, self.on_timer)  # 10Hz
        self.start = self.get_clock().now()

    def on_timer(self):
        t = (self.get_clock().now() - self.start).nanoseconds / 1e9
        msg = Twist()

        if t < 3.0:            # 3秒直進
            msg.linear.x = 0.2
        elif t < 6.0:          # 3秒その場旋回
            msg.angular.z = 0.5
        else:                  # 停止
            self.get_logger().info('done')
            self.pub.publish(Twist())  # 速度ゼロ
            self.timer.cancel()
            return

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MoveRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())  # 終了時に止める
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
