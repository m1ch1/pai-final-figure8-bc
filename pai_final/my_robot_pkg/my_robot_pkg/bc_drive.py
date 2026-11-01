"""
学習した方策(policy.pt)でロボットを走らせる推論ノード

  カメラ画像 → 縮小(学習時と同じ) → モデル → 前進/左/右 → /cmd_vel
  速度は記録時と同じ値(前進 0.3281 m/s、旋回 0.6561 rad/s)を使う。
  直近 smooth 回の確率を平均して、判断がぱたぱた切り替わるのを抑える。

起動:
  ros2 run my_robot_pkg bc_drive
  ros2 run my_robot_pkg bc_drive --ros-args -p smooth:=1   # 平滑化なし
"""
import collections
import os

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

CLASSES = ['forward', 'left', 'right']


def image_to_rgb(msg):
    """bag_to_dataset.py と同じ変換"""
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        img = img[:, :, ::-1]
    return img


class BCDrive(Node):
    def __init__(self):
        super().__init__('bc_drive')
        self.declare_parameter('model_path', os.path.expanduser('~/ros2_ws/policy.pt'))
        self.declare_parameter('linear', 0.3281)   # 記録時の speed
        self.declare_parameter('angular', 0.6561)  # 記録時の turn
        self.declare_parameter('width', 64)        # 学習時の画像サイズ
        self.declare_parameter('height', 48)
        self.declare_parameter('rate', 10.0)       # 1秒あたりの判断回数
        self.declare_parameter('smooth', 3)        # 確率を平均する回数(1 で平滑化なし)

        g = lambda name: self.get_parameter(name).value
        self.linear, self.angular = g('linear'), g('angular')
        self.w, self.h = g('width'), g('height')

        torch.set_num_threads(2)  # Gazebo の分の CPU を残す
        self.model = torch.jit.load(g('model_path'))
        self.model.eval()
        self.probs = collections.deque(maxlen=max(1, g('smooth')))

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb,
                                 qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / g('rate'), self.on_timer)
        self.image = None
        self.last_action = None
        self.get_logger().info(f'モデル読み込み完了: {g("model_path")}  '
                               f'linear={self.linear} angular={self.angular}')

    def image_cb(self, msg):
        self.image = msg

    def on_timer(self):
        if self.image is None:
            return
        img = cv2.resize(image_to_rgb(self.image), (self.w, self.h),
                         interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # uint8 (1,H,W,3)
        with torch.no_grad():
            p = torch.softmax(self.model(x), dim=1)[0].numpy()
        self.probs.append(p)
        avg = np.mean(self.probs, axis=0)
        a = int(np.argmax(avg))

        msg = Twist()
        if a == 0:
            msg.linear.x = self.linear
        elif a == 1:
            msg.angular.z = self.angular
        else:
            msg.angular.z = -self.angular
        self.pub.publish(msg)

        if a != self.last_action:  # 行動が変わったときだけ表示(動画の説明にも使える)
            self.get_logger().info(f'{CLASSES[a]:>7s}   '
                                   f'(forward {avg[0]:.2f} / left {avg[1]:.2f} / right {avg[2]:.2f})')
            self.last_action = a


def main():
    rclpy.init()
    node = BCDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())  # Ctrl+C の後は送れないこともある
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
