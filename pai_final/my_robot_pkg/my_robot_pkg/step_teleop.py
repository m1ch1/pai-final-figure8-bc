"""
ステップ操縦＋記録ノード
  i: 前進(1ステップ)   j: 左90°   l: 右90°
  r: このエピソードを保存して初期位置へ   d: 破棄して初期位置へ   q: 終了(途中のエピソードは破棄)

キーを押した瞬間のカメラ画像と、選んだ行動を1組として保存する。
保存先: ~/demos_step/ep_000/0000.png ... と labels.csv (file, action, x, y, yaw)
"""
import csv
import math
import os
import shutil
import sys
import termios
import threading
import time
import tty

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

HELP = """
==================================================
 i: 前進(1ステップ)   j: 左90°   l: 右90°
 r: 保存して初期位置へ   d: 破棄して初期位置へ
 q: 終了(途中のエピソードは破棄)
==================================================
"""


def get_key():
    """Enter なしで1文字読む"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def flush_keys():
    """動作中に押されたキーを捨てる"""
    termios.tcflush(sys.stdin, termios.TCIFLUSH)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a, b):
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class StepTeleop(Node):
    def __init__(self):
        super().__init__('step_teleop')
        self.declare_parameter('step_dist', 0.25)   # 1ステップの前進距離 [m]
        self.declare_parameter('lin_speed', 0.2)    # 前進速度の上限 [m/s]
        self.declare_parameter('ang_speed', 0.6)    # 旋回速度の上限 [rad/s]
        self.declare_parameter('save_dir', os.path.expanduser('~/demos_step'))
        self.step_dist = self.get_parameter('step_dist').value
        self.lin_speed = self.get_parameter('lin_speed').value
        self.ang_speed = self.get_parameter('ang_speed').value
        self.save_dir = self.get_parameter('save_dir').value

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb,
                                 qos_profile_sensor_data)
        self.reset_cli = self.create_client(Empty, '/reset_world')

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.pose = None    # (x, y, yaw)
        self.image = None   # 最新の画像メッセージ
        self.ep_dir = None
        self.rows = []

    # ---------- コールバック(別スレッドの spin から呼ばれる) ----------
    def odom_cb(self, msg):
        p = msg.pose.pose
        with self.lock:
            self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def image_cb(self, msg):
        with self.lock:
            self.image = msg

    def get_pose(self):
        with self.lock:
            return self.pose

    def get_image(self):
        with self.lock:
            return self.image

    # ---------- 動作 ----------
    def send(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def stop(self):
        self.send(0.0, 0.0)
        time.sleep(0.5)  # 止まりきって、新しい画像が届くのを待つ

    def forward(self):
        x0, y0, _ = self.get_pose()
        t_end = time.time() + 5.0 * self.step_dist / self.lin_speed + 2.0  # ぶつかった時の保険
        while rclpy.ok() and time.time() < t_end:
            x, y, _ = self.get_pose()
            remain = self.step_dist - math.hypot(x - x0, y - y0)
            if remain <= 0.005:
                break
            self.send(max(0.05, min(self.lin_speed, 1.5 * remain)), 0.0)  # 目標の手前で減速
            time.sleep(0.02)
        else:
            print('  ※ 時間切れ(ぶつかった可能性)')
        self.stop()

    def turn(self, direction):  # +1: 左, -1: 右
        _, _, yaw0 = self.get_pose()
        target = math.pi / 2
        t_end = time.time() + 5.0 * target / self.ang_speed + 2.0
        while rclpy.ok() and time.time() < t_end:
            _, _, yaw = self.get_pose()
            remain = target - abs(angle_diff(yaw, yaw0))
            if remain <= math.radians(1.0):
                break
            self.send(0.0, direction * max(0.1, min(self.ang_speed, 2.0 * remain)))
            time.sleep(0.02)
        else:
            print('  ※ 時間切れ(ぶつかった可能性)')
        self.stop()

    def reset_world(self):
        self.send(0.0, 0.0)
        if not self.reset_cli.wait_for_service(timeout_sec=3.0):
            print('  ※ /reset_world が見つかりません')
            return
        fut = self.reset_cli.call_async(Empty.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.05)
        time.sleep(1.0)  # 初期位置の画像と odom が届くのを待つ

    # ---------- 記録 ----------
    def new_episode(self):
        os.makedirs(self.save_dir, exist_ok=True)
        i = 0
        while os.path.exists(os.path.join(self.save_dir, f'ep_{i:03d}')):
            i += 1
        self.ep_dir = os.path.join(self.save_dir, f'ep_{i:03d}')
        os.makedirs(self.ep_dir)
        self.rows = []
        print(f'\n--- {os.path.basename(self.ep_dir)} 開始 ---')

    def record(self, action):
        msg, pose = self.get_image(), self.get_pose()
        if msg is None or pose is None:
            print('  ※ 画像か odom がまだ届いていません')
            return False
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        fname = f'{len(self.rows):04d}.png'
        cv2.imwrite(os.path.join(self.ep_dir, fname), img)
        self.rows.append([fname, action, f'{pose[0]:.3f}', f'{pose[1]:.3f}', f'{pose[2]:.3f}'])
        return True

    def save_episode(self):
        name = os.path.basename(self.ep_dir)
        if not self.rows:
            shutil.rmtree(self.ep_dir)
            print(f'{name}: 空なので保存しません')
            return
        with open(os.path.join(self.ep_dir, 'labels.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'action', 'x', 'y', 'yaw'])
            w.writerows(self.rows)
        print(f'{name}: 保存しました({len(self.rows)} ステップ)')

    def discard_episode(self):
        shutil.rmtree(self.ep_dir, ignore_errors=True)
        print(f'{os.path.basename(self.ep_dir)}: 破棄しました')


def main():
    rclpy.init()
    node = StepTeleop()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    print('画像と odom を待っています…')
    while rclpy.ok() and (node.get_pose() is None or node.get_image() is None):
        time.sleep(0.1)
    print(HELP)
    node.new_episode()

    actions = {'i': 'forward', 'j': 'left', 'l': 'right'}
    try:
        while rclpy.ok():
            key = get_key()
            if key in actions:
                a = actions[key]
                if node.record(a):          # 動く前の画像と行動を保存
                    if a == 'forward':
                        node.forward()
                    elif a == 'left':
                        node.turn(+1)
                    else:
                        node.turn(-1)
                    print(f'  {len(node.rows):3d}: {a}')
                flush_keys()
            elif key == 'r':
                node.save_episode()
                node.reset_world()
                node.new_episode()
                flush_keys()
            elif key == 'd':
                node.discard_episode()
                node.reset_world()
                node.new_episode()
                flush_keys()
            elif key in ('q', '\x03'):     # q または Ctrl+C
                node.discard_episode()
                break
    finally:
        node.send(0.0, 0.0)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
