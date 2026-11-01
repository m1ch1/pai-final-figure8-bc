#!/usr/bin/env python3
"""
~/demos/ep_*/ の rosbag を、学習用の (画像, ラベル) に変換する。

ラベル(各画像の直前に送られていた /cmd_vel で決める)
  0: forward  (linear.x > 0, angular.z = 0)   … i
  1: left     (linear.x = 0, angular.z > 0)   … j
  2: right    (linear.x = 0, angular.z < 0)   … l
  上記以外(停止中 k、u/o などの弧、後退、最初のキーを押す前)は除外する。

出力: <out>/ep_N.npz  (1エピソード = 1ファイル)
  images : uint8 (N, H, W, 3)  RGB、縮小済み
  labels : int64 (N,)
  stamp  : float64 (N,)  画像のシミュレーション時刻 [s]
  pose   : float64 (N, 3)  その時点の odom (x, y, yaw)
<out>/preview.png : 各クラスの画像の例(縮小後の見え方の確認用)

使い方(ROS 2 を source した端末で):
  python3 bag_to_dataset.py --bags ~/demos --out ~/dataset --width 64 --height 48
"""
import argparse
import glob
import math
import os
import re

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CLASSES = ['forward', 'left', 'right']
EPS = 1e-3


def label_from_twist(tw):
    v, w = tw.linear.x, tw.angular.z
    other = abs(tw.linear.y) + abs(tw.linear.z) + abs(tw.angular.x) + abs(tw.angular.y)
    if other > EPS:
        return None
    if v > EPS and abs(w) < EPS:
        return 0
    if abs(v) < EPS and w > EPS:
        return 1
    if abs(v) < EPS and w < -EPS:
        return 2
    return None  # 停止・弧・後退など


def image_to_rgb(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        img = img[:, :, ::-1]
    elif msg.encoding != 'rgb8':
        raise ValueError(f'未対応のエンコーディング: {msg.encoding}')
    return img


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def convert_episode(bag_dir, width, height):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                            output_serialization_format='cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    cur_label = None          # 直前の /cmd_vel から決めたラベル
    cur_pose = None           # 直前の /odom
    images, labels, stamps, poses = [], [], [], []
    skipped = 0

    while reader.has_next():   # メッセージは記録時刻の順に出てくる
        topic, data, _ = reader.read_next()
        if topic == '/cmd_vel':
            cur_label = label_from_twist(deserialize_message(data, types[topic]))
        elif topic == '/odom':
            p = deserialize_message(data, types[topic]).pose.pose
            cur_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))
        elif topic == '/camera/image_raw':
            if cur_label is None or cur_pose is None:
                skipped += 1
                continue
            msg = deserialize_message(data, types[topic])
            img = cv2.resize(image_to_rgb(msg), (width, height), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(cur_label)
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            poses.append(cur_pose)

    return (np.array(images, dtype=np.uint8).reshape(-1, height, width, 3),
            np.array(labels, dtype=np.int64),
            np.array(stamps, dtype=np.float64),
            np.array(poses, dtype=np.float64).reshape(-1, 3),
            skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bags', default=os.path.expanduser('~/demos'))
    ap.add_argument('--out', default=os.path.expanduser('~/dataset'))
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=48)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ep_dirs = [d for d in glob.glob(os.path.join(args.bags, 'ep_*')) if os.path.isdir(d)]
    ep_dirs.sort(key=lambda d: int(re.search(r'ep_(\d+)', d).group(1)))  # ep_2 < ep_10

    total = np.zeros(3, dtype=int)
    examples = {}
    for d in ep_dirs:
        name = os.path.basename(d)
        try:
            images, labels, stamps, poses, skipped = convert_episode(d, args.width, args.height)
        except Exception as e:  # 壊れた bag は飛ばす
            print(f'{name}: 失敗 ({e})')
            continue
        np.savez_compressed(os.path.join(args.out, f'{name}.npz'),
                            images=images, labels=labels, stamp=stamps, pose=poses)
        counts = np.bincount(labels, minlength=3)
        total += counts
        dur = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
        print(f'{name}: {len(labels):5d} 枚 (除外 {skipped:4d})  '
              + '  '.join(f'{c}={n}' for c, n in zip(CLASSES, counts))
              + f'  sim {dur:.1f} s')
        for k in range(3):
            if k not in examples and counts[k] > 0:
                examples[k] = images[np.where(labels == k)[0][len(np.where(labels == k)[0]) // 2]]

    print('-' * 60)
    print(f'合計 {total.sum()} 枚  ' + '  '.join(
        f'{c}={n} ({n / max(total.sum(), 1):.0%})' for c, n in zip(CLASSES, total)))

    if examples:  # 縮小後の見え方を確認する画像
        tiles = []
        for k in range(3):
            tile = examples.get(k, np.zeros((args.height, args.width, 3), np.uint8))
            tile = cv2.resize(tile, (args.width * 4, args.height * 4),
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, CLASSES[k], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)
            tiles.append(tile)
        cv2.imwrite(os.path.join(args.out, 'preview.png'),
                    cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
        print(f'確認用画像: {os.path.join(args.out, "preview.png")}')


if __name__ == '__main__':
    main()
