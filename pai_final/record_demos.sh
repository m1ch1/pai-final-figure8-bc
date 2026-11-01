#!/bin/bash
# デモ記録スクリプト(1周 = 1エピソード = 1つの rosbag)
#
# 使い方:
#   1. ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
#   2. 別の端末で ros2 run teleop_twist_keyboard teleop_twist_keyboard
#      (i: 前進 / j: 左旋回 / l: 右旋回 / k: 停止)
#   3. この端末で ./record_demos.sh
#
# 操作(この端末を選択して押す):
#   Enter : この周を保存して初期位置へ
#   d     : この周を破棄して初期位置へ(同じ番号でやり直し)
#   終了   : Ctrl+C の後、pkill -INT -f "ros2 bag record"

set -m   # スクリプト内でも、裏で動く rosbag に SIGINT を届けるために必要

OUTDIR=~/demos
TOPICS="/camera/image_raw /cmd_vel /odom"

mkdir -p "$OUTDIR"
i=0
while true; do
  # 既にある番号は飛ばす
  while [ -e "$OUTDIR/ep_$i" ]; do i=$((i+1)); done

  # < /dev/null: rosbag がキー入力を読もうとして一時停止するのを防ぐ
  ros2 bag record -o "$OUTDIR/ep_$i" $TOPICS < /dev/null &
  pid=$!

  echo "ep_$i 記録中… Enter=保存 / d=やり直し"
  while true; do
    read -n 1 -s k
    [ -z "$k" ] && break        # Enter
    [ "$k" = "d" ] && break     # d
  done

  kill -INT $pid; wait $pid                                                   # 記録を閉じる
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null     # 止める
  ros2 service call /reset_world std_srvs/srv/Empty > /dev/null               # 初期位置へ

  if [ "$k" = "d" ]; then
    rm -rf "$OUTDIR/ep_$i"
    echo "ep_$i を消しました"
  else
    echo "ep_$i を保存しました"
  fi
done
