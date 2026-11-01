#!/bin/bash
# 評価用の走行ループ(学習した方策 policy.pt で自律走行させる)
#
# 使い方:
#   1. ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
#   2. cd ~/ros2_ws && source install/setup.bash && ./eval_loop.sh
#
# 操作(この端末を選択して押す):
#   Enter(開始前) : 初期位置に戻して走行開始
#   Enter(走行中) : 推論ノードを止め、ロボットを止めて初期位置へ
#   q(開始前)     : 終了
#
# 平滑化の強さなどを変えるときは、BC_ARGS を書き換える(例: "--ros-args -p smooth:=5")

set -m

BC=~/ros2_ws/install/my_robot_pkg/lib/my_robot_pkg/bc_drive
BC_ARGS=""

n=1
while true; do
  read -n 1 -s -p "[走行 $n] Enter=開始 / q=終了" k; echo
  [ "$k" = "q" ] && break
  ros2 service call /reset_world std_srvs/srv/Empty > /dev/null

  $BC $BC_ARGS < /dev/null &
  pid=$!
  read -n 1 -s -p "[走行 $n] 走行中… Enter で停止して初期位置へ" k; echo

  kill -INT $pid; wait $pid
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null
  ros2 service call /reset_world std_srvs/srv/Empty > /dev/null
  n=$((n+1))
done
