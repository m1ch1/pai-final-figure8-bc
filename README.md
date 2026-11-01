# カメラ画像のみを用いた行動模倣による8の字走行(PAI最終課題)

ROS 2 Humble + Gazebo の差動二輪ロボット(Happy Mini)を、
カメラ画像だけを入力とする行動模倣(3クラス分類のCNN)で8の字走行させる。

## 環境
- Docker イメージ: airobotbook/ros2-desktop-ai-robot-book-humble
- ワールド: turtlebot3_world

## ファイル
| ファイル | 役割 |
| --- | --- |
| record_demos.sh | デモの記録(1周ごとに rosbag) |
| bag_to_dataset.py | rosbag → 画像とラベル(npz) |
| train.py | 学習(policy.pt を出力) |
| my_robot_pkg/ | ROS 2 パッケージ(bc_drive: 推論ノード) |
| eval_loop.sh | 評価用の走行ループ |
| policy.pt | 学習済みモデル |

## 実行手順
1. `ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`
2. デモ記録: `./record_demos.sh`(teleop_twist_keyboard で操縦)
3. `python3 bag_to_dataset.py --bags ~/demos --out ~/dataset`
4. `python3 train.py --data ~/dataset`
5. `ros2 run my_robot_pkg bc_drive`
