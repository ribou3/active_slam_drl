from setuptools import setup
import os
from glob import glob

package_name = 'active_slam_rl'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
]

for root, dirs, files in os.walk('models'):
    if files:
        install_path = os.path.join('share', package_name, root)
        source_paths = [os.path.join(root, f) for f in files]
        data_files.append((install_path, source_paths))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='Active SLAM with D3QN and Frontier Exploration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 既存のコマンド
            'train_slam = active_slam_rl.ros2_train_d3qn:main',
            'train_no_slam = active_slam_rl.ros2_train_no_slam:main',
            'uncertainty_reward_node = active_slam_rl.uncertainty_reward_node:main',
            'uncertainty_node = active_slam_rl.uncertainty_reward_node:main',
            
            # ★新規追加: Frontier版 Active SLAM 用コマンド
            'train_frontier = active_slam_rl.ros2_train_frontier:main',
            'active_slam_reward_node = active_slam_rl.active_slam_reward_node:main',
        ],
    },
)