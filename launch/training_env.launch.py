import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # 引数の定義 (デフォルトは circuit2.world)
    world_arg = DeclareLaunchArgument(
        'world', default_value='circuit2.world',
        description='Name of the world file to load'
    )
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world')

    # パッケージパスの取得
    pkg_active_slam = get_package_share_directory('active_slam_rl')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_slam_toolbox = get_package_share_directory('slam_toolbox')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # ワールドファイルのフルパス
    world_path = PathJoinSubstitution([pkg_active_slam, 'worlds', world_name])

    # モデルパスの設定 (GAZEBO_MODEL_PATH にこのパッケージの models を追加)
    # これがないと model://Circuit_ql_2 が読み込めない
    models_path = os.path.join(pkg_active_slam, 'models')
    
    # 既存のパスがあれば追加、なければ新規設定
    if 'GAZEBO_MODEL_PATH' in os.environ:
        model_path_env = os.environ['GAZEBO_MODEL_PATH'] + ':' + models_path
    else:
        model_path_env = models_path

    env_var = SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path_env)

    # 1. Gazebo Server (gzserver) の起動
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # 2. Gazebo Client (gzclient) の起動
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # 3. Robot State Publisher の起動
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 4. TurtleBot3 のスポーン (Gazeboにロボットを出現させる)
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '0.0', # Circuit2に合わせて初期位置を調整 (必要なら)
            'y_pose': '0.0'
        }.items()
    )

    # 5. SLAM Toolbox起動
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_slam_toolbox, 'launch', 'online_async_launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 6. 報酬計算ノード起動
    reward_node = Node(
        package='active_slam_rl',
        executable='uncertainty_reward_node',
        name='uncertainty_reward_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        world_arg,
        env_var,
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        slam_launch,
        reward_node,
    ])