#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from carla_msgs.msg import CarlaEgoVehicleStatus
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from sensor_msgs.msg import Image # 센서 fov 출력용
# from warning_mode_interfaces.action import WarningMode
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import carla
import time
import math
import threading
import asyncio
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf_transformations 
import numpy as np
import cv2
from cv_bridge import CvBridge
from collections import namedtuple

LIDAR_TIMEOUT = 0.5    # 무신호 감지 임계 (초)
CHECK_PERIOD  = 0.01    # 타임아웃 검사 주기 (초)
PUBLISH_RATE  = 10.0   # 제어용 Python API 호출 주기 (Hz)
wheel_base = 2.89
# 액션 라이브러리 사용해서 behavior Tree로 부터 액션 goal을 받으면 (0, 저속운전 , 1. 갓길 이동 , 2. 차선 평행 회전 , 3. 핸드파킹)

### 위험도 파라미터 #######
K = 1.5 #P에 대한 가중치 ##
lamb = 20.273   # λ      ##
TH = 3.374              ## 100 -> 20으로 수정
########################

### 센서 FOV 정의 ####################################################################
Sensor = namedtuple('Sensor',
        'id cx cy yaw fov range')       # yaw, fov 는 rad

SENSORS = [
    Sensor('rgb_front', 2.0, 0.0, 0.0,              math.radians(90),  100),
    Sensor('lidar',     0.0, 0.0, 0.0,              math.radians(360), 50),
    Sensor('semantic_lidar', 0.0, 0.0, 0.0,         math.radians(360), 50),
    Sensor('radar_front', 2.0, 0.0, 0.0,            math.radians(30),  100),
    Sensor('seg_cam',   2.0, 0.0, 0.0,              math.radians(90),  100),
    Sensor('depth_cam', 2.0, 0.0, 0.0,              math.radians(90),  100),
]
####################################################################################
def force_all_traffic_lights_green(client):
    world = client.get_world()
    lights = world.get_actors().filter("traffic.traffic_light")

    for light in lights:
        light.set_state(carla.TrafficLightState.Green)
        light.set_green_time(9999.0)
        light.freeze(True)
        print(f"신호등 {light.id} → 초록불 고정")


def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle


class LidarFailSafe(Node):
    def __init__(self):
        super().__init__('carla_failsafe_client')
        

        # ① /lidar_alive, /risk_level 퍼블리셔 추가
        self.alive_pub = self.create_publisher(Bool, '/lidar_alive', 10)
        self.risk_pub = self.create_publisher(Float64,'/risk_level',10)
        self.thresh_pub = self.create_publisher(Float64,'/threshold',10)
        # 요청 응답 
        self.pub_w_res = self.create_publisher(Float64,'warningmode_result',  10)
        self.pub_s_res = self.create_publisher(Float64,'shouldershift_result',10)
        self.pub_fov = self.create_publisher(Image,'/sensor_fov',10)
        # 요청 받기
        self.sub_w_cmd = self.create_subscription(Float64, 'warningmode',   self.cb_warn_cmd,  10)
        self.sub_s_cmd = self.create_subscription(Float64, 'shouldershift', self.cb_shift_cmd, 10)

        self.warn_active   = False
        self.shift_active  = False
        # 경로 퍼블리시
        self.path_pub = self.create_publisher(Path, '/predicted_path',10)

        # 차량 센서들 구독
        self.create_subscription( # 라이다
            PointCloud2,
            '/carla/hero/lidar',
            self.lidar_cb,
            10)
        
        self.create_subscription(
            PointCloud2,
            '/carla/hero/semantic_lidar',
            self.semantic_lidar_cb,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/carla/hero/radar_front',
            self.radar_front_cb,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/carla/hero/depth_front/image',
            self.depth_front_cb,
            10
        )

        self.create_subscription(
            Image,
            '/carla/hero/rgb_front/image',
            self.rgb_front_cb,
            10
        )

        self.create_subscription(
            PointCloud2,
            '/carla/hero/semantic_segmentation_front/image',
            self.semantic_segmentation_front_cb,
            10
        )

        # ③ ROS: 차량 속도(Status) 구독
        self.vehicle_speed = 0.0
        self.vehicle_steering = 0.0          # ← 추가!
        self.vehicle_steering_radian = 0.0   # ← 추가!
        self.rgb_front_last_stamp = 0.0
        self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/hero/vehicle_status',
            self.status_cb,
            10)

        # CARLA Python API 연결
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        #client.load_world(self.map_name)
        self.world = self.client.get_world()
        force_all_traffic_lights_green(self.client) #강제 초록불
        self.tm = self.client.get_trafficmanager(8000)
        
        # HERO 차량 찾기 및 Autopilot 비활성
        self.hero = None
        for v in self.world.get_actors().filter('vehicle.*'):
            self.get_logger().info(f"{v.id}, {v.attributes.get('role_name')}")
            print(v.id, v.attributes.get('role_name'))
            if v.attributes.get('role_name') == 'hero':
                self.get_logger().info(f"[DEBUG] 차량 ID={v.id}, role_name={v.attributes.get('role_name')}")
                self.hero = v
                #self.hero.set_autopilot(False) #emp
                break
            # if v.attributes.get('role_name') == 'ego_vehicle':
            #     self.get_logger().info(f"[DEBUG] 차량 ID={v.id}, role_name={v.attributes.get('role_name')}")
            #     self.hero = v
            #     print(f'{k}: {v}')
            #     #self.hero.set_autopilot(False) #emp
            #     break
        if not self.hero:
            self.get_logger().error('Hero 차량을 찾을 수 없습니다!')

        # HERO 차량 찾기 및 Autopilot 비활성
        # self.hero = None
        # for v in self.world.get_actors().filter('vehicle.*'):
        #     # role_name 과 타입도 같이 찍어봐서, 정확히 어떤 값이 들어오는지 확인
        #     role = v.attributes.get('role_name', '')
        #     self.get_logger().info(f"[DEBUG] Actor ID={v.id}, type={v.type_id}, role_name='{role}'")

        #     if role == 'ego_vehicle':
        #         self.hero = v
        #         self.get_logger().info(f"[INFO] Hero vehicle found: ID={v.id}, type={v.type_id}")

        #         # Autopilot 끄기 (예외 처리도 해주면 안전)
        #         try:
        #             self.hero.set_autopilot(False)
        #             self.get_logger().info("Autopilot disabled on hero vehicle")
        #         except AttributeError:
        #             self.get_logger().warn("set_autopilot() 메서드가 없습니다")
        #         except Exception as e:
        #             self.get_logger().warn(f"Autopilot 비활성화 중 오류 발생: {e}")

        #         break

        if self.hero is None:
            self.get_logger().error("Hero 차량을 찾을 수 없습니다! role_name 또는 타입을 다시 확인하세요.")



        # 상태 변수
        self.lidar_last_stamp = time.time()
        self.in_fail    = False
        self.current_risk = 0.0
        self.has_parked = False

        # ====[수정]==== 히스테리시스(상태유지)를 위한 변수 추가
        self.RECOVERY_TIME_THRESHOLD = 2.0 # 2초 동안 신호가 안정적이어야 복구
        self.recovery_start_time = None    # 복구 시작 시간 기록
        # =============

        # 타이머 설정
        self.create_timer(CHECK_PERIOD, self.check_timeout)
        self.create_timer(1.0 / PUBLISH_RATE, self.publish_ctrl) # 모드 확인해서 실행
        self.create_timer(0.01, self.publish_risk)
        self.create_timer(1.0, self.next_line)
        self.create_timer(0.1, self.publish_th)
        self.create_timer(0.1, self.publish_fov)
        self.create_timer(0.1,self.calculate_risk)
        self.create_timer(0.5, self.generate_future_path)

        # 초기화 - waypoint와 차선 정보
        self.waypoint = None
        self.right_lane_marking = None
        self.left_lane_marking = None


    def cb_warn_cmd(self, msg: Float64):
        self.warn_active  = msg.data
        if self.warn_active == 1.0:
            self.has_parked = False
            self.get_logger().info(f"[sensor] WarningMode command={msg.data}")
        if self.warn_active == 2.0:
            self.pub_w_res.publish(Float64(data=1.0))
            self.get_logger().info("[sensor] ShoulderShift SUCCESS 전달됨")




    def cb_shift_cmd(self, msg: Float64):
        self.shift_active = msg.data
        if self.shift_active == 1.0:
            self.has_parked = False
            self.get_logger().info(f"[sensor] ShoulderShift command={msg.data}")
            # 여기서 바로 결과 보내기
        if self.shift_active == 2.0:
            self.pub_s_res.publish(Float64(data=1.0))
            self.get_logger().info("[sensor] ShoulderShift SUCCESS 전달됨")

 
    #################################################################################################################

    def define_setP(self):
        # 센서들 영역과 예상 경로 점으로 샘플링해서 유클리드 거리 계산,
        # 연관성 높은 3개 센서를 P집합으로
        pass

    def calculate_risk(self):
        # 모든 센서들 for문돌려서 마지막 타임스탬프 기준 t계산, 고장여부 L을 최종R에 합산하는 함수
        # 우선 hero 차량에 어떤 센서들 있는지 확인
        pass

    def get_lane_lotation(self):
        if self.waypoint:
            self.lane_yaw = self.waypoint.transform.rotation.yaw #차 yaw와 오른쪽 차선 yaw 일치시키기
        

    def next_line(self):
        if self.hero:
            self.waypoint = self.world.get_map().get_waypoint(self.hero.get_location(), project_to_road=True, lane_type=carla.LaneType.Any)
            self.right_lane_marking = self.waypoint.right_lane_marking
            self.left_lane_marking = self.waypoint.left_lane_marking
            self.get_logger().info(f"왼쪽 차선: {self.left_lane_marking.type}, 오른쪽 차선: {self.right_lane_marking.type}")

    def publish_th(self): # 쓰레시홀드 퍼블리시 함수
        threshold = Float64()
        threshold.data = TH
        self.thresh_pub.publish(threshold)

    def publish_fov(self): #FOV = field if view
        self.bridge = CvBridge()
        fov_lidar_front_fail = cv2.imread('/home/taewook/ros2_ws/src/sensor_toggle_node/sensor_toggle_ros2/resource/fov_lidar_front_fail.png')
        fov_front_fail = cv2.imread('/home/taewook/ros2_ws/src/sensor_toggle_node/sensor_toggle_ros2/resource/fov_front_fail.png')
        fov_lidar_fail = cv2.imread('/home/taewook/ros2_ws/src/sensor_toggle_node/sensor_toggle_ros2/resource/fov_lidar_fail.png')       
        fov_safe = cv2.imread('/home/taewook/ros2_ws/src/sensor_toggle_node/sensor_toggle_ros2/resource/fov_safe.png')

        lidar_t = time.time() - self.lidar_last_stamp #(현재 시간 - 최근 수신 시간)
        rgb_front_t = time.time() - self.rgb_front_last_stamp
        
        image = self.bridge.cv2_to_imgmsg(fov_safe, encoding="bgr8")

        if lidar_t>1.2 and rgb_front_t>1.2 :
            image = self.bridge.cv2_to_imgmsg(fov_lidar_front_fail, encoding="bgr8")
        elif lidar_t>1.2: # 10hz 감안해서 1.2초
            image = self.bridge.cv2_to_imgmsg(fov_lidar_fail, encoding="bgr8")
        elif rgb_front_t>1.2:
            image = self.bridge.cv2_to_imgmsg(fov_front_fail, encoding="bgr8")

        self.pub_fov.publish(image)



    def publish_risk(self): # 위험도 토픽 퍼블리시 함수
        risk_msg = Float64()
        risk_msg.data = self.current_risk
        self.risk_pub.publish(risk_msg)

    def lidar_cb(self, msg):
        # 라이다 메시지 수신 시점은 항상 최신으로 갱신
        self.lidar_last_stamp = time.time()

        # 만약 고장 상태(in_fail)라면, 즉시 복구하지 않고 복구 조건을 확인
        if self.in_fail:
            # 복구 타이머가 아직 시작되지 않았다면, 지금 시작
            if self.recovery_start_time is None:
                self.get_logger().info("고장 상태 중 Lidar 신호 수신 시작")
                self.recovery_start_time = time.time()

            # 복구 시작 후 충분한 시간이 지났는지 확인
            elapsed_recovery_time = time.time() - self.recovery_start_time
            if elapsed_recovery_time > self.RECOVERY_TIME_THRESHOLD:
                self.get_logger().info(f"{self.RECOVERY_TIME_THRESHOLD}초간 신호 안정. 정상 주행으로 복구합니다.")
                self.in_fail = False
                self.has_parked = False
                self.recovery_start_time = None # 타이머 리셋
                self.current_risk = 0.0 # 위험도도 0으로 초기화
                if self.hero:
                    self.hero.set_autopilot(True)
            else:
                # 아직 복구 시간이 충분하지 않으면 로그만 남기고 아무것도 안 함
                self.get_logger().info(f"복구 진행 중... ({elapsed_recovery_time:.1f}/{self.RECOVERY_TIME_THRESHOLD}초)", throttle_duration_sec=1.0)

        # 고장 상태가 아닐 때는 alive 토픽만 발행 (기존 로직과 유사)
        else:
            alive_msg = Bool()
            alive_msg.data = True
            self.alive_pub.publish(alive_msg)


    # FOV 라이다 = 시멘틱 라이다 , 

    def semantic_lidar_cb(self,msg): #
        self.semantic_lidar_last_stamp = time.time()

    def radar_front_cb(self,msg): # 
        self.radar_front_last_stamp = time.time()

    def depth_front_cb(self,msg): #
        self.depth_front_last_stamp = time.time()

    def rgb_front_cb(self,msg): #
        self.rgb_front_last_stamp = time.time()

    def semantic_segmentation_front_cb(self,msg): #
        self.semantic_segmentation_front_last_stamp = time.time()
    

    def status_cb(self, msg):
        self.vehicle_speed = msg.velocity #m/s기준
        # self.vehicle_steering= msg.control.steer # -1 ~ 1
        self.vehicle_steering_radian = math.radians(100*msg.control.steer) # msg.control.steer는 최대가 0.69 고로 100배후 라디안

    def check_timeout(self):
        t = time.time() - self.lidar_last_stamp

        if t < LIDAR_TIMEOUT:
            # (이전 코드와 동일)
            alive = True
            self.current_risk = 0.0
            if self.recovery_start_time is not None:
                self.recovery_start_time = None
        else:
            # 타임아웃 이후: 위험도 계산 시작
            alive = False
            t_delta = t - LIDAR_TIMEOUT
            raw_risk = K * math.exp(lamb * t_delta)
            self.current_risk = min(raw_risk, 1000.0)

            # ====================[디버깅 코드 추가]====================
            # 계산에 사용된 모든 변수와 결과를 매 순간 출력합니다.
            print(f"[DEBUG] t_delta={t_delta:.4f} | K={K} | lamb={lamb} | risk={self.current_risk:.4f}")
            # ==========================================================

        # 기존 로그는 잠시 주석 처리하거나 그대로 두어도 됩니다.
        # self.get_logger().info(f'현재 위험도: {self.current_risk:.2f}')
        
        alive_msg = Bool()
        alive_msg.data = alive
        self.alive_pub.publish(alive_msg)

        if not self.in_fail and self.current_risk > TH:
            self.get_logger().warn(f'위험도 초과 {self.current_risk} — 급정지 모드')
            self.in_fail = True
            self.recovery_start_time = None
            if self.hero:
                self.hero.set_autopilot(False)



    def generate_future_path(self):
        if self.hero is None:
            self.get_logger().warn(" hero 차량이 없어 경로 생성을 건너뜀")
            return
        # self.velocity = self.hero.get_velocity()
        max_delta = math.radians(30)       # ±70°
        transform = self.hero.get_transform()
        
        x = transform.location.x #월드맵 기준 차량의 좌표
        y = transform.location.y
        yaw =transform.rotation.yaw #월드 좌표계 기준으로 차량 차체가 바라보는 방향 ~도
        theta = math.radians(yaw) # ~도 -> 라디안
        delta_t = 0.1 # 0.01초 기준 샘플링
        ## 계산 시작
        path = []
        for _ in range(15): # 경로점 150개 생성 (1.5초간의 예상 경로)
            delta = max(-max_delta, min(max_delta, self.vehicle_steering_radian)) # 앞바퀴의 조향각
            # delta = 0.15 * delta
            beta = math.atan(0.5* math.tan(delta)) # beta
            ##
            # angular_velocity = self.vehicle_speed / wheel_base * math.tan(delta) * cosb #각속도 구하기

            #예상경로 
            x += self.vehicle_speed * math.cos(theta+beta) * delta_t
            y += self.vehicle_speed * math.sin(theta+beta) * delta_t
            theta += (self.vehicle_speed / (0.5 * wheel_base)) * math.sin(beta) * delta_t
            path.append((x,y,theta))

        path_msg = Path() # Path 타입 토픽 퍼블리시
        path_msg.header.frame_id = "map"  # Rviz에서 보는 프레임
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x,y,theta in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = -y
            pose.pose.position.z = 0.0

            qx, qy, qz, qw = self.yaw_to_quaternion(theta)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        return path
        

### 센서 FOV 정의 ####################################################################
# Sensor = namedtuple('Sensor',
#         'id cx cy yaw fov range')       # yaw, fov 는 rad

# SENSORS = [
#     Sensor('rgb_front', 2.0, 0.0, 0.0,              math.radians(90),  100),
#     Sensor('lidar',     0.0, 0.0, 0.0,              math.radians(360), 50),
#     Sensor('semantic_lidar', 0.0, 0.0, 0.0,         math.radians(360), 50),
#     Sensor('radar_front', 2.0, 0.0, 0.0,            math.radians(30),  100),
#     Sensor('seg_cam',   2.0, 0.0, 0.0,              math.radians(90),  100),
#     Sensor('depth_cam', 2.0, 0.0, 0.0,              math.radians(90),  100),
# ]
####################################################################################

    def yaw_to_quaternion(self,yaw):
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return q  # x, y, z, w
    
    def publish_ctrl(self):
        if self.warn_active == 1.0: #저속
            self.tm.vehicle_percentage_speed_difference(self.hero,5.0) #트래픽매니저가 제어
            return

        if self.shift_active == 2.0: 
            self.pub_s_res.publish(Float64(data=1.0))
            return

        if self.shift_active == 1.0: #토픽 값이 1이 되면 페일세이프 기능 on
            if not self.in_fail or not self.hero:
                return
                
            if self.has_parked:
                return

            if not self.waypoint or not self.left_lane_marking or not self.right_lane_marking:
                self.next_line()  # waypoint가 없으면 업데이트
                return
                
            left_type = self.left_lane_marking.type
            right_type = self.right_lane_marking.type

            # 왼쪽 Solid + 오른쪽 None → 평행 맞추고 정지
            if left_type == carla.LaneMarkingType.Solid and right_type == carla.LaneMarkingType.NONE:
                # 차량과 차선의 yaw 차이 계산
                hero_yaw = self.hero.get_transform().rotation.yaw
                lane_yaw = self.waypoint.transform.rotation.yaw
                angle_diff = abs(normalize_angle(hero_yaw - lane_yaw))

                # yaw 차이가 크면 조향 보정
                if angle_diff > 3.0:
                    steer = max(-1.0, min(1.0, normalize_angle(lane_yaw - hero_yaw) / 45.0))
                    ctrl = carla.VehicleControl(throttle=0.2, steer=steer, brake=0.0)
                    self.hero.apply_control(ctrl)
                    self.get_logger().info(f"▶ 평행 맞추는 중 (angle_diff={angle_diff:.2f})")
                    return

                # yaw 일치하면 정지
                ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                self.hero.apply_control(ctrl)
                self.has_parked = True
                self.get_logger().info("▶▶▶ 주차 조건 + 방향 일치 → 차량 정지 및 핸드브레이크")
                return

            # 아직 주차 조건 미달 → 우측 이동 계속
            ctrl = carla.VehicleControl(throttle=0.3, steer=0.1, brake=0.0)
            self.hero.apply_control(ctrl)
            self.get_logger().info("▶ 갓길 탐색 중: 우측으로 이동")
        


def main():
    rclpy.init()
    node = LidarFailSafe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
