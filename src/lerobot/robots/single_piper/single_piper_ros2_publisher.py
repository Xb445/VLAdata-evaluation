#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from lerobot.errors import DeviceNotConnectedError

from .single_piper import SinglePiper

logger = logging.getLogger(__name__)


class SinglePiperRos2Publisher(Node):
    def __init__(self, robot: SinglePiper, image_topic: str = "/single_piper/image", joint_topic: str = "/single_piper/joints", frequency: float = 30.0):
        super().__init__("single_piper_publisher")
        self.robot = robot
        self.frequency = frequency

        self.image_pub = self.create_publisher(Image, image_topic, 10)
        self.joint_pub = self.create_publisher(Float32MultiArray, joint_topic, 10)

        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)

        self.bridge = CvBridge()

    def timer_callback(self) -> None:
        try:
            obs = self.robot.get_observation()
        except DeviceNotConnectedError:
            return

        joint_values = []
        for key in sorted(obs.keys()):
            if key.endswith(".pos"):
                joint_values.append(obs[key])

        if joint_values:
            joint_msg = Float32MultiArray()
            joint_msg.data = joint_values
            self.joint_pub.publish(joint_msg)

        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                height, width, channels = value.shape
                if channels == 3:
                    image_msg = self.bridge.cv2_to_imgmsg(value, encoding="rgb8")
                    image_msg.header.stamp = self.get_clock().now().to_msg()
                    image_msg.header.frame_id = key
                    self.image_pub.publish(image_msg)


class SinglePiperRos2Wrapper:
    def __init__(self, robot: SinglePiper, frequency: float = 30.0):
        self.robot = robot
        self.frequency = frequency
        self._ros_node: SinglePiperRos2Publisher | None = None
        self._executor = None
        self._thread = None
        rclpy.init()

    def start(self) -> None:
        self._ros_node = SinglePiperRos2Publisher(self.robot, frequency=self.frequency)
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._ros_node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        logger.info(f"SinglePiper ROS2 publisher started at {self.frequency}Hz")

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._ros_node is not None:
            self._ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("SinglePiper ROS2 publisher stopped")
