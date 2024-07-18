'''
2d versions of classes in teleop.py
'''
from threading import Thread, Event
import numpy as np
from spatialmath import SE3, UnitQuaternion

from skill_utils.teleop import KeyboardCommandHandler
from skill_utils.format_pose import to_format, from_format

class KeyboardPoseController(KeyboardCommandHandler):
    def __init__(self, pose):
        self.moveeps=0.01
        self.pose=pose
        
        super().__init__()

    @property
    def formatted_pose(self):
        return to_format(self.pose)

    @formatted_pose.setter
    def formatted_pose(self, val):
        self.pose=from_format(val)

    def w(self):
        # forward
        self.pose=SE3.Trans(-self.moveeps,0,0) * self.pose

    def s(self):
        # backward
        self.pose=SE3.Trans(self.moveeps,0,0) * self.pose

    def a(self):
        # right
        self.pose=SE3.Trans(0,-self.moveeps,0) * self.pose

    def d(self):
        # left
        self.pose=SE3.Trans(0,self.moveeps,0) * self.pose
