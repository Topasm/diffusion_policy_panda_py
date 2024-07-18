'''Utilities for simple teleoperation with the end-effector
orientation staying constant.
'''
from threading import Thread, Event
import time
from datetime import timedelta, datetime

import numpy as np
from spatialmath import SE3, UnitQuaternion

from skill_utils.getch import getch

FREQ=100

def translate_input(char):
    '''
    turn keypress into appropriate command.
    '''
    keymap={
        'w': "forward",
        's': "backward",
        'a': "left",
        'd': "right",
        'q': "down",
        'e': "up",
        ' ': "vacuum",
        'h': "home",
        'u': "update",
        't': "quit",
    }
    return keymap[char]

        
class KeyboardInput(Thread):
    def __init__(self):
        """
        A multithreaded keyboard input receiver.
        """
        super().__init__()
        self.stop_event=Event()
        self.dat=None

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while not self.stop_event.is_set():
            inp=getch()
            self.dat=inp
            
class KeyboardHandler(KeyboardInput):
    def __init__(self):
        """
        A multithreaded keyboard input receiver. Each letter pressed calls its respective function, if such a function exists.
        """
        super().__init__()

    def process_key(self, char):
        """
        Process the keypress and call the relevant function

        Parameters
        ----------
        char : string, single char
            input keypress
        """
        if char:
            func=getattr(self, char, None)
            if callable(func):
                func()
            else:
                print(f"WARNING: {char} key not bound")

    @property
    def dat(self):
        return self.dat
    
    @dat.setter
    def dat(self, val):
        self.process_key(val)

class PoseControl(KeyboardHandler):
    def __init__(self, pose):
        """
        Simple 3D controller. Pose is the initial endeff pose.
        """
        self.pose=pose
        super().__init__()
        
    def w(self):
        # forward
        self.pose=SE3.Trans(self.moveeps,0,0) * self.pose

    def s(self):
        # backward
        self.pose=SE3.Trans(-self.moveeps,0,0) * self.pose

    def a(self):
        # right
        self.pose=SE3.Trans(0,-self.moveeps,0) * self.pose

    def d(self):
        # left
        self.pose=SE3.Trans(0,self.moveeps,0) * self.pose

    def e(self):
        # up
        self.pose=SE3.Trans(0,0,self.moveeps) * self.pose

    def q(self):
        # down
        self.pose=SE3.Trans(0,0,-self.moveeps) * self.pose

class KeyboardCommandHandler(KeyboardHandler):
    def __init__(self):
        self.endevent=Event()
        self.policyevent=Event()
        self.startevent=Event()
        self.stopevent=Event()
        self.delevent=Event()
        self.stagecounter=0
        
        super().__init__()
        
    def q(self):
        # quit program
        self.endevent.set()
        print("Quitting!")

    def c(self):
        # hand control to policy
        self.policyevent.set()

    def g(self):
        # start recording
        self.startevent.set()

    def h(self):
        # stop recording
        self.stopevent.set()

    def j(self):
        # delete most recent episode
        self.delevent.set()

    def i(self):
        self.stagecounter+=1


class Teleop():
    def __init__(self, mode="real", ip="10.0.0.2", record=False):
        """
        Multi-threaded teleop interface.

        Parameters
        ----------
        mode : string: "real" or "sim"
            whether to run teleop in real life or in simulation
        ip : string
            ip of robot for "real" option

        """
        self.record=record
        self.homeq=None # optional, if the home configuration is to be modified
        
        if mode=="real":
            # run in real life
            self.real=True
            self.panda=panda_py.Panda(ip)
            self.gripper=panda_py.libfranka.VacuumGripper(ip)
            self._endeff=SE3(self.panda.get_pose()) #store as SE3
            # use a cartesianimpedance controller
            tra_stiff=200 #default is 200
            rot_stiff=20 #default is 10
            self.ctrl=controllers.CartesianImpedance(
                impedance=np.array(
                    [[tra_stiff,0,0,0,0,0],
                     [0,tra_stiff,0,0,0,0],
                     [0,0,tra_stiff,0,0,0],
                     [0,0,0,rot_stiff,0,0],
                     [0,0,0,0,rot_stiff,0],
                     [0,0,0,0,0,rot_stiff]]),
                nullspace_stiffness=0.1,
                filter_coeff=1.0)
            self.panda.start_controller(self.ctrl)
            if self.record:
                # start logging
                # do it for 120 seconds for now
                self.panda.enable_logging(FREQ*120)
        else:
            # run in simulation maybe?
            self.real=False
            self.env=swift.Swift()
            self.env.launch(realtime=True)
            self.panda=rtb.models.Panda()
            self._endeff=self.panda.fkine(self.panda.q)
            self.env.add(self.panda)
            if self.record:
                print("WARNING: recording in sim not implemented yet")

        # how much to move by in each press
        self.moveeps=0.01
        # how long to try to vacuum for
        self.gripeps=timedelta(seconds=0.5)

        self.keyhandler=KeyboardHandler(self.moveeps, self._endeff)

    def update_endeff(self):
        if self.real:
            #real
            self._endeff=SE3(self.panda.get_pose())
        else:
            #sim
            self._endeff=self.panda.fkine(self.panda.q)

    @property
    def endeff(self):
        return self._endeff
    
    @endeff.setter
    def endeff(self, val):
        """
        set endeff pose and move robot to new pose
        """
        self._endeff=val
        if self.real:
            #real
            # self.panda.move_to_joint_position(panda_py.ik(self._endeff.data[0]))
            self.ctrl.set_control(self._endeff.t, UnitQuaternion(self._endeff).vec_xyzs)
        else:
            #sim
            arrived=False
            while not arrived:
                v,arrived=rtb.p_servo(self.panda.fkine(self.panda.q),self._endeff,1)
                self.panda.qd = np.linalg.pinv(self.panda.jacobe(self.panda.q)) @ v
                self.env.step(0.05)
            # self.panda.q=self.panda.ikine_LM(self._endeff, q0=self.panda.q).q
            
    def home(self):
        if self.real:
            #real
            if self.homeq==None:
                self.panda.move_to_start()
            else:
                self.panda.move_to_joint_position(self.homeq)
            self.update_endeff()
            self.panda.start_controller(self.ctrl)
        else:
            #sim
            self.panda.q=self.panda.qr
            self.update_endeff()
            self.env.step(0.05)

    def vacuum(self):
        # is vacuum on?
        state=self.gripper.read_once()
        if state.part_present:
            try:
                self.gripper.drop_off(self.gripeps)
            except:
                # if unsuccessful
                self.gripper.stop()
        else:
            try:
                self.gripper.vacuum(3,self.gripeps)
            except:
                # if unsuccessful
                self.gripper.stop()

    def step(self):
        self.endeff=self.keyhandler.pose
                
    def quit(self):
        if self.real:
            self.gripper.stop()
            if self.record:
                # stop logging, save log
                self.panda.disable_logging()
                log=self.panda.get_log()
                filename=f"log_{datetime.now().isoformat()}.npy"
                np.save(filename,log)
                print(f"Saved log as {filename}")
        quit()


def keyboard_test():
    import time
    with KeyboardHandler() as k:
        print("Listening...")
        time.sleep(10)
        
if __name__=="__main__":
    keyboard_test()
