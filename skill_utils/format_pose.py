"""
Utilities to convert recording data into the format the model takes in and outputs
"""
import numpy as np


def to_format(pose):
    '''
    format the 16d homogeneous transform into a 3d+6d=9d representation:
    (x,y,z) transform + first two columns of SO(3)
    as per the paper Zhou, Barnes & Lu et al. (2020-06-08) On the Continuity of Rotation Representations in Neural Networks, .
    '''
    if pose.shape!=(4,4):
        pose=np.reshape(pose,(4,4))
    assert pose.shape==(4,4)
    return np.hstack((pose[:3,3], pose[:3,0], pose[:3,1]))
    

def from_format(pose):
    '''
    format 3d+6d=9d representation into 16d homogeneous transform
    as per the paper Zhou, Barnes & Lu et al. (2020-06-08) On the Continuity of Rotation Representations in Neural Networks, .
    '''
    t=pose[:3]
    b1=normalize(pose[3:6])
    b2=normalize(pose[6:9]-(np.dot(b1,pose[6:9])*b1))
    b3=np.cross(b1,b2)
    return np.vstack((np.stack((b1,b2,b3,t)).T,
                      np.array([0,0,0,1])))

def normalize(vec):
    '''
    normalize a vector
    '''
    return vec/np.linalg.norm(vec)
