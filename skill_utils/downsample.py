import numpy as np
from math import floor

rec=[[10.3,0]]
period=3.4

def downsample(rec, period):
    """
    Downsample a given recording (list composed of lists [timestamp, image])
    and return a version that matches the given period.
    """
    res=[]
    reclen=rec[-1][0] #recording length in miliseconds
    ts=list(np.linspace(0, reclen-(reclen%period), floor(reclen/period)+1))
    ts.reverse()
    # not the best method of doing it, but iterate over everything to find the individual frames closest to a given timestep.
    target_t=ts.pop()
    for i,frame in enumerate(rec):
        if frame[0]-target_t>=0:
            print(f"Found one with {frame[0]} for {target_t}")
            # check the previous index as well
            if (i>0) and target_t-rec[i-1][0]<frame[0]-target_t:
                res.append(rec[i-1])
            else:
                res.append(frame)
            if len(ts):
                target_t=ts.pop()
                print(f"new target: {target_t}")

    return res
