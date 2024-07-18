'''
operate arm and record as binarized np array
'''
import numpy as np
import time
import sys
from datetime import datetime

import panda_py
from panda_py import controllers
import spatialmath.base as smb


if len(sys.argv) == 1:
    raise RuntimeError("Please specify recording length.")
LEN = int(sys.argv[1])

# connect to robot
panda = panda_py.Panda("172.16.0.2")
panda.move_to_start()

input(f'Teach a trajectory for {LEN} seconds. Press enter to begin.')
panda.teaching_mode(True)
panda.enable_logging(LEN * 1000)
time.sleep(LEN)
panda.teaching_mode(False)
print("Recording has ended.")

log = panda.get_log()
print(log.keys())
# print(f"endeffs: {endeffs[0:2]} total size:{endeffs.shape}")
# print(f"pos/rot for first: {poss[0]} {rots[0]}")

# save the log
filename = f"log_{datetime.now().isoformat()}.npy"
np.save(filename, log)
print(f"Saved log as {filename}")
