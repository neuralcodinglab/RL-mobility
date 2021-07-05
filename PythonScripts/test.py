import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
from PIL import Image

import pyClientRLagentPytorch
import utils
import time

# Connect to Unity environment
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size = 128
screen_height = screen_width = size
environment = pyClientRLagentPytorch.Environment(ip = ip, port = port, size = size)
print(environment.client)

# reset the environment
end, reward, state_raw = environment.reset()

# Create window to display the frames
disp = Window(windowname='Hallway', size=(600,600))
if environment.client:
    disp.start()


while disp.stopped == False:



    image_array = environment.state2usableArray(state_raw)
    disp.frame = image_array[:,:,::-1].astype('uint8')

    key = disp.getKey()

    if key == ord('w'):
        end, reward, state_raw = environment.step(0)
        print(reward)

    if key == ord('a'):
        end, reward, state_raw = environment.step(1)
        print(reward)

    if key == ord('d'):
        end, reward, state_raw = environment.step(2)
        print(reward)

    if key == ord('r'):
        end, reward, state_raw = environment.reset()
        print(reward)

    if key == ord('q'):
        break


    time.sleep(0.01)
    # end, reward, state_raw = environment.step(0)

disp.stop()

#
# for i in range(10):
#     end, reward, state_raw = environment.step(0)
#     environment.state2usableArray(state_raw)
#     image_array = environment.state2usableArray(state_raw)
#     display.frame = image_array
#     time.sleep(0.2)
# display.stop()
