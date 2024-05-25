# Numpy, OpenCV, os, sys
import cv2
import os, sys

# Local files
pathname = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.insert(0,pathname)
import pyClient

# Connect to Unity environment
ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 12000       # Port number that the TCP/IP interface listens to
size       = 128
grayscale  = False
screen_height = screen_width = size
channels = 1 if grayscale else 16
environment = pyClient.Environment(ip = ip, port = port, size = size, channels=channels)
assert (environment.client is not None), "Please start Unity server environment first!"

# reset the environment
end, reward, state_raw = environment.reset()

# Create window to display the frames
class Window:
    def __init__(self,windowname='Hallway', size=(1200,1200), frame=None):
        self._size          = size
        self._windowname    = windowname

    def show(self,frame):
        frame = cv2.resize(frame, self._size)
        cv2.imshow(self._windowname, frame)

disp = Window(windowname='Hallway', size=(600,600))


# Initialize some phosphene simulations
class Simulator():
    def __init__(self, low=(26,26),high=(60,60),**kwargs):
        # self.low_res  = imgproc.PhospheneSimulator(phosphene_resolution=low,**kwargs)
        # self.high_res = imgproc.PhospheneSimulator(phosphene_resolution=high,**kwargs)
        self.sim_mode = 0

    def __call__(self,frame):
        if self.sim_mode==0:
            if grayscale:
                return frame
            else:
                return frame[:,:,::-1].astype('uint8')
        elif self.sim_mode == 1:
            raise NotImplementedError
            # frame = cv2.resize(frame, (480,480))
            # contours = cv2.Canny(frame,35,70)
            # phosphenes = self.low_res(contours)
            # return (255*phosphenes/phosphenes.max()).astype('uint8')
        elif self.sim_mode == 2:
            raise NotImplementedError
            # frame = cv2.resize(frame, (480,480))
            # contours = cv2.Canny(frame,35,70)
            # phosphenes = self.high_res(contours)
            # return (255*phosphenes/phosphenes.max()).astype('uint8')

simulator = Simulator(low=(26,26),high=(60,60),sigma=1.2)

while environment.client:

    # Display current state
    if grayscale:
        frame = simulator(state_raw['grayscale'])
    else:
        frame = simulator(state_raw['colors'])
    disp.show(frame)

    # Get key
    key = cv2.waitKey(0)
    if key == ord('w'):
        end, reward, state_raw = environment.step(0)
        print('action: {}, reward: {}, end {}'.format(0, reward, end))

    if key == ord('a'):
        end, reward, state_raw = environment.step(1)
        print('action: {}, reward: {}, end {}'.format(0, reward, end))

    if key == ord('d'):
        end, reward, state_raw = environment.step(2)
        print('action: {}, reward: {}, end {}'.format(0, reward, end))

    if key == ord('r'):
        end, reward, state_raw = environment.reset(2)
        print('action: {}, reward: {}, end {}'.format(0, reward, end))

    if key == 49:
        simulator.sim_mode = 0

    if key == 50:
        simulator.sim_mode = 1

    if key == 51:
        simulator.sim_mode = 2

    if key == ord('q'):
        break
