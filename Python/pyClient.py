import numpy as np
import socket

class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000, size = 128, channels=3):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.channels   = channels

        try:
            self.client.connect((ip, port))
        except OSError as msg:
            self.client = None

    def reset(self, kind=0):
        """" {'tr_complex': 0, 'tr_simple': 1,'test_complex': 2, 'test_simple': 3}"""
        self._send(1, kind)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()
    
    def setRandomSeed(self, action):
        self._send(3, action)
        return self._receive()

    def _receive(self):
        data   = self.client.recv(2 + self.channels * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, len(data))] # raw state
        return end, reward, state
    
    def state2arrays(self,state):
        if self.channels == 3:
            return {'colors' : self.state2usableArray(state),}
        
        else:
            state  = np.array(state, "uint8").reshape(self.size, self.size, self.channels)
            arrays = {'colors' : state[...,:3],
                    'objseg' : state[...,3:6],
                    'semseg': state[...,6:9],
                    'normals'   : state[...,9:12],
                    'flow'   : state[...,12:15],
                    'depth'  : state[...,15]}
            return arrays
    
    def state2usableArray(self, state):
        return np.array(state, "uint8").reshape(self.size, self.size, 3)

    def _send(self, action, command):
        self.client.send(bytes([action, command]))
