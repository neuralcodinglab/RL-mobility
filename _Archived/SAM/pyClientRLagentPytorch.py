import numpy as np
import socket
from PIL import Image


class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000, size = 128, timescale = 1):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.timescale  = timescale

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def reset_val(self):
        self._send(2, 0)
        return self._receive()
    
    def reset_test(self):
        self._send(3, 0)
        return self._receive()
    
    def step(self, action):
        self._send(4, action)
        return self._receive()
    
    def step_val(self, action):
        self._send(5, action)
        return self._receive()
    
    def step_test(self, action):
        self._send(6, action)
        return self._receive()

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))
    
    def state2usableArray(self, state):
        return np.array(state, "uint8").reshape(self.size, self.size, 3)
    
    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data   = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))