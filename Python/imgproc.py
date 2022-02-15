import numpy as np
import cv2
import torch
import itertools
import re
import pandas as pd


class FrameStack(object):
    def __init__(self, stack_size=4, *args,**kwargs):
        self.stack_size = stack_size
        self.stack = []

    def update_with(self,frame):
        self.stack.append(frame)
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)
        return self.get()

    def get(self):
        return torch.cat(self.stack, dim=1)

    def __len__(self):
        return len(self.stack)

class ImageProcessor(object):
    def __init__(self, phosphene_resolution=None, imsize=128, mode='edge_detection', canny_threshold=70, *args,**kwargs):
        """ @TODO
        - Extended image processing
        """
        self.mode = mode
        self.thr_high = canny_threshold
        self.thr_low  = canny_threshold // 2
        self.imsize = imsize
        if phosphene_resolution is not None:
            self.simulator = PhospheneSimulator(phosphene_resolution=(phosphene_resolution,phosphene_resolution),size=(imsize,imsize),
                                                     jitter=0.25,intensity_var=0.9,aperture=.66,sigma=0.60,)
        else:
            self.simulator = None

    def __call__(self,state,):
        if self.mode == 'edge_detection':
            frame = state['colors'] # RGB input
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.simulator is not None:
            frame = cv2.Canny(frame, self.thr_low,self.thr_high)
            frame = self.simulator(frame)
        frame = frame.astype('float32')
        return torch.Tensor(frame / 255.).view(1,1,self.imsize, self.imsize)

class PhospheneSimulator(object):
    def __init__(self,phosphene_resolution=(50,50), size=(480,480),  jitter=0.35, intensity_var=0.9, aperture=.66, sigma=0.8, custom_grid=None, *args,**kwargs):
        """Phosphene simulator class to create gaussian-based phosphene simulations from activation mask
        on __init__, provide custom phosphene grid or use the grid parameters to create one
        - aperture: receptive field of each phosphene (uses dilation of the activation mask to achieve this)
        - sigma: the size parameter for the gaussian phosphene simulation """
        if custom_grid is None:
            self.phosphene_resolution = phosphene_resolution
            self.size = size
            self.phosphene_spacing = np.divide(size,phosphene_resolution)
            self.jitter = jitter
            self.intensity_var = intensity_var
            self.grid = self.create_regular_grid(self.phosphene_resolution,self.size,self.jitter,self.intensity_var)
            self.aperture = np.round(aperture*self.phosphene_spacing[0]).astype(int) #relative aperture > dilation kernel size
        else:
            self.grid = custom_grid
            self.aperture = aperture
        self.sigma = sigma
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.aperture,self.aperture))
        self.k_size = 11 #np.round(4*sigma+1).astype(int) # rule of thumb: choose k_size>3*sigma

    def __call__(self,activation_mask):
        """ returns the phosphene simulation (image), given an activation mask"""
        assert self.grid.shape == activation_mask.shape
        self.mask = cv2.dilate(activation_mask, self.dilation_kernel, iterations=1)
        phosphenes = self.grid * self.mask
        phosphenes = cv2.GaussianBlur(phosphenes,(self.k_size,self.k_size),self.sigma)
        return phosphenes

    def create_regular_grid(self, phosphene_resolution, size, jitter, intensity_var):
        """Returns regular eqiodistant phosphene grid of shape <size> with resolution <phosphene_resolution>
         for variable phosphene intensity with jitterred positions"""
        grid = np.zeros(size)
        phosphene_spacing = np.divide(size,phosphene_resolution)
        for x in np.linspace(0,size[0],num=phosphene_resolution[0],endpoint=False)+0.5*phosphene_spacing[0] :
            for y in np.linspace(0,size[1],num=phosphene_resolution[1],endpoint=False)+0.5*phosphene_spacing[0]:
                deviation = np.multiply(jitter*(2*np.random.rand(2)-1),phosphene_spacing)
                intensity = intensity_var*(np.random.rand()-0.5)+1
                rx = np.clip(np.round(x+deviation[0]),0,size[0]-1).astype(int)
                ry = np.clip(np.round(y+deviation[1]),0,size[1]-1).astype(int)
                grid[rx,ry]= intensity
        return grid
