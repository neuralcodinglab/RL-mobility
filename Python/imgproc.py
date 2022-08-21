import numpy as np
import cv2
import torch
import itertools
import re
import pandas as pd
from torch import nn
import math
import scikit_canny


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
        assert len(self.stack)>0
        if type(self.stack[0]) == torch.Tensor:
            return torch.cat(self.stack, dim=1) # Concatenate along channel dimension
        else:
            return self.stack

    def __len__(self):
        return len(self.stack)

class ImageProcessor(object):
    def __init__(self, phosphene_resolution=None, imsize=128, mode='edge-detection', edge_threshold=70,
                 device='cpu', adaptive_threshold=False, *args,**kwargs):
        """ @TODO
        - Extended image processing
        """
        self.mode = mode
        self.thr_high = edge_threshold
        self.thr_low  = edge_threshold // 2
        self.imsize = imsize
        self.device = device
        
        if phosphene_resolution is not None:
#             self.simulator = PhospheneSimulator(phosphene_resolution=(int(phosphene_resolution),int(phosphene_resolution)),
#                                                 size=(imsize,imsize),jitter=0.25,intensity_var=0.9,aperture=.66,sigma=0.60,)
            self.simulator =  DifferentiablePhospheneSimulator(phosphene_resolution=(int(phosphene_resolution),int(phosphene_resolution)),
                                                          size=(imsize,imsize),jitter=0.25,intensity_var=0.9,sigma=0.60,intensity=1., 
                                                          device=device)
        else:
            self.simulator = None
            
        self.canny = scikit_canny.DifferentiableCannyModule(sigma = 1., threshold=edge_threshold/255,
                                                            device=device, adaptive_threshold=adaptive_threshold)

    def __call__(self,state,):
        if self.mode == 'edge-detection' :
            frame = state['colors'] # RGB input
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.Canny(frame, self.thr_low,self.thr_high)
            frame = self.canny(frame)
            if self.simulator is not None:
                frame = self.simulator(frame, dilate_mask=True)
        elif self.mode == 'camera-vision' :
            frame = state['colors'] # RGB input
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = torch.from_numpy(frame).float() /255. 
            frame = frame.to(self.device).view(1,1,self.imsize, self.imsize)
            if self.simulator is not None:
                frame = self.simulator(frame)
        elif self.mode == 'no-vision':
            return torch.rand(1,1,self.imsize, self.imsize, device=self.device)
        else:
            raise NotImplementedError("\n\n'{}' is not recognized as valid image processing method. Try e.g. ('edge-detection' | 'camera-vision')".format(self.mode))
        return frame

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

    
class DifferentiablePhospheneSimulator(nn.Module):
    """ Uses two steps to convert  the stimulation vectors to phosphene representation:
    1. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,phosphene_resolution=(50,50),
                 size=(480,480),  jitter=0.35,
                 intensity_var=0.8, aperture=None,
                 sigma=0.8, intensity=15,
                 device=torch.device('cpu'),channels=1):
        super(DifferentiablePhospheneSimulator, self).__init__()
        
        # Device
        self.device = device
        
        # Gaussian kernel
        self.KERNEL_SIZE = 11 
        self.gaussian = self.get_gaussian_layer(kernel_size=self.KERNEL_SIZE, sigma=sigma, channels=channels).to(device)
        
        # Dilation layer
        self.dilation = self.get_dilation_layer().to(device)
        
        # Phosphene grid
        self.pMask = self.create_regular_grid(phosphene_resolution, size, jitter, intensity_var).to(device)
        self.intensity = intensity 
        
        if aperture:
            raise NotImplementedError # Don't know whether we will need this, but for now aperture is not implemented
            
            
        for p in self.parameters():
            p.requires_grad = False
    
    def get_dilation_layer(self):
        layer = nn.Conv2d(in_channels=1,
                   out_channels=1,
                   kernel_size=3,
                   stride = 1,
                   padding = 1,
                   bias=False  )
        layer.weight.data = torch.ones_like(layer.weight.data)
        return layer
    
    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter    
    
    def create_regular_grid(self, phosphene_resolution, size, jitter, intensity_var):
        """Returns regular eqiodistant phosphene grid of shape <size> with resolution <phosphene_resolution>
         for variable phosphene intensity with jitterred positions"""
        grid = torch.zeros(size,device=self.device)
        phosphene_spacing = np.divide(size,phosphene_resolution)
        for x in np.linspace(0,size[0],num=phosphene_resolution[0],endpoint=False)+0.5*phosphene_spacing[0]:
            for y in np.linspace(0,size[1],num=phosphene_resolution[1],endpoint=False)+0.5*phosphene_spacing[0]:
                deviation = np.multiply(jitter*(2*np.random.rand(2)-1),phosphene_spacing)
                intensity = intensity_var*(np.random.rand()-0.5)+1
                rx = np.clip(np.round(x+deviation[0]),0,size[0]-1).astype(int)
                ry = np.clip(np.round(y+deviation[1]),0,size[1]-1).astype(int)
                grid[rx,ry]= intensity
        return grid
    
    def forward(self, stimulation, dilate_mask=False):
        
        # Phosphene simulation
        if dilate_mask:
            stimulation = self.dilation(stimulation) # line-thickening of the activation mask
            stimulation = stimulation.clip(0,1)
            
        
        phosphenes = stimulation*self.pMask
        phosphenes = nn.functional.pad(phosphenes, ((self.KERNEL_SIZE-1)//2,)*4, mode='constant', value=0)
        phosphenes = self.gaussian(phosphenes) 
        return self.intensity*phosphenes