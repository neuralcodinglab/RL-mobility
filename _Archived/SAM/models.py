import torch
from torch import nn
import numpy as np
import math
import cv2

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
        
        # Phosphene grid
        self.pMask = self.create_regular_grid(phosphene_resolution, size, jitter, intensity_var).to(device)
        self.intensity = intensity 
        
        if aperture:
            raise NotImplementedError # Don't know whether we will need this, but for now aperture is not implemented
    
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
    
    def forward(self, stimulation):
        
        # Phosphene simulation
        phosphenes = stimulation*self.pMask
        phosphenes = nn.functional.pad(phosphenes, ((self.KERNEL_SIZE-1)//2,)*4, mode='constant', value=0)
        phosphenes = self.gaussian(phosphenes) 
        return self.intensity*phosphenes   
    
    
class ThresholdedSobel(nn.Module):
    def __init__(self, k_size=3, initial_threshold=0.5, channels=1, device='cpu'):
        super(ThresholdedSobel, self).__init__()
        
        # device
        self.device = device

        # Layers
        self.sobel_filter_x = nn.Conv2d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        groups=channels,
                                        bias=False).to(device)


        self.sobel_filter_y = nn.Conv2d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        groups=channels,
                                        bias=False).to(device)
        
        # Parameters
        sobel_2D = torch.tensor(self.get_sobel_kernel(k_size),device=device, dtype=torch.float)
        x_weight = sobel_2D.view(1,1,k_size,k_size).repeat(channels,1,1,1)
        y_weight = sobel_2D.T.view(1,1,k_size,k_size).repeat(channels,1,1,1)
        self.sobel_filter_x.weight = nn.Parameter(x_weight,requires_grad=False)
        self.sobel_filter_y.weight = nn.Parameter(y_weight,requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(initial_threshold,
                                device=device, requires_grad=True))
        
    def get_sobel_kernel(self,k=5):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D
        
    def forward(self, img):
        grad_x = self.sobel_filter_x(img)
        grad_y = self.sobel_filter_y(img)
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        
        out      = grad_magnitude-self.threshold
        out_forw = torch.heaviside(out,torch.tensor(0, dtype=torch.float))
        out_bckw = torch.sigmoid(out)

        return out_bckw + out_forw.detach() - out_bckw.detach()

    
class OrientedSobel(nn.Module):
    def __init__(self, k_size=3, initial_threshold=0.5, channels=1, device='cpu'):
        super(OrientedSobel, self).__init__()
        
        # device
        self.device = device

        # Layers
        self.sobel_layer = nn.Conv2d(in_channels=channels,
                                        out_channels=channels*8,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        groups=channels,
                                        bias=False).to(device)
        self.weighting = nn.Conv2d(in_channels=8*channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            groups=channels,                       
            bias=False).to(device)

        
        # Parameters ###TODO: MULTI-CHANNEL
        self.sobel_layer.weight = nn.Parameter(oriented_sobel_kernels().to(device).repeat(channels,1,1,1),requires_grad=False)
        self.weighting.weight   = nn.Parameter(torch.ones(channels,8,1,1,device=device)/8,requires_grad=True)
        self.threshold = nn.Parameter(torch.tensor(initial_threshold,
                                device=device, requires_grad=True))
        
    def forward(self, img):
        edges    = nn.functional.relu(self.sobel_layer(img),inplace=True)
        edges    = self.weighting(edges)
        edges    = edges-self.threshold
        out_forw = torch.heaviside(edges,torch.tensor(0, dtype=torch.float))
        out_bckw = torch.sigmoid(edges)
        return out_bckw + out_forw.detach() - out_bckw.detach()

    
class SurrogateCanny(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 k_size=3,
                 channels=1,
                 device='cpu',
                 initial_threshold=0.5):
        super(SurrogateCanny, self).__init__()
        # device
        self.device = device


        ## layers
        # sobel
        self.sobel_filter_x = nn.Conv2d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        groups=channels,
                                        bias=False).to(device)


        self.sobel_filter_y = nn.Conv2d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        groups=channels,
                                        bias=False).to(device)
        
        # thinning layer
        self.directional_filter = nn.Conv2d(in_channels=channels,
                                            out_channels=channels*8,
                                            kernel_size=3,
                                            padding=1,
                                            groups=channels,
                                            bias=False).to(device)
        
        
        ## Weights
        # Sobel
        sobel_2D = self.get_sobel_kernel(k_size).to(device)
        x_weight = sobel_2D.view(1,1,k_size,k_size).repeat(channels,1,1,1)
        y_weight = sobel_2D.T.view(1,1,k_size,k_size).repeat(channels,1,1,1)
        self.sobel_filter_x.weight = nn.Parameter(x_weight,requires_grad=False)
        self.sobel_filter_y.weight = nn.Parameter(y_weight,requires_grad=False)

        # Thinning layer
        directional_weight = self.get_thin_kernels().repeat(channels,1,1,1).to(device)
        self.directional_filter.weight = nn.Parameter(directional_weight,requires_grad=False)
        
        # Threshold (optimizable)
        self.threshold = nn.Parameter(torch.tensor(initial_threshold,device=device, requires_grad=True))

    def get_sobel_kernel(self,k=3):
        # get range
        range_ = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range_, range_)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return torch.tensor(sobel_2D, dtype=torch.float)

    def get_thin_kernels(self,start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return torch.tensor(thin_kernels,dtype=torch.float).unsqueeze(dim=1)
    
    def forward(self, img,):
     
        # visual gradients
        grad_x = self.sobel_filter_x(img)
        grad_y = self.sobel_filter_y(img)
        
        # thick edges
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # threshold (& straight-through estimation)
        out      = thin_edges-self.threshold
        out_forw = torch.heaviside(out,torch.tensor(0, dtype=torch.float))
        out_bckw = torch.sigmoid(out)

        return out_bckw + out_forw.detach() - out_bckw.detach()
    
def oriented_sobel_kernels():
    w = torch.tensor([[[-0.5,  0. ,  0.5],
                      [-1. ,  0. ,  1. ],
                      [-0.5,  0. ,  0.5]],

                      [[-1. , -0.5,  0. ],
                      [-0.5,  0. ,  0.5],
                      [ 0. ,  0.5,  1. ]],

                      [[-0.5, -1. , -0.5],
                      [ 0. ,  0. ,  0. ],
                      [ 0.5,  1. ,  0.5]],

                      [[ 0. , -0.5, -1. ],
                      [ 0.5,  0. , -0.5],
                      [ 1. ,  0.5,  0. ]],

                      [[ 0.5,  0. , -0.5],
                      [ 1. ,  0. , -1. ],
                      [ 0.5,  0. , -0.5]],

                      [[ 1. ,  0.5,  0. ],
                      [ 0.5,  0. , -0.5],
                      [ 0. , -0.5, -1. ]],

                      [[ 0.5,  1. ,  0.5],
                      [ 0. ,  0. ,  0. ],
                      [-0.5, -1. , -0.5]],

                      [[ 0. ,  0.5, 1. ],
                      [ -0.5,  0. , 0.5],
                      [-1., -0.5  , 0.]]]).unsqueeze(dim=1)
    return w