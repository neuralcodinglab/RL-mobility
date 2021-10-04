# -*- coding: utf-8 -*-
"""
@author: Laura Pijnacker
"""
import numpy as np
import math
import cv2
import random

class Memory(object):
    def __init__(self, theta, decay, trace_increase, decay_activation, input_effect, n_phosphenes):
        self.theta = theta #0 for now, can be higher, e.g. minimal electrode current to elicit phosphene
        self.decay = decay #decay of memory
        self.decay_activation = decay_activation #decay of activation 
        self.input_effect = input_effect #effect of new input and trace
        self.trace_increase = trace_increase #how much new input increases the memory trace
        self.n_phosphenes = n_phosphenes
        self.memory = np.zeros((1,self.n_phosphenes))        #establish empty memory trace for each phosphene
        self.outputs = np.zeros((1,self.n_phosphenes))   #establish empty array to store output values for size
        
class PhospheneSimulator(object):
    def __init__(self, ecc_coverage = 30, polar_coverage = 2*math.pi, phosphenes_per_polar = 25,
                 phosphenes_per_ecc = 70, noise_scale = 0.14, coeff = 60, eff_activation = 1,
                 windowSize = 480, custom_map = None):
        """ Phosphene simulator class to create gaussian-based phosphene simulations from activation mask
        on __init__, provide custom polar phosphene map or provide parameters to create one
        
        """
        if custom_map is None:
            self.ecc_coverage = ecc_coverage
            self.polar_coverage = polar_coverage
            self.phosphenes_per_polar = phosphenes_per_polar
            self.phosphenes_per_ecc = phosphenes_per_ecc
            self.noise_scale = noise_scale
            self.phosphenes = self.get_phosphene_map(self.ecc_coverage, 
                                                     self.polar_coverage, 
                                                     self.phosphenes_per_polar, 
                                                     self.phosphenes_per_ecc, 
                                                     self.noise_scale)
        else:
            self.phosphenes = custom_map
            self.total_phosphenes = len(self.phosphenes)
            
        self.coeff = coeff
        self.eff_activation = eff_activation
        self.windowSize = windowSize


        """"Size in degrees is calculated for each phosphene 
        - using Horton and Hoyt's formula for cortical magnification.
        Next, a coefficient is used to translate this value to size in pixels.
        """
        self.sizes_deg = self.eff_activation/(17.3/(0.75+self.phosphenes[:,1]))
        self.sizes = self.sizes_deg*self.coeff

        # Translating polar phosphene coordinates to x and y coordinates.
        self.c_x, self.c_y = self.pol2cart(self.phosphenes[:,0] ,self.phosphenes[:,1])


        
    def __call__(self, activation_mask, memory_trace):
        """ Returns the phosphene simulation (image) and a memory trace object, given an activation mask
        """

        # Create empty phosphene array
        image_phosphenes = np.zeros((activation_mask.shape[0],activation_mask.shape[1],3), 'float32')
        # Activate phosphenes in empty array
        active_phosphenes, memory_trace = self.activate_phosphenes(image_phosphenes, activation_mask, memory_trace)
        
        #image generated is rgb, convert to grayscale for the RL agent
        grayscale = cv2.cvtColor(active_phosphenes, cv2.COLOR_BGR2GRAY)
        
        #img_gabor = self.apply_gabor(grayscale,10)
        
        return grayscale, memory_trace
    

    
    def pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    
            
    def get_phosphene_map(self, ecc_coverage, polar_coverage, phosphenes_per_polar, phosphenes_per_ecc, noise_scale):
        """ Make a list of phosphene coordinates, linearly 
        spaced across a map of polar angles and eccentricities.
        """
        
        pols = np.power(np.linspace(0,1,phosphenes_per_polar), 1) * polar_coverage
        eccs = np.power(np.linspace(0,1,phosphenes_per_ecc), 4) * ecc_coverage 
        
        self.total_phosphenes = int ( pols.shape[0] * eccs.shape[0] ) 
        phosphenes = np.empty( (self.total_phosphenes, 2 ) )
        
        count = 0
    
        # Creating polar and ecc angles
        for pol in pols:
            for ecc in eccs:
                #create random noise to add to angles
                noise = np.random.normal(0.0, noise_scale, 
                                              size = (2,))
                p = pol + noise[0] 
                e = ecc + np.power(noise[1], 6) 
                phosphenes[count][0], phosphenes[count][1] = p, e
                count += 1
                
        return phosphenes
    

    def activate_phosphenes(self, image_phosphenes, image_filtered, memory_trace):
        """ Takes a filterd image and activates phosphenes on all pre-generated 
        c_x and c_y values.
        """
        
        new_memory = np.zeros(self.total_phosphenes)                            #Generate a new memory trace
        memory_trace.memory = np.vstack([memory_trace.memory, new_memory])  #Append new memory to trace
        memory_trace.outputs = np.vstack([memory_trace.outputs, new_memory]) #do the same to store output values
            
        #print('memory trace rows - frame number')
        #print(len(memory_trace.memory))
        #print('memory trace columns' - total number of phosphenes)
        #print(len(memory_trace.memory[0]))
        #print('n_frame')
            
        #loop over every phosphene
        for i in range(0, self.total_phosphenes):
            
            # Making a Gaussian in a numpy with the shape of a tiny window
            # SIZE_CONTROL THE EXTENSION OF THE PHOSPHENES. SHOULD BE A PARAMETER.
            SIZE_CONTROL = 17 
            x = int(self.c_x[i] * SIZE_CONTROL + self.windowSize/2)
            y = int(self.c_y[i] * SIZE_CONTROL + self.windowSize/2)
        
            s = int(self.sizes[i])
            
            ## Apply the memory trace.
            n_frame = len(memory_trace.memory)-1    #looks at size of memory trace to determine frame number

            #print(n_frame)
            #print('i')
            #print(i)
            
            memory_trace.memory[n_frame, i] = memory_trace.decay*memory_trace.memory[n_frame-1, i] + memory_trace.trace_increase*s
            memory_trace.outputs[n_frame, i] = memory_trace.decay_activation*memory_trace.outputs[n_frame-1, i] + memory_trace.input_effect*(s - memory_trace.memory[n_frame-1, i])
        
            if memory_trace.outputs[n_frame, i] < memory_trace.theta:
                memory_trace.outputs[n_frame, i] = 0
            
            #s is then the size of the phosphene based on the memory trace and cortical magnification
            s = int(memory_trace.outputs[n_frame, i])

                
            #before we make the gaussian, use bosking's algorithm to ajust for the effect of current strength
            halfs = s // 2
            MD = 2 #1 #mm, max diam of activated cortex (5.3 is for surface)
            slope = 0.55 # mm/mA, max slope for increase in diam of activity with increase in current
            I = np.sum(image_filtered[y-halfs:y+halfs, x-halfs:x+halfs]) / np.power(s,2) #current used for simulation
            if np.isnan(I):
                I = 0
            #I currently tends to be 0 or 2-50 range
            #we need a more realistic way to translate this to current, but for now...
            I_half = 5 #mA, current at which half of the saturation value is reached
            if I != 0:
                AC = MD / ( 1 + math.exp(-slope*(I-I_half)))
                s = AC*s
                
                if s < 2:
                # print('Tiny phosphene: artificially making size == 2')
                    s = 2
                    
                elif (s % 2) != 0:
                    s =  s + 1
                else:
                    None
                
                halfs = int(s // 2)
                
                # if there is activated pixels in the area for that phosphene,
                # then we activate that phosphene 
                if len(np.where(image_filtered[y-halfs:y+halfs, x-halfs:x+halfs] > 0)[0]) > 0:
                    
        
                    g = self.makeGaussian(size = s , fwhm = s / 3, center=None)
                    g = np.expand_dims(g,-1)
                        
                    # Here, we implement a very simple intensity calculation strategy,
                    # based on the percentage of the phosphene area with active pixels
                    # we should call this current and then implement a current->intensity function
                    g = g * (np.sum(image_filtered[y-halfs:y+halfs, x-halfs:x+halfs]) / np.power(s,2))
                      
                    #idea for modulating intensity by decay effect?
                    #g = g*(diff/10)

                    # normalizing to 0-1
                    g = g / 100
        
                    
                    windowShape = image_phosphenes[y-halfs:y+halfs, x-halfs:x+halfs].shape
                    g = g[0:windowShape[0],0:windowShape[1]]
        
                    #print(g.shape)
                    image_phosphenes [y-halfs:y+halfs, x-halfs:x+halfs] = image_phosphenes [y-halfs:y+halfs, x-halfs:x+halfs] + np.repeat(g, 3, -1)
                else:
                    # print(print(len(np.where(auto[y-halfs:y+halfs, x-halfs:x+halfs][0] > 0)[0])))
                    None
            else:
                s = 0
            #this rsults in a lot of values 0.02157... but this is for I=0
            
            
        return image_phosphenes, memory_trace

    
    def makeGaussian(self, size, fwhm = 3, center=None):
        """ Make a square gaussian kernel.
    
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
    
        x = np.arange(0, size, 1, 'float32')
        y = x[:,np.newaxis]
    
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
            
        gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)  
        
        filtered_gaussian = self.apply_gabor(gaussian, size)
    
        
        return filtered_gaussian    

    def apply_gabor(self, img, size):
        #applies a gabor filter to the image/kernel
        
        # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        #ksize = 9         # = 21  size of gabor filter (n, n) - odd number (min 9?)
        ksize = int(np.ceil(size) // 2 * 2 + 1)
        if ksize<9:
            ksize = 9
        sigma = 2          # = 8.0 standard deviation of the gaussian function
        #theta = np.pi/4    # = np.pi/4 orientation of the normal to the parallel stripes
        theta = random.uniform(0, 181)*np.pi/180
        g_lambda = 10     # = 10.0 wavelength of the sunusoidal factor
        gamma = 0.5        # = 0.5  spatial aspect ratio
        psi = 0             # = 0 phase offset
        ktype = cv2.CV_32F  # = cv2.CV_32F type and range of values that each pixel in the gabor kernel can hold
        
        g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, g_lambda, gamma, psi, ktype)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        
        return filtered_img        
