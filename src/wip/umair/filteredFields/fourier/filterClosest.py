def closest (num, arr):
    curr = arr[0]
    for index in range (len (arr)):
        if abs (num - arr[index]) < abs (num - curr):
            curr = arr[index]
    return curr


import numpy as np

#x = np.linspace(0, 2*np.pi, 1024)
#deltaX = (x[1]-x[0])
#kX  = np.fft.fftfreq(len(x),  d=deltaX)
kX = [0.01, 0.03, 0.07, 0.07853981633961, 0.0785398163396, 0.015, 0.0715]
lambdaX = 40
kC  = np.pi/lambdaX  # Cutoff wave no. in axial(z)      direction

number = kC
array = kX
print (kX)
print (kC)
print (closest(number, array))


