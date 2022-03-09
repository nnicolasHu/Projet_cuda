import numpy as np


ref = np.loadtxt('resRef.txt',dtype="float64",usecols =(0,2))
res = np.loadtxt('res.txt',dtype="float64",usecols =(0,2))

sh = np.shape(ref)
print("The loaded array shape is:", sh, ref[1,0],ref[1,1])

diff = ref[:,1]-res[:,1]

good = True
for i in range(sh[0]):
   if abs(diff[i]) > 1.e-13:
       print('variation =',diff[i], i)
       good = False

if good: print("Calcul // OK")
else: print("error programme parallele")
