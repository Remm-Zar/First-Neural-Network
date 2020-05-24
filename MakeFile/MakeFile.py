import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
nx=5
ny=5
y_1=[100,150,200,250,300]
y=np.array(y_1)
x_1=[94,96,100,93,90]
x=np.array(x_1)
z=np.cos(x)
s=100
data=[[100,94],[150,96],[200,100],[250,93],[300,90]]
arr=np.array(data).reshape(2,5)

print(arr)
hf=plt.figure()
ha=hf.add_subplot(111,projection='3d')
#X,Y=np.meshgrid(x,y)
#ha.plot(x,y,z,label='parametric curve')
#ha.scatter(x,y,z,s=s)
ha.plot_wireframe(x,y,arr)
plt.show()
