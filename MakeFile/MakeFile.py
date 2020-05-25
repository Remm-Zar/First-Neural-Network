import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
s=150
data=np.load("C:/Users/Hp/Desktop/my python programs/NN/NN/SData.npz")
print("LR:\n",data['ARR_LR'])
size1=data['ARR_H'].size
size2=data['ARR_LR'].size
print(size1," ",size2)
LR_LIST=[]
for i in range(size2):
    for j in range(size2):
        LR_LIST.append(data['ARR_LR'][i])
        pass
    pass
X=np.array(LR_LIST).reshape(1,size1)
print(X)
Y=data['ARR_H']
Z=data['ARR_EF']
print("HN:\n",data['ARR_H'])
print("EFF:\n",data['ARR_EF'])
hf=plt.figure()
ha=hf.add_subplot(111,projection='3d')
#x,y=np.meshgrid(X,Y)
#ha.plot(x,y,Z,label='parametric curve')
ha.scatter(X,Y,Z,s=s,label="Statistic")
#ha.plot_wireframe(X,Y,Z)
plt.show()
