import numpy as np
data_handle=open("C:/Users/Hp/Desktop/my python programs/3_test.txt",'r')
for i in range(3):
    data_file=data_handle.readline()
    data_file_3=data_file.split(',')
    inputs=np.asfarray(data_file_3[0:])
    #inputs=inputs/255.0*0.99+0.01
    np.savetxt("Test.txt",inputs)
    pass
data_handle.close()
arr=np.loadtxt("Test.txt")
print(arr)


