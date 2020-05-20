import matplotlib.pyplot as plt
import NN
import numpy as np
#main unformation
in_nodes=784
h_nodes=100
out_nodes=10
learn_rate=0.3
#NN object
n=NN.NN()
n._init_(in_nodes,h_nodes,out_nodes,learn_rate)
#training data
data_file=open("C:/Users/Hp/Desktop/my python programs/3_test.txt",'r')
data_list=data_file.readlines()
data_file.close()
#modification fo training data
for line in data_list:
    all_val_train=line.split(',')
    inputs=np.asfarray(all_val_train[1:])
    inputs=inputs/255.0*0.99+0.01
    targets=np.zeros(out_nodes)+0.01
    targets[int(all_val_train[0])]=0.99
    n.train(inputs,targets)
    pass
np.savez_compressed("Data",n.ih,n.ho)
A=np.load("Data.npz")
print(A['arr_0'])



