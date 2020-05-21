import matplotlib.pyplot as plt
import NN
import numpy as np
def Training():
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
    pass
def Query():
    scorecard=[]
    data_file=open("C:/Users/Hp/Desktop/my python programs/3_test.txt",'r')
    data_list=data_file.readlines()
    data_file.close()
    #modification fo training data
    for line in data_list:
        all_val_ask=line.split(',')
        inputs=np.asfarray(all_val_ask[1:])
        correct=all_val_ask[0]
        print("correct: ",correct)
        inputs=inputs/255.0*0.99+0.01
        output=n.query(inputs)
        netanswer=np.argmax(output)
        print("net answer: ",netanswer)
        if (int(correct)==netanswer):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
    i=0
    sum=0
    for record in scorecard:
        i+=1
        sum+=record
        pass
    rate=sum/float(i)*100
    print(scorecard,"correctness: ",rate,"%")
    

#main unformation
in_nodes=784
h_nodes=100
out_nodes=10
learn_rate=0.3
#NN object
n=NN.NN()
n._init_(in_nodes,h_nodes,out_nodes,learn_rate)
print("NN is ready to work\nFor exit press ENTER\ninput>>")
order=input()
while (order!=""):
    if (order=="training"):
        Training()
        print("Done.")
    elif (order=="query"):
       Query()
       print("Done.")
    else:
        print("wrong input.Repeat")
    print("\ninput>>")
    order=input()
    pass

   

   



