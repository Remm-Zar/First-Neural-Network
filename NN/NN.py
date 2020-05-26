import numpy as np
import scipy.special as scp  
class NN:
    def _init_(self,innodes,hnodes,outnodes,lr):
        self.inodes=innodes
        self.hnodes=hnodes
        self.onodes=outnodes
        self.lr=lr
        #data=np.load("Data.npz")
        self.ih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #data['arr_0']
        #np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.ho=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #data['arr_1']
        #np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.act_func=lambda x:scp.expit(x)
        pass
    def Set_learn_rate(self,new_lr):
        self.lr=new_lr
        pass
    def Set_hnodes(self,new_hnodes):
        self.hnodes=new_hnodes
        self.ih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.ho=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        pass
    def query(self,in_list):
        inputs=np.array(in_list,ndmin=2).T
        hid_in=np.dot(self.ih,inputs)
        hid_out=self.act_func(hid_in)
        final_in=np.dot(self.ho,hid_out)
        final_out=self.act_func(final_in)
        return final_out
    def train(self,in_list,tar_list):
        inputs=np.array(in_list,ndmin=2).T
        targets=np.array(tar_list,ndmin=2).T
        hid_in=np.dot(self.ih,inputs)
        hid_out=self.act_func(hid_in)
        final_in=np.dot(self.ho,hid_out)
        final_out=self.act_func(final_in)
        out_err=targets-final_out
        h_err=np.dot(self.ho.T,out_err)
        self.ho+=self.lr*np.dot(out_err*final_out*(1.0-final_out),np.transpose(hid_out))
        self.ih+=self.lr*np.dot(h_err*hid_out*(1.0-hid_out),np.transpose(inputs))
        pass

