import numpy as np
import matplotlib.pyplot as plt

def afficheTenseur(T):
    for i in range(0,T.shape[0]):
        plt.figure(i)
        plt.title(f"Y {i}")
        plt.imshow(T[i])
    plt.show()
    

def gen_contact_tensors(path, interval) :
    data=np.loadtxt(path, usecols=range(0, 3), dtype=int)
    nb_personne = data[:,1:].max() + 1
    if (data[:,0].max()-data[:,0].min())//interval==0 :
        prfd=1
    else :
        prfd=(data[:,0].max()-data[:,0].min())//interval
    contact_tensors=np.zeros((prfd,nb_personne, nb_personne))
    contact_matrix=np.zeros((nb_personne, nb_personne))
    t_ini=data[0,0]
    max_line_in_time=0
    ligne_ini=0
    nbvide=0
    for k in range(0, prfd):   
        i=max_line_in_time
        if i<data[:,0].shape[0] and data[i,0]<=t_ini+interval :
            while i<data[:,0].shape[0] and data[i,0]<=t_ini+interval :
                max_line_in_time=i
                i+=1
            max_line_in_time+=1
            act_data=data[ligne_ini:max_line_in_time, 1:]
            for j in range(0, act_data.shape[0]) :
                contact_matrix[act_data[j,0], act_data[j,1]]=1
                contact_matrix[act_data[j,1], act_data[j,0]]=1
                contact_matrix[act_data[j,1], act_data[j,1]]=1
                contact_matrix[act_data[j,0], act_data[j,0]]=1
            contact_tensors[k-nbvide,:,:]=contact_matrix
            contact_matrix=np.zeros((nb_personne, nb_personne))
            t_ini=t_ini+interval
            ligne_ini=max_line_in_time
            max_line_in_time+=1
        else :
            nbvide+=1
            t_ini=t_ini+interval
    if max_line_in_time<data[:,0].shape[0]:
        act_data=data[max_line_in_time:data[:,0].shape[0], 1:]
        for j in range(0, act_data.shape[0]) :
            contact_matrix[act_data[j,0], act_data[j,1]]=1
            contact_matrix[act_data[j,1], act_data[j,0]]=1
            contact_matrix[act_data[j,1], act_data[j,1]]=1
            contact_matrix[act_data[j,0], act_data[j,0]]=1
        contact_tensors[prfd-nbvide,:,:]=contact_matrix
        nbvide-=1
    return contact_tensors[0:prfd-nbvide,:,:]
        





