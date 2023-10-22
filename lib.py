import numpy as np
import random
import matplotlib.pyplot as plt

def get_label(X_aug,X_min,y_min,num_nbrs):
    y_aug=[]
    for i in range(X_aug.shape[0]):
        src=X_aug[i]
        distances=np.linalg.norm(X_min-src,axis=1)
        dist_indices_sorted=np.argsort(distances)
        numerator=0
        denom=0
        for nbr_indx in range(1,num_nbrs+1):
            y_nbr=y_min[dist_indices_sorted[nbr_indx]]
            dist_nbr=distances[dist_indices_sorted[nbr_indx]]
            numerator+=(1/dist_nbr)*y_nbr
            denom+=(1/dist_nbr)
        y_src=numerator/denom
        y_aug.append(y_src)
    y_aug= np.array(y_aug)
    return y_aug
            
            
            
                

def oversample(X_min,y_min,num_nbrs,proportion_of_minority,count_to_add,alpha=0.5):
    count_of_minority=int(X_min.shape[0]*proportion_of_minority)
    # print("Count of usable minority",count_of_minority)
    X_aug=[]
    if count_of_minority<=num_nbrs:
        print("Not enough minorities")
        return X_aug,None
    dist_to_nth_nbr=[]
    for i in range(X_min.shape[0]):
        src=X_min[i]
        # print(src.shape,X_min.shape)
        distances=np.linalg.norm(X_min-src,axis=1)
        # print(distances)
        dist_indices_sorted=np.argsort(distances)
        # print(dist_indices_sorted)
        # print("distance to nth nbr",distances[dist_indices_sorted[:num_nbrs+1]])
        dist_to_nth_nbr.append(distances[dist_indices_sorted[num_nbrs]])
    # print("distance to ",num_nbrs,"nbr:",dist_to_nth_nbr)
    dist_to_nth_nbr.sort()
    # print("sorted distance to ",num_nbrs,"nbr:",dist_to_nth_nbr)    
    threshold_dist=dist_to_nth_nbr[count_of_minority]        
    # print("threshold_dist",threshold_dist)
    
    while count_to_add>0:
        for i in range(X_min.shape[0]):
            src=X_min[i]
            distances=np.linalg.norm(X_min-src,axis=1)
            # print("dists",distances)
            sorted_indices_dist=np.argsort(distances)
            # print(distances[sorted_indices_dist[num_nbrs]],"<=",threshold_dist)
            if distances[sorted_indices_dist[num_nbrs]]<=threshold_dist:
                # print("Can use this point")

                for j in range(1,num_nbrs+1):
                    nbr_index=sorted_indices_dist[j]
                    X_nbr=X_min[nbr_index]
                    fractn=random.uniform(0,alpha)
                    # print("src",list(src))
                    # print("X_nbr",list(X_nbr))
                    # print("fractn",fractn)
                   
                    new_point=src+fractn*(X_nbr-src)
                    # print('New point is',new_point)
                    src=new_point
                # print("Final new points is ",src)
                X_aug.append(src)
                count_to_add-=1
                if count_to_add==0:
                    break

    X_aug=np.array(X_aug)
    y_aug=get_label(X_aug,X_min,y_min,num_nbrs)
    # print(X_aug.shape,y_aug.shape)
    
    return X_aug,y_aug
                


def oversample_data(X_train,y_train,num_nbrs=3,proportion_of_minority=0.9):
    '''
    The main call
    '''
    hist=plt.hist(y_train,bins=50)
    avg_freq=20
    vals=hist[1]

    
    X_aug_all=[]
    y_aug_all=[]
    for i in range(len(vals)-1):
        
        freq=hist[0][i]
        # print("Freq",freq,"avg_freq",avg_freq)
        if freq<avg_freq:
            # print(vals[i],vals[i+1],freq)
            maj_indices=np.argwhere( (y_train<vals[i]) | (y_train>vals[i+1])).flatten()
            
    
            min_indices=np.argwhere( (y_train>=vals[i]) & (y_train<=vals[i+1])).flatten()        
            
            X_min=X_train[min_indices]
            if X_min.shape[0]>num_nbrs:
                X_maj=X_train[maj_indices]
                
                y_maj=y_train[maj_indices]
                y_min=y_train[min_indices]
                count_to_add=int(abs(avg_freq-X_min.shape[0]))
                # print("count_to_add",count_to_add)
                # print(X_maj.shape,X_min.shape,y_maj.shape,y_min.shape,count_to_add)        
                X_aug,y_aug=oversample(X_min,y_min,num_nbrs,proportion_of_minority,count_to_add)
                # print("aug:",y_aug,"\n","actual:",y_min)
                if len(X_aug)!=0:
                    X_aug_all.append(X_aug)
                    y_aug_all.append(y_aug)
                # break
                
            
        
        
    X_aug_all=np.concatenate(X_aug_all)
    y_aug_all=np.concatenate(y_aug_all)
    X_train_aug=np.concatenate([X_aug_all,X_train])
    y_train_aug=np.concatenate([y_aug_all,y_train])

    return X_train_aug,y_train_aug