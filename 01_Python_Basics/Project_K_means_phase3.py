# Final Project phase 3
#   Program to implement k-means clustering
#   Date: 11/25/2022
#   Author: Sangzun Park

import pandas as pd
import numpy as np
import random as rd
import math

#Euclidian distance calculating
def distance(data,mu):
    compute_mu =[]
    temp = 0
    for i in range(len(data)):
        d = list(data.iloc[i,1:10])
        for j in range(len(mu)):
            val = ((d[j] - mu[j])**2)
            temp += val
        temp = math.sqrt(temp)
        compute_mu.append(temp)
        temp = 0
    return compute_mu

#Assign class
def predicted_class(data,mu_2,mu_4):
    mu_2_cluster = []
    mu_4_cluster = []
    predicted_cluster=[]
    
    for i in range(len(mu_2)):  
        if mu_2[i] > mu_4[i]:
            mu_4_cluster.append(i)
            mu_4[i] = 4
            predicted_cluster.append(mu_4[i])
        else:
            mu_2_cluster.append(i)
            mu_2[i] = 2
            predicted_cluster.append(mu_2[i])   
    data["Predicted_Class"] = predicted_cluster
 
    return mu_2_cluster,mu_4_cluster

def compute_mean(data,cluster):
    #Dataframe slicing
    ndata = data.drop(cluster)
    #Calculate mean
    mu = []
    for i in range(2,11):
        result = ndata["A"+str(i)].sum() / len(ndata)
        result = result
        mu.append(result)
    return mu

def check_errors(ndata):
    pclass_2 = []
    pclass_4 = []
    class_all = pclass_2 + pclass_4
    error_24 = []
    error_42 = []
    error_all = error_24 + error_42
    
    pdata = ndata["Predicted_Class"]
    for i in range(len(ndata)):
        if pdata[i] == 2:
            pclass_2.append(ndata.iloc[i])
            
        if i == 4:
            pclass_4.append(i)
    pclass_2 = pd.DataFrame(pclass_2)
    return pclass_2
    
    
def main():
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    df = pd.read_csv("breast-cancer-wisconsin.data", na_values = "?", names = col)
    A7 = df["A7"]   
    mean_A7 = np.sum(A7) / len(A7)
    data = round(df.fillna(mean_A7),1)

    #Initial step
    centroids = rd.sample(range(0,len(data)),2)
    mu_2 = list(data.iloc[centroids[0], 1:10])
    mu_4 = list(data.iloc[centroids[1], 1:10]) 

    #Assign step
    dist_mu2 = distance(data,mu_2)
    dist_mu4 = distance(data,mu_4)

    cluster_result = predicted_class(data,dist_mu2,dist_mu4)
    cluster_2=cluster_result[0]
    cluster_4=cluster_result[1]

    new_mu_2 = compute_mean(data,cluster_4)
    new_mu_4 = compute_mean(data,cluster_2)
    
    #Recompute step
    iterations = 0
    mu_2 = []
    mu_4 = []
    while iterations <= 50:       
        iterations += 1
        final_clusters = []
        
        dist_mu2 = distance(data,new_mu_2)
        dist_mu4 = distance(data,new_mu_4)
        
        cluster_result = predicted_class(data,dist_mu2,dist_mu4)
        cluster_2=cluster_result[0]
        cluster_4=cluster_result[1]

        new_mu_2 = compute_mean(data,cluster_4)
        new_mu_4 = compute_mean(data,cluster_2)
        
        if new_mu_2 == mu_2 and new_mu_4 == mu_4:
            final_clusters.append(new_mu_2)
            final_clusters.append(new_mu_4)
                
            #Create new data dataframe 
            cluster_columns = ["A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
            centroid_df = pd.DataFrame(final_clusters, columns = cluster_columns)
            row, col = centroid_df.shape
            
            #Phase 3 
            pclass_2 = (data.loc[data["Class"]==2])
            pclass_4 = (data.loc[data["Class"]==4])
            class_all = pd.concat([pclass_2,pclass_4])
            class_all = class_all.sort_index()
            
            error_24 = (data.loc[(data["Class"]==2)&(data["Predicted_Class"]==4)])
            error_42 = (data.loc[(data["Class"]==4)&(data["Predicted_Class"]==2)])
            error_all = pd.concat([error_24,error_42])
            error_all = error_all.sort_index()
            
            error_B = (len(error_24)/len(pclass_2))*100
            error_M = (len(error_42)/len(pclass_4))*100
            error_T = (len(error_all)/len(class_all))*100
            
            if error_T < 50:
                print("Total errors:\t\t\t\t\t ",round(error_T,1),"\n-End-")
                break
                       
            if error_T > 50:                             
                print("Total errors:\t\t\t\t\t ",round(error_T,1),"%\n"
                      "Clusters are swapped!\nSwapping Predicted_Class\n")
                
                data.loc[data["Predicted_Class"]==2,"Predicted_Class"] =0
                data.loc[data["Predicted_Class"]==4,"Predicted_Class"] =2
                data.loc[data["Predicted_Class"]==0,"Predicted_Class"] =4
                
                pclass_2 = (data.loc[data["Class"]==2])
                pclass_4 = (data.loc[data["Class"]==4])
                class_all = pd.concat([pclass_2,pclass_4])
                class_all = class_all.sort_index()
                
                error_24 = (data.loc[(data["Class"]==2)&(data["Predicted_Class"]==4)])
                error_42 = (data.loc[(data["Class"]==4)&(data["Predicted_Class"]==2)])
                error_all = pd.concat([error_24,error_42])
                error_all = error_all.sort_index()
                
                n_error_B = (len(error_24)/len(pclass_2))*100
                n_error_M = (len(error_42)/len(pclass_4))*100
                n_error_T = (len(error_all)/len(class_all))*100
                
                print("Data points in Predicted Class 2:",len(pclass_2))
                print("Data points in Predicted Class 4:",len(pclass_4),"\n")
                
                print("Error data points, Predicted Class 2:\n")
                print(error_24[["Scn", "Class", "Predicted_Class"]],"\n")
                print("Error data points, Predicted Class 4:\n")
                print(error_42[["Scn", "Class", "Predicted_Class"]],"\n")
                
                print("Number of all data points:\t\t  ",len(class_all))
                print("Number of error data points:\t\t  ",len(error_all))
                print("Error rate for class 2:\t\t\t  ",round(n_error_B,1),"%")
                print("Error rate for class 4:\t\t\t  ",round(n_error_M,1),"%")
                print("Total error rate:\t\t\t\t  ",round(n_error_T,1),"%")
                                   
            break            
        else:
            mu_2 = new_mu_2
            mu_4 = new_mu_4              
 
main()

