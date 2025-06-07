# Final Project phase 2
#   Program to implement k-means clustering
#   Date: 11/18/2022
#   Author: Sangzun Park

import pandas as pd
import numpy as np
import random as rd
import math

#euclidian distance calculating
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

#assign class
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
    #dataframe slicing
    ndata = data.drop(cluster)
    #calculate mean
    mu = []
    for i in range(2,11):
        result = ndata["A"+str(i)].sum() / len(ndata)
        result = result
        mu.append(result)
    return mu
         
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

    print("Randomly selected row",centroids[0],"for centroid mu_2.\n\nInitial centroid mu_2:")
    print(data.iloc[centroids[0], 1:10])
    print("")
    print("Randomly selected row",centroids[1],"for centroid mu_4.\n\nInitial centroid mu_4:")
    print(data.iloc[centroids[1], 1:10])
    print("")
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
            print("Program ended after",iterations,"iterations.")
            print("")
            final_clusters.append(new_mu_2)
            final_clusters.append(new_mu_4)
                
            #create new data dataframe 
            cluster_columns = ["A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
            centroid_df = pd.DataFrame(final_clusters, columns = cluster_columns)
            row, col = centroid_df.shape

            #print Final statements
            print("Final centroid for mu_2:")
            print(centroid_df.iloc[0].to_string(dtype=True))
            print("")
            print("Final centroid for mu_4:")
            print(centroid_df.iloc[1].to_string(dtype=True))
            print("")
            print("Final cluster assignment: \n")
            print(data[["Scn", "Class", "Predicted_Class"]].head(21))         
            break    
        else:
            mu_2 = new_mu_2
            mu_4 = new_mu_4     
               
main()
        