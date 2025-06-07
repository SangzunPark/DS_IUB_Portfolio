# Final Project phase 1
#   Program to compute data statistics and plot basic graphs
#   Date: 11/10/2022
#   Author: Sangzun Park

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#set calculation functions[Mean / Median / Variance / Standard Deviation]
def mean(col):
    result = np.mean(col)
    return result
#try to get median "for this data" without function
def median(col):
    sort = col.sort_values(ignore_index=True)
    median = ((len(sort)+1)/2)
    result = sort[median]
    return float(result)

def variance(col):
    result = np.var(col)
    return result

def sd(col):
    result = np.std(col)
    return result

# Main function  
def main():
    #Read data and replace NaN data of column7 with mean value of column7
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    df = pd.read_csv("breast-cancer-wisconsin.data", na_values = "?", names = col)
    A7 = df["A7"]    
    mean_A7 = np.sum(A7) / len(A7)
    data = df.fillna(mean_A7)
    #Print attribute of each column and their histogram with functions and loop
    for i in range(2,11):
        print("Attribute A"+str(i),"----------------\n"
              "   Mean:              ",round(mean(data["A"+str(i)]),1),"\n"+
              "   Median:            ",round(median(data["A"+str(i)]),1),"\n"+
              "   Variance:          ",round(variance(data["A"+str(i)]),1),"\n"+
              "   Standard Deviation:",round(sd(data["A"+str(i)]),1),"\n")
        #Plots
        fig = plt.figure()
        sp = fig.add_subplot(1,1,1)
        sp.set_title("Histogram of attribute A"+str(i))
        sp.set_xlabel("Value of the attribute")
        sp.set_ylabel("Number of data prints")
        sp.hist(data["A"+str(i)],bins=10,color="blue",edgecolor="black",alpha=0.5)
    
main()

   