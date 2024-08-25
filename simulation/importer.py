import pandas as pd
import numpy as np
import csv

def import_smhi():
    df = pd.read_csv("SMHI_pthbv_t_2023_2024_daily_4326.csv")
    print(df)  # Process the DataFrame as needed

def import_stockholm():
    df = pd.read_csv("stockholm.csv")
    print(df.columns)
                       
    print(df[df.iloc[:, 0] == 2018])

def import_huskvarna():
    df = pd.read_csv("huskvarna.csv", sep=";")
    
    # Get data of december month of 2023.
    data = pd.DataFrame(df.iloc[range(77198, 77940)])
    
    # Get the columns of the data.
    temps = data.loc[:, "Timmedel"].dropna()
    return np.array(temps)

def import_huskvarna2022():
    df = pd.read_csv("huskvarna.csv", sep=";")
    
    # Get data of december month of 2022.
    data = pd.DataFrame(df.iloc[range(68438, 69181)])
    
    # Get the columns of the data.
    temps = data.loc[:, "Timmedel"].dropna()
    return np.array(temps)