import numpy as np
import pandas as pd
import seaborn as sns 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import random
import math

# remove some meaningless columns and merge all table
def remove_columns_and_merge_tables():
    demographic_df = pd.read_csv("html2021final/demographics.csv", index_col="Customer ID")
    demographic_df.drop(columns=["Count"], inplace=True)

    location_df = pd.read_csv("html2021final/location.csv", index_col="Customer ID")
    location_df.drop(columns=["Count", "Country","State","City", "Lat Long"], inplace=True)

    population_df = pd.read_csv("html2021final/population.csv", index_col="Zip Code")
    location_df["Zip Code"] = location_df["Zip Code"].map(population_df["Population"])
    location_df.rename(columns={'Zip Code': 'Population'}, inplace=True)

    satisfaction_df = pd.read_csv("html2021final/satisfaction.csv", index_col="Customer ID")

    services_df = pd.read_csv("html2021final/services.csv", index_col="Customer ID")
    services_df.drop(columns=["Count", "Quarter"], inplace=True)


    data = pd.merge(demographic_df, location_df, left_index=True, right_index=True, how="outer")
    data = pd.merge(data, satisfaction_df, left_index=True, right_index=True, how="outer")
    data = pd.merge(data, services_df, left_index=True, right_index=True, how="outer")
    return data


def extract_train_test(data):
    status_df = pd.read_csv("html2021final/status.csv", index_col="Customer ID")
    class2idx = {
        "No Churn":0,
        "Competitor":1,
        "Dissatisfaction":2,
        "Attitude":3,
        "Price":4,
        "Other":5,
    }

    status_df['Churn Category'].replace(class2idx, inplace=True)
    train_df = pd.read_csv("html2021final/Train_IDs.csv", index_col="Customer ID")
    train_data = pd.merge(data, status_df, left_index=True, right_index=True, how="inner")
    train_data = pd.merge(train_data, train_df, left_index=True, right_index=True)
    test_df = pd.read_csv("html2021final/Test_IDs.csv", index_col="Customer ID")
    test_data = pd.merge(data, test_df, left_index=True, right_index=True)
    return train_data, test_data


def data_preprocessing(data, statis):
    # drop rows missing too many columns ( only train_data )
    if statis == {}:
        data.dropna(thresh=15, inplace=True)
    
    #  ------ handle_missing_data ------
    gender = statis.get("gender", data["Gender"].mode()[0])
    statis["gender"] = gender
    data["Gender"].fillna(gender, inplace=True)
    
    def guess_age(row, under30, normal, upper65):
        if np.isnan(row["Age"]):
            if row["Under 30"] == "Yes":
                return under30
            elif row["Senior Citizen"] == "Yes":
                return upper65
            else:
                return normal
        else:
            return row["Age"]
        
    under30 = statis.get("under30", data["Age"].loc[data["Age"] < 30].median())
    statis["under30"] = under30
    normal = statis.get("normal", data["Age"].median())
    statis["normal"] = normal
    upper65 = statis.get("upper65", data["Age"].loc[data["Age"] >= 65].median())
    statis["upper65"] = upper65
    data["Age"] = data.apply(guess_age, axis=1, args=(under30, normal, upper65))
    data.drop(columns=["Under 30", "Senior Citizen"], inplace=True)
    
    married = statis.get("married", data["Married"].mode()[0])
    statis["married"] = married
    data["Married"].fillna(married, inplace=True)
    
    def guess_dependents(row, normal):
        if np.isnan(row["Number of Dependents"]):
            if row["Dependents"] == "No":
                return 0.0
            else:
                return normal
        else:
            return row["Number of Dependents"]
    
    number_of_dependents = statis.get("number_of_dependents", data["Number of Dependents"].loc[data["Number of Dependents"]>0].mode()[0])
    statis["number_of_dependents"] = number_of_dependents
    data["Number of Dependents"] = data.apply(guess_dependents, axis=1, args=(number_of_dependents, ))
    data.drop(columns=["Dependents"], inplace=True)
    
    
    population = statis.get("population", data["Population"].mode()[0])
    statis["population"] = population
    latitude = statis.get("latitude", data["Latitude"].mode()[0])
    statis["latitude"] = latitude
    longitude = statis.get("longitude", data["Longitude"].mode()[0])
    statis["longitude"] = longitude
    data["Population"].fillna(population, inplace=True)
    data["Latitude"].fillna(latitude, inplace=True)
    data["Longitude"].fillna(longitude, inplace=True)
    
    satisfaction_score = statis.get("satifaction_score", data["Satisfaction Score"].mode()[0])
    statis["satisfaction_score"] = satisfaction_score
    data["Satisfaction Score"].fillna(satisfaction_score, inplace=True)
    
    def guess_referrals(row, normal):
        if np.isnan(row["Number of Referrals"]):
            if row["Referred a Friend"] == "No":
                return 0.0
            else:
                return normal
        else:
            return row["Number of Referrals"]
    
    number_of_referrals = statis.get("number_of_referrals", data["Number of Referrals"].loc[data["Number of Referrals"]>0].mode()[0])
    statis["number_of_referrals"] = number_of_referrals
    data["Number of Referrals"] = data.apply(guess_referrals, axis=1, args=(number_of_referrals, ))
    data.drop(columns=["Referred a Friend"], inplace=True)
    
    tenure_in_months = statis.get("tenure_in_months", data["Tenure in Months"].mode()[0])
    statis["tenure_in_months"] = tenure_in_months
    data["Tenure in Months"].fillna(tenure_in_months, inplace=True)
    
    offer = statis.get("offer", data["Offer"].mode()[0])
    statis["offer"] = offer
    data["Offer"].fillna(offer, inplace=True)
    
    phone_service = statis.get("phone_service", data["Phone Service"].mode()[0])
    statis["phone_service"] = phone_service
    data["Phone Service"].fillna(phone_service, inplace=True)
    
    
    charges = statis.get("charges", data["Avg Monthly Long Distance Charges"].mode()[0])
    statis["charges"] = charges
    data["Avg Monthly Long Distance Charges"].fillna(charges, inplace=True)
    
    multiple_lines = statis.get("multiple_lines", data["Multiple Lines"].mode()[0])
    statis["multiple_lines"] = multiple_lines
    data["Multiple Lines"].fillna(multiple_lines, inplace=True)
    
    def guess_internet_service(row, normal):
        if pd.isna(row["Internet Service"]):
            if (not pd.isna(row["Internet Type"]) and row["Internet Type"] != "None") or                 row["Avg Monthly GB Download"] > 0 or                 row["Online Security"] == "Yes" or                 row["Online Backup"] == "Yes" or                 row["Device Protection Plan"] == "Yes" or                 row["Premium Tech Support"] == "Yes" or                 row["Streaming TV"] == "Yes" or                 row["Streaming Movies"] == "Yes" or                 row["Streaming Music"] == "Yes" or                 row["Unlimited Data"] == "Yes":
                return "Yes"
            else:
                return normal
        else:
            return row["Internet Service"]
    
    internet_service = statis.get("internet_service", data["Internet Service"].mode()[0])
    statis["internet_service"] = internet_service
    data["Internet Service"] = data.apply(guess_internet_service, axis=1, args=(internet_service, ))

    def guess_internet_services(row, service_name, normal):
        if pd.isna(row[service_name]):
            if row["Internet Service"] == "No":
                if service_name == "Internet Type":
                    return "None"
                elif service_name == "Avg Monthly GB Download":
                    return 0.0
                else:
                    return "No"
            else:
                return normal
        else:
            return row[service_name]
        
    internet_list = ["Internet Type", "Avg Monthly GB Download", "Online Security", "Online Backup",
                     "Device Protection Plan", 'Premium Tech Support', "Streaming TV", "Streaming Movies",
                     "Streaming Music", "Unlimited Data"]
    
    for service in internet_list:
        if service == "Avg Monthly GB Download":
            internet_type = statis.get(service, data[service].loc[data[service]>0].mode()[0])
            statis[service] = internet_type
        else:
            internet_type = statis.get(service, data[service].mode()[0])
            statis[service] = internet_type
        data[service] = data.apply(guess_internet_services, axis=1, args=(service, internet_type, ))
    
    contract = statis.get("contract", data["Contract"].mode()[0])
    statis["contract"] = contract
    data["Contract"].fillna(contract, inplace=True)
    
    paperless_billing = statis.get("paperless_billing", data["Paperless Billing"].mode()[0])
    statis["paperless_billing"] = paperless_billing
    data["Paperless Billing"].fillna(paperless_billing, inplace=True)
    
    payment_method = statis.get("payment_method", data["Payment Method"].mode()[0])
    statis["payment_method"] = payment_method
    data["Payment Method"].fillna(payment_method, inplace=True)
    
    monthly_charge = statis.get("monthly_charge", data["Monthly Charge"].mode()[0])
    statis["monthly_charge"] = monthly_charge
    data["Monthly Charge"].fillna(monthly_charge, inplace=True)
    
    total_charges = statis.get("total_charges", data["Total Charges"].mode()[0])
    statis["total_charges"] = total_charges
    data["Total Charges"].fillna(total_charges, inplace=True)
    
    total_refunds = statis.get("total_refunds", data["Total Refunds"].mode()[0])
    statis["total_refunds"] = total_refunds
    data["Total Refunds"].fillna(total_refunds, inplace=True)
    
    extra_data_charges = statis.get("extra_data_charges", data["Total Extra Data Charges"].mode()[0])
    statis["extra_data_charges"] = extra_data_charges
    data["Total Extra Data Charges"].fillna(extra_data_charges, inplace=True)
    
    long_distance_charges = statis.get("long_distance_charges", data["Total Long Distance Charges"].mode()[0])
    statis["long_distance_charges"] = long_distance_charges
    data["Total Long Distance Charges"].fillna(long_distance_charges, inplace=True)
    
    total_revenue = statis.get("total_revenue", data["Total Revenue"].mode()[0])
    statis["total_revenue"] = total_revenue
    data["Total Revenue"].fillna(total_revenue, inplace=True)
       
    # transform boolean field to 0/1
    for col in data:
        if data[col].iloc[0] == "Yes" or data[col].iloc[0] == "No":
            data[col] = data[col].map({"Yes":1.0, "No":0.0}).astype(float)
    data["Gender"] = data["Gender"].map({"Male":1.0, "Female":0.0}).astype(float)
    
    # one-hot encoding
    return pd.get_dummies(data).astype(float)

data = remove_columns_and_merge_tables()
train_data, test_data = extract_train_test(data)
#train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data.loc[:, ["Churn Category"]])

statis = {}
train_data = data_preprocessing(train_data, statis)
#val_data = data_preprocessing(val_data, statis)
test_data = data_preprocessing(test_data, statis)

train_data.to_csv('preprocessed_train_data.csv')


