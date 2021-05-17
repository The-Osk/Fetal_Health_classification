from logging import log
from os import write
from joblib import dump, load
import pandas as pd
import numpy as np
from scipy.sparse import data
import streamlit as st
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



testSet = pd.read_csv("TestSet.csv")
testSet = testSet.drop(["fetal_health"], axis = 1)
clf = load('VCHard_model.joblib')

user_inputs = {}
zeroValue = 0.00



cases = {1: "Normal" , 2: "Suspect", 3: "Pathological"}

class Doctor:
    def __init__(self):
        patient_id = st.number_input("patient ID: ", min_value=0, step = 1)
        if st.button('Show & Predict'):
            if testSet.loc[0].size > patient_id:
                patientSearchedFor = testSet.loc[patient_id]
                st.write(patientSearchedFor)

                col_names = list(patientSearchedFor)
                X_df = scaler.fit_transform([patientSearchedFor])
                X_df = pd.DataFrame(X_df, columns = col_names)

                y = clf.predict(X_df)
                st.write(f'The patient is {cases[y[0]]}')
                

                col1, col2= st.beta_columns(2)
                with col1:
                    if st.button('Agree with the result'):
                        st.write("Databases Has been updated")
                with col2:
                    if st.button('Disagree with the result'):
                        st.write("Databases Has been updated")

            else:
                st.write("Wrong patient ID")
    


class DataEntry:
    def __init__(self):
        user_inputs = {}
        st.write("Enter the patient Data: ")
        st.number_input("Patient ID: ", min_value=0, step = 1)

        col1, col2, col3= st.beta_columns(3)
        with col1:
            user_inputs["baseline value"] = [st.number_input("baseline value: ", min_value=0, step = 1)]

            user_inputs["accelerations"] = [st.number_input("accelerations: ", min_value=0.000, step = 0.001, format= "%.3f") ]

            user_inputs["fetal_movement"] = [st.number_input("fetal_movement: ", min_value=0.000, step = 0.001, format= "%.3f")]

            user_inputs["uterine_contractions"] = [st.number_input("uterine_contractions: ", min_value=0.000, step = 0.001, format= "%.3f")]

            user_inputs["light_decelerations"] = [st.number_input("light_decelerations: ", min_value=0.0000, step = 0.001, format= "%.3f")]

            user_inputs["severe_decelerations"] = [st.number_input("severe_decelerations: ", min_value=0.000, step = 0.01)]

            user_inputs["prolongued_decelerations"] = [st.number_input("prolongued_decelerations: ", min_value=0.000, step = 0.001, format= "%.3f")]

            with col2:
                user_inputs["abnormal_short_term_variability"] = [st.number_input("abnormal_short_term_variability: ", min_value=0, step = 1)]


                user_inputs["mean_value_of_short_term_variability"] = [st.number_input("mean_value_of_short_term_variability: ", min_value=0.0, step = 0.1, format= "%.1f")]

                user_inputs["percentage_of_time_with_abnormal_long_term_variability"] = [st.number_input("percentage_of_time_with_abnormal_long_term_variability: ", min_value=0, step = 1)]

                user_inputs["mean_value_of_long_term_variability"] = [st.number_input("mean_value_of_long_term_variability: ", min_value=0.0, step = 0.1, format= "%.1f")]


                user_inputs["histogram_width"] = [st.number_input("histogram_width: ", min_value=0, step = 1)]

                user_inputs["histogram_min"] = [st.number_input("histogram_min: ", min_value=0, step = 1)]

                user_inputs["histogram_max"] = [st.number_input("histogram_max: ", min_value=0, step = 1)]

            with col3:

                user_inputs["histogram_number_of_peaks"] = [st.number_input("histogram_number_of_peaks: ", min_value=0, step = 1)]

                user_inputs["histogram_number_of_zeroes"] = [st.number_input("histogram_number_of_zeroes: ", min_value=0, step = 1)]

                user_inputs["histogram_mode"] = [st.number_input("histogram_mode: ", min_value=0, step = 1)]

                user_inputs["histogram_mean"] =[ st.number_input("histogram_mean: ", min_value=0, step = 1)]

                user_inputs["histogram_median"] = [st.number_input("histogram_median: ", min_value=0, step = 1)]

                user_inputs["histogram_variance"] = [st.number_input("histogram_variance: ", min_value=0, step = 1)]

                user_inputs["histogram_tendency"] = [st.number_input("histogram_tendency: ", min_value=0, step = 1)]

        if st.button("Save Data"):
            st.write("Patient Data Saved")



class DataScientist:
    def __init__(self):
        operation = st.radio("Operation ", ("Enter Data", "Review Data"))

        if operation == "Enter Data":
            dataSource = st.radio("How are you entering the data: ", ("From a CSV file", "Enter manually"))

            if dataSource == "From a CSV file":
                CSVfile = st.file_uploader("Upload data file", type=None, accept_multiple_files=False, key=None, help=None)
                if CSVfile:
                    df = pd.read_csv(CSVfile)
                    y = df['fetal_health']
                    df = df.drop(["fetal_health"], axis = 1)
                    st.write(df)    
                    col_names = list(df)
                    X_df = scaler.fit_transform(df)
                    X_df = pd.DataFrame(X_df, columns = col_names)
                    y = clf.predict(X_df)
                    

                    if st.button('Predict CSV'):
                        predictionY = clf.predict(X_df)
                        acc =accuracy_score(y, predictionY)
                        #st.write(acc)
                        for i,j in enumerate(y):
                            st.write(f'Patient {i} is {cases[y[i]]}')


            else:

                col1, col2, col3= st.beta_columns(3)
                with col1:
                    user_inputs["baseline value"] = [st.number_input("baseline value: ", min_value=0, step = 1)]

                    user_inputs["accelerations"] = [st.number_input("accelerations: ", min_value=0.000, step = 0.001, format= "%.3f") ]

                    user_inputs["fetal_movement"] = [st.number_input("fetal_movement: ", min_value=0.000, step = 0.001, format= "%.3f")]

                    user_inputs["uterine_contractions"] = [st.number_input("uterine_contractions: ", min_value=0.000, step = 0.001, format= "%.3f")]

                    user_inputs["light_decelerations"] = [st.number_input("light_decelerations: ", min_value=0.0000, step = 0.001, format= "%.3f")]

                    user_inputs["severe_decelerations"] = [st.number_input("severe_decelerations: ", min_value=0.000, step = 0.01)]

                    user_inputs["prolongued_decelerations"] = [st.number_input("prolongued_decelerations: ", min_value=0.000, step = 0.001, format= "%.3f")]

                    with col2:
                        user_inputs["abnormal_short_term_variability"] = [st.number_input("abnormal_short_term_variability: ", min_value=0, step = 1)]


                        user_inputs["mean_value_of_short_term_variability"] = [st.number_input("mean_value_of_short_term_variability: ", min_value=0.0, step = 0.1, format= "%.1f")]

                        user_inputs["percentage_of_time_with_abnormal_long_term_variability"] = [st.number_input("percentage_of_time_with_abnormal_long_term_variability: ", min_value=0, step = 1)]

                        user_inputs["mean_value_of_long_term_variability"] = [st.number_input("mean_value_of_long_term_variability: ", min_value=0.0, step = 0.1, format= "%.1f")]


                        user_inputs["histogram_width"] = [st.number_input("histogram_width: ", min_value=0, step = 1)]

                        user_inputs["histogram_min"] = [st.number_input("histogram_min: ", min_value=0, step = 1)]

                        user_inputs["histogram_max"] = [st.number_input("histogram_max: ", min_value=0, step = 1)]

                    with col3:

                        user_inputs["histogram_number_of_peaks"] = [st.number_input("histogram_number_of_peaks: ", min_value=0, step = 1)]

                        user_inputs["histogram_number_of_zeroes"] = [st.number_input("histogram_number_of_zeroes: ", min_value=0, step = 1)]

                        user_inputs["histogram_mode"] = [st.number_input("histogram_mode: ", min_value=0, step = 1)]

                        user_inputs["histogram_mean"] =[ st.number_input("histogram_mean: ", min_value=0, step = 1)]

                        user_inputs["histogram_median"] = [st.number_input("histogram_median: ", min_value=0, step = 1)]

                        user_inputs["histogram_variance"] = [st.number_input("histogram_variance: ", min_value=0, step = 1)]

                        user_inputs["histogram_tendency"] = [st.number_input("histogram_tendency: ", min_value=0, step = 1)]

                inputs_pd = pd.DataFrame(data = user_inputs)

                if st.button('Predict'):
                    col_names = list(inputs_pd)
                    X_df = scaler.fit_transform(inputs_pd)
                    X_df = pd.DataFrame(X_df, columns = col_names)
                    y = clf.predict([inputs_pd.loc[0]])
                    st.write(f'The patient is {cases[y[0]]}')
                    zeroValue = 0.00


        else:
            patient_id = st.number_input("patient ID: ", min_value=0, step = 1)
            if st.button('Show & Predict'):
                if testSet.loc[0].size > patient_id:
                    patientSearchedFor = testSet.loc[patient_id]
                    st.write(patientSearchedFor)

                    col_names = list(patientSearchedFor)
                    X_df = scaler.fit_transform([patientSearchedFor])
                    X_df = pd.DataFrame(X_df, columns = col_names)
                    y = clf.predict([patientSearchedFor])
                    st.write(f'The patient is {cases[y[0]]}')
                    

                    col1, col2= st.beta_columns(2)
                    with col1:
                        st.write("Doctor Agreed with the model Prediction")

                else:
                    st.write("Wrong patient ID")

class Login:
    def __init__(self):
        logedin = True
        usernameInput = st.empty()
        usernamePassword = st.empty()
        loginBtn = st.empty()
        if logedin:
            x = st.sidebar.radio("What type of user are you: ", ("Doctor", "DataEntry", "DataScientist"))
            if x == "Doctor":
                d = Doctor()
            elif x == "DataEntry":
                d = DataEntry()
            else: d = DataScientist()
        else:
            username = usernameInput.text_input("Enter Username: ", "osama")
            password = usernamePassword.text_input("Enter Password: ", "3", type="password")
            if loginBtn.button("Login"):
                if username == "osama" and password == "3":
                    logedin = True
                    x = st.sidebar.radio("What type of user are you: ", ("Doctor", "DataEntry", "DataScientist"))
                    if x == "Doctor":
                        d = Doctor()
                    elif x == "DataEntry":
                        d = DataEntry()
                    else: d = DataScientist()
                    usernameInput.empty()
                    usernamePassword.empty()
                    loginBtn.empty()
                else:
                    st.write("invalid user or password")


#user_inputs["fetal_health"] = st.number_input("fetal_health: ", min_value=0.00, step = 0.01)



logedin = False
i = 0
st.set_page_config(page_title="Fetal Health Prediction System", page_icon="🧊",layout="wide", initial_sidebar_state="expanded")
st.write(""" 
# Fetal Health Prediction System
""")

if not logedin:
    l = Login()
    logedin = True


logedin = True




