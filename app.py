import pandas as pd
import streamlit as st
import requests

df = pd.read_csv("./datamodel.csv")

def run():
    st.title("Default Risk Prediction")
    Ext2 = st.number_input("Ext Source 2")
    Ext3 = st.number_input("Ext Source 3")
    YearLastPhoneChange = st.number_input("Year Last Phone Change")
    RegRateCity = st.selectbox("Region Rate City", df.REGION_RATING_CLIENT_W_CITY.unique())
    YearsEmp = st.number_input("Years Employed",format = "%d", step = 1)
    RegRateCli = st.selectbox("Region Rate Client",
                            df.REGION_RATING_CLIENT.unique())
    AmtGd = st.number_input("Amount Good Price", format="%d", step=1)
    AmtBlcMean = st.number_input("Amount Balance Mean")
    AmtCredit = st.number_input("Amt Credit", format="%d", step=1)
    DaysBirth = st.number_input("Days Birth", format="%d", step=1)
    FloMaxMode = st.number_input("Floor Maximum Mode")
    EdType = st.selectbox(
        "Education Type", df.NAME_EDUCATION_TYPE.unique())
    CodeGen = st.selectbox(
        "Gender", df.CODE_GENDER.unique())
    IncType = st.selectbox(
        "Income Type", df.NAME_INCOME_TYPE.unique())
    OccType = st.selectbox(
        "Occupation Type", df.OCCUPATION_TYPE.unique())
    OrgType = st.selectbox(
        "Organization Type", df.ORGANIZATION_TYPE.unique())
    OwnCar = st.selectbox(
        "Own Car", df.FLAG_OWN_CAR.unique())
    data = {
        "Ext2": float(Ext2),
        "Ext3": float(Ext3),
        "YearLastPhoneChange": float(YearLastPhoneChange),
        "RegRateCity": int(RegRateCity),
        "YearsEmp": float(YearsEmp),
        "RegRateCli": int(RegRateCli),
        "AmtGd": float(AmtGd),
        "AmtBlcMean": float(AmtBlcMean),
        "AmtCredit": float(AmtCredit),
        "DaysBirth": int(DaysBirth),
        "FloMaxMode": float(FloMaxMode),
        "EdType": str(EdType),
        "CodeGen": str(CodeGen),
        "IncType": str(IncType),
        "OccType": str(OccType),
        "OrgType": str(OrgType),
        "OwnCar": str(OwnCar)
        }
    if st.button("Predict"):
        response = requests.post(
            "http://ec2-108-137-94-218.ap-southeast-3.compute.amazonaws.com:8090/predict", json=data)
        prediction = response.text
        if prediction == "0":
            st.caption(f"The prediction from model: {prediction}")
            st.success("The model predict this client is default risk")
        else:
            st.success("The model predict this client is default risk")
            st.caption(f"The prediction from model: {prediction}")


if __name__ == '__main__':
    run()
