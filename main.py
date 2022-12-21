import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from os.path import abspath


app = FastAPI(title='Default Risk Prediction: Random Forest', version='1.0')
model_path = abspath('./rf_model.pkl')
model = joblib.load(model_path)


class Data(BaseModel):
    Ext2: float
    Ext3: float
    YearLastPhoneChange: float
    RegRateCity: int
    YearsEmp: float
    RegRateCli: int
    AmtGd: float
    AmtBlcMean: float
    AmtCredit: float
    DaysBirth: int
    FloMaxMode: float
    EdType: str
    CodeGen: str
    IncType: str
    OccType: str
    OrgType: str
    OwnCar: str

@app.get("/")

@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the    application.
    """
    return {'message': 'System is healthy'}

@app.post("/predict")
def predict(data: Data):
    df = pd.DataFrame(columns=['EXT_SOURCE_2',
                            'EXT_SOURCE_3', 
                            'YEARS_LAST_PHONE_CHANGE',
                            'REGION_RATING_CLIENT_W_CITY', 
                            'YEARS_EMPLOYED',
                            'REGION_RATING_CLIENT',
                            'AMT_GOODS_PRICE',
                            'AMT_BALANCE_MEAN',
                            'AMT_CREDIT',
                            'DAYS_BIRTH',
                            'FLOORSMAX_MODE',
                            'NAME_EDUCATION_TYPE',
                            'CODE_GENDER',
                            'NAME_INCOME_TYPE',
                            'OCCUPATION_TYPE',
                            'ORGANIZATION_TYPE',
                            'FLAG_OWN_CAR'],
                 data=np.array([data.Ext2, data.Ext3, data.YearLastPhoneChange, data.RegRateCity,
                                data.YearsEmp, data.RegRateCli, data.AmtGd, data.AmtBlcMean, data.AmtCredit, data.DaysBirth, data.FloMaxMode,
                                data.EdType,data.CodeGen,data.IncType,data.OccType,data.OrgType,data.OwnCar]).reshape(1, 17))
    result = model.predict(df)
    
    return (result.tolist())[0]


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8090, reload=True)