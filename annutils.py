from tensorflow import keras
import joblib
import pandas as pd
import numpy as np

class ANNModel:
    '''
    model consists of the model file + the scaling of inputs and outputs
    '''
    def __init__(self,model,xscaler,yscaler):
        self.model=model
        self.xscaler=xscaler
        self.yscaler=yscaler
    def predict(self, dfin,columns=['prediction'],ndays=8,window_size=11,nwindows=10):
        return predict(self.model,dfin,self.xscaler,self.yscaler,columns=columns,ndays=ndays,window_size=window_size,nwindows=nwindows)

class myscaler():
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = -float('inf')
    
    def fit_transform(self, data):
        data[data==-2]=float('nan')
        self.min_val = data.min()
        self.max_val = data.max()

    def update(self, other_scaler):
        self.min_val = np.minimum(self.min_val,other_scaler.min_val)
        self.max_val = np.maximum(self.max_val,other_scaler.max_val)

    def transform(self, data):
        return (data - self.min_val) * 1.0 / (self.max_val - self.min_val)
    
    def inverse_transform(self, data):
        if type(data)==np.ndarray:
            max_val = self.max_val.to_numpy().reshape(1,-1)
            min_val = self.min_val.to_numpy().reshape(1,-1)
            return data * (max_val - min_val) + min_val
        else:
            return data * (self.max_val - self.min_val) + self.min_val
    
def create_antecedent_inputs(df,ndays=8,window_size=11,nwindows=10):
    '''
    create data frame for CALSIM ANN input
    Each column of the input dataframe is appended by :-
    * input from each day going back to 7 days (current day + 7 days) = 8 new columns for each input
    * 11 day average input for 10 non-overlapping 11 day periods, starting from the 8th day = 10 new columns for each input

    Returns
    -------
    A dataframe with input columns = (8 daily shifted and 10 average shifted) for each input column

    '''
    arr1=[df.shift(n) for n in range(ndays)]
    dfr=df.rolling(str(window_size)+'D',min_periods=window_size).mean()
    arr2=[dfr.shift(periods=(window_size*n+ndays),freq='D') for n in range(nwindows)]
    df_x=pd.concat(arr1+arr2,axis=1).dropna()# nsamples, nfeatures
    return df_x

def predict(model,dfx,xscaler,yscaler,columns=['prediction'],ndays=8,window_size=11,nwindows=10):
    dfx=pd.DataFrame(xscaler.transform(dfx),dfx.index,columns=dfx.columns)
    xx=create_antecedent_inputs(dfx,ndays=ndays,window_size=window_size,nwindows=nwindows)
    oindex=xx.index
    yyp=model.predict(xx)
    dfp=pd.DataFrame(yscaler.inverse_transform(yyp),index=oindex,columns=columns)
    return dfp

def load_model(location,custom_objects):
    '''
    load model (ANNModel) which consists of model (Keras) and scalers loaded from two files
    '''
    model=keras.models.load_model('%s.h5'%location,custom_objects=custom_objects)    
    xscaler,yscaler=joblib.load('%s_xyscaler.dump'%location)

    return ANNModel(model,xscaler,yscaler)