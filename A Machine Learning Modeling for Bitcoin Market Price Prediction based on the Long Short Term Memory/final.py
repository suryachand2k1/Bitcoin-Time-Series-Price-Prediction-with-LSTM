from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

main = tkinter.Tk()
main.title("Bitcoin Classification")
main.geometry("1300x1200")

def upload_data():
        global filename
        text.delete('1.0', END)
        filename = askopenfilename(initialdir = ".")
        pathlabel.config(text=filename)
        text.insert(END,"Dataset loaded\n\n")

def loaddataset():
        global data
        data = pd.read_csv(filename)
        text.insert(END,"Top 5 Rows: "+str(data.head())+"\n")
        text.insert(END,"Columns Information: "+str(data.columns)+"\n")

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def dataPreprocesSplit():
        global data
        global train,test
        global trainX,trainY,testX,testY
        global scaled,scaler
        data['Weighted Price'].replace(0, np.nan, inplace=True)
        data['Weighted Price'].fillna(method='ffill', inplace=True)
        values = data['Weighted Price'].values.reshape(-1,1)

        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        train_size = int(len(scaled) * 0.7)
        test_size = len(scaled) - train_size
        train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
        text.insert(END,"Length of Train Data: "+str(len(train))+"\n")
        text.insert(END,"Length of Test Data: "+str(len(test))+"\n")        

        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

def runlstm():
        global trainX,trainY,testX,testY
        global model_lstm,lstm_rmse
        global yhat_inverse,testY_inverse
        print(trainX.shape[1], trainX.shape[2])
        model_lstm = Sequential()
        model_lstm.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mae', optimizer='adam')
        history = model_lstm.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

        yhat = model_lstm.predict(testX)
        yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
        print("Tesing")
        lstm_rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        text.insert(END,'Test RMSE: '+str(lstm_rmse)+"\n")

        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        
        
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def runmultilstm():
        global data,scaled
        global yhat_inverse,testY_inverse
        global model_multi_lstm,multi_lstm_rmse
        predictDates = data.tail(len(testX)).index
        testY_reshape = testY_inverse.reshape(len(testY_inverse))
        yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
        sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
        values = data[['Weighted Price'] + ['Volume (BTC)'] + ['Volume (Currency)']].values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.head()
        reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
        reframed.dropna(inplace=True)
        

        values = reframed.values
        n_train_hours = int(len(values) * 0.7)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        
        print(train_X.shape[1], train_X.shape[2])
        model_multi_lstm = Sequential()
        model_multi_lstm.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
        model_multi_lstm.add(Dense(1))
        model_multi_lstm.compile(loss='mae', optimizer='adam')
        multi_history = model_multi_lstm.fit(train_X, train_y, epochs=300, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        
        yhat = model_multi_lstm.predict(test_X)
       
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        multi_lstm_rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        text.insert(END,'Test RMSE: '+str(multi_lstm_rmse)+"\n")
        pyplot.plot(yhat, label='predict')
        pyplot.plot(test_y, label='true')
        pyplot.legend()
        pyplot.show()

def graph():
        height=[lstm_rmse,multi_lstm_rmse]
        bars = ('LSTM RMSE', 'MULTI LSTM RMSE')
        y_pos = np.arange(len(bars))
        pyplot.bar(y_pos, height)
        pyplot.xticks(y_pos, bars)
        pyplot.show()   
        


        
font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Modeling for Bitcoin Market Price Prediction based on the Long Short Term Memory')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload_data)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

dt = Button(main, text="Import Data and Understanding", command=loaddataset)
dt.place(x=700,y=200)
dt.config(font=font1)

dp = Button(main, text="Data Preprocess", command=dataPreprocesSplit)
dp.place(x=700,y=250)
dp.config(font=font1)

lstm = Button(main, text="RUN LSTM", command=runlstm)
lstm.place(x=700,y=300)
lstm.config(font=font1)

mlstm = Button(main, text="RUN MULTI-LSTM", command=runmultilstm)
mlstm.place(x=700,y=350)
mlstm.config(font=font1)

gr = Button(main, text="Comparion of LSTM and Multi-LSTM", command=graph)
gr.place(x=700,y=400)
gr.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='slateGray1')
main.mainloop()
