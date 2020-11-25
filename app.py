#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go

from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet,BayesianRidge,PassiveAggressiveRegressor,LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from statsmodels.tsa.ar_model import AR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.svm import SVR
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import math
import time

#import dash
import dash_core_components as dcc
import dash_html_components as html
#from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask
import os
import dash_daq as daq
import base64
import io
import pandas_datareader as web
from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, State, Trigger


prediction_length = 5*12*30


scaler_names = [
           {'label': 'Standard', 'value': 'Standard'},
           {'label': 'Min-Max', 'value': 'Min-Max'}
          ]


scalers = {
            'Standard': StandardScaler(),
           'Min-Max': MinMaxScaler((-1,1))
           }
          


    

model_names = [{'label':model,'value':model} for model in sorted(['AdaBoost', 
                                                                  'Random Forest', 
                                                                  'Multilayer Perceptron (Neural Net)', 
                                                                  'Stochastic Gradient Descent', 
                                                                  'Support Vector Machine',
                                                                  'Lasso',
                                                                  'Ridge',
                                                                  'Elastic net',
                                                                  'K Nearest Neighbors',
                                                                  'Bayesian Ridge',
                                                                 'Passive Aggressive Regressor', 
                                                                  'Linear Regression',
                                                                  'Gradient Boost',
                                                                  'Extra Trees Regression'])]


models = {
         'AdaBoost':AdaBoostRegressor(random_state=42,n_estimators=100),
          
          'Multilayer Perceptron (Neural Net)':MLPRegressor(random_state = 42,
                     hidden_layer_sizes=(100,50,25,2),activation='tanh'),
          'Random Forest':RandomForestRegressor(random_state = 42,
                             n_jobs=-1,
                             n_estimators=100,
                             oob_score = True),
          'Stochastic Gradient Descent':SGDRegressor(random_state = 42, penalty = 'elasticnet'),
            'Support Vector Machine':SVR(gamma='auto'),
            'Lasso':Lasso(random_state=42),
            'Ridge':Ridge(random_state=42),
            'Elastic net':ElasticNet(random_state=42),
            'K Nearest Neighbors':KNeighborsRegressor(n_neighbors=100, n_jobs=-1),
            'Bayesian Ridge':BayesianRidge(),
            'Passive Aggressive Regressor':PassiveAggressiveRegressor(random_state=42),
            'Linear Regression':LinearRegression(n_jobs=-1),
            'Gradient Boost':GradientBoostingRegressor(random_state=42, n_estimators=100),
            'Extra Trees Regression':ExtraTreesRegressor(random_state = 42,n_jobs=-1,n_estimators=100)
           
            
            
            
         }




metadata = pd.read_excel('yahoo_metadata_stable.xlsx').set_index('Name')




server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, server = server,prevent_initial_callbacks=True)


app.title = 'My Finance AI'



data = []


def serve_layout():
    return html.Div(
        
                    children= [
            
                
            #html.Br(),
            html.H1('Analyzing financial instrument prices with machine learning', style=dict(textAlign='center',fontSize=55, fontFamily='Arial',color = "blue")),
            html.Br(),
            
            html.P("Investors base their investment decisions on expected future returns on their investments. Such knowledge required for a sophisticated investment decision should be a result of a thorough analysis of the invested item. With machine learning one can search for such patters in an investment's (e.g. stock, ETF etc.) history data that are invisible for a human observer but reacheable with the utilization of data science. Much work related to predicting stock prices with machine learning is mostly related with deep neural works, especially LSTM networks. These models can grow very heavy in terms of required processing power and execution time. This application aims to give the user the possibility to test more light-weight machine learning models to predict investment prices. This mainly by selecting the number of features (i.e. the number of days before the current closing date) and using supervised learning algorithms to predict the last closing date's absolute change in price. This application is meant for testing simple machine learning techniques in the context of asset prices. This is only a work of a hobbyist intended for self-learning and for academic purposes only.",style=dict(textAlign='center')),
            html.Br(),
            html.P("The data used in this tool is retrieved from Yahoo Finance API using a datareader object from Python's pandas library. For the use of this application tens of thousands of different financial symbols with their metadata have been scraped from the Internet mainly via Wikipedia and Yahoo Finance web site. The data is thereby openly distributed without any need for login credentials. There might occur some quality issues related to retrieving the data (e.g. missing data from certain days) that are due to limitations on the data providers side. This is hosted on Heroku's free server which also has its limitations. If for some reason the data won't get visualized, it is most probably due to the server's lack of capacity to process that much data. If that happens, you try again by retrieving less past data. The application is meant for light weight purposes to run on a relatively weak server. The code though is open and you can get it from Github (links down below) and try it or even develop it on your own computer.",style=dict(textAlign='center')),
            html.Br(),
            html.P("Below users can conlude their analyses in three phases encompassing twelve steps indicated with corresponding order numbers. The first step includes retrieving the data from Yahoo Finance by selecting the asset type, the country in which it is traded and the actual asset itself. In the second phase one can tweak the machine learning parameters (i.e. the model, scaling, test size and features which are the number of preceding days). After tweaking the machine learning part, in the final phase users can select the number of days to forecast into the future and apply forecasting. This article uses autoregression as a regression baseline that the machile learning algorithm tries to beat.",style=dict(textAlign='center')),
            html.H2('Disclaimer',style=dict(textAlign='center',fontSize=26,color='red', fontFamily='Arial')),
            html.H3('This article is intended for academic and educational purposes and is not an investment recommendation. The information that is provided or that is derived from this website should not be a substitute for advice from an investment professional or profound research of financial instruments. The hypothetical models used in this tool do not reflect the investment performance of any actual product or strategy in existence during the periods tested and there is no guarantee that if such product or strategy existed it would have displayed similar performance characteristics. A decision to invest in any instrument or strategy should not be based on the information or conclusions contained herein. This is neither an offer to sell nor a solicitation for an offer to buy interests in financial instruments.',style=dict(textAlign='center',fontSize=20, fontFamily='Arial',color='red')),
            html.Br(),
            html.Div(className = 'row', children =[
                    

                     
                     
                     html.Div(className='three columns',children=[
                             
                          html.H3('1. Select an asset type.'),
                          dcc.Dropdown(id='equity_selection',
                                                              
                     
                                         options = [{'label':equity, 'value':equity} for equity in sorted(list(pd.unique(metadata.Type)))],
                                          value='Stocks',
                                           multi=False
                                          
                                   )
                     ]),
                         html.Div(id='country_selection',className='three columns',children=[
                             html.H3('2. Select a country.'),
                             
                             dcc.Dropdown(id = 'countries',
                                         multi = False,
                                   
                                         options = [{'label':country,'value':country} for country in sorted(list(pd.unique(metadata[metadata.Type=='Stocks'].Country)))],
                                         value='Finland'
                                         )
                         ]),


                    
                     html.Div(id = 'asset_selection',className='five columns',children=[
                     
                         html.Div(id='asset_title',children=[html.H3("3. Select from Finland's businesses")]),
                         
                         dcc.Dropdown(id = 'equity',multi = False, options = [{'label':name,'value':name} for name in sorted(list(pd.unique(metadata[(metadata.Type=='Stocks')&(metadata.Country=='Finland')].index)))], value = 'Nokia Corporation')
                     
                     ]),

            ]),
            html.Div(className='row',children=[
                        html.H3('4. Select days from the past.'),
                        dcc.Slider(id = 'history',
                                   min = 1*12*30,
                                   max = 10*12*30,
                                   value = 7*12*30,
                                   marks={
                                       1*12*30: 'a year',
                                       5*12*30: 'five years',
                                      #12*12*30: 'twelve years',
                                        10*12*30:'ten years',
                                        #15*12*30:'fifteen years',
                                       # 20*12*30:'twenty years'
                                   }
                                         ),
                         html.Div(id='history-container')

                     
                     ]),
            
       
          
            html.Br(),
            html.Div(className='row',style={'width':'88%', 'margin':5, 'textAlign': 'center'}, children=[
                html.Button('5. Get data from Yahoo Finance!', id = 'launch', n_clicks=0)
             ]),
             html.Br(),
             
             html.Div(id='data_store', children=[dcc.Store(id='yahoo_data'),dcc.Store(id='prepro_data')]),
         
             dcc.Loading(id='spinner0',fullscreen=False,type='dot',children=[html.Div(id='yahoo')]),
            html.P(id="yesterday"),
             
             dcc.Loading(id='spinner', fullscreen=False, type = 'circle', children=[html.Div(id='rearrange')]),
             dcc.Loading(id='spinner2', fullscreen=False, type = 'graph', children=[html.Div(id='test_predict')]),
            dcc.Loading(id='spinner3', fullscreen=False, type = 'cube', children=[html.Div(id='forecast')]),
        
                                html.Label(['Data Source: ', 
                                html.A('Yahoo Finance via Pandas Data Reader', href='https://finance.yahoo.com/')
                               ]),

                    html.Label(['Supervised learning with scikit-learn: ', 
                                html.A('Scikit-learn', href='https://scikit-learn.org/stable/supervised_learning.html')
                               ]),
                    html.Label(['About regression metrics: ', 
                                html.A('Medium', href='https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914')
                               ]),
                    html.Label(['Get from ', 
                                html.A('GitHub', href='https://github.com/tuopouk/myfinanceai')
                               ]),
                    html.Label(['by Tuomas Poukkula. ', 
                                html.A('Follow on Twitter.', href='https://twitter.com/TuomasPoukkula')
                               ]),
                    html.Label(['Follow also on ', 
                                html.A('LinkedIn.', href='https://www.linkedin.com/in/tuomaspoukkula/')
                               ])
    ])
     




@app.callback(
    Output('countries', 'options'),
    [Input('equity_selection', 'value')])
def update_countries(equity_type):

    
       return [{'label':country,'value':country} for country in sorted(list(pd.unique(metadata[metadata.Type==equity_type].Country)))]




@app.callback(
    Output('asset_title','children'),
    [Input('equity_selection', 'value'),
    Input('countries','value')],

)
def update_asset_title(equity_type, country):
    
    return html.H3("3. Select from "+(country.replace('United States of ','')+"'s").replace("Currency's",'').replace("Cryptocurrency's",'')+ " "+equity_type.replace('s','').replace('x','xe').replace('cy','cie').replace('Stock','businesse').lower().replace('etf','ETF')+'s.')


@app.callback(
    Output('equity','options'),
    [Input('equity_selection', 'value'),
    Input('countries','value')],

)
def update_assets(equity_type, country):
    
    return [{'label':symbol, 'value':symbol} for symbol in sorted(list(pd.unique(metadata[(metadata.Type==equity_type)&(metadata.Country==country)].index)))]



@app.callback(
    Output('equity','value'),
    [Input('equity_selection', 'value'),
    Input('countries','value')],

)
def update_asset_value(equity_type, country):
    
    try:
        value = sorted(list(pd.unique(metadata[(metadata.Type==equity_type)&(metadata.Country==country)].index)))[0]
    except:
        value = sorted(list(pd.unique(metadata[(metadata.Type==equity_type)].index)))[0]
    
    return value
    

@app.callback(
    ServersideOutput('yahoo_data','data'),
    [Trigger('launch', 'n_clicks'),
    
     State('equity_selection','value'),   
     State('countries','value'),
    State('equity','value'),
     State('history','value')]
    
)
def query(equity_type, country, equity, history):
    
    symbol = metadata[(metadata.Type==equity_type)&(metadata.Country==country)&(metadata.index==equity)].Symbol.values[0]
    currency = metadata[(metadata.Type==equity_type)&(metadata.Country==country)&(metadata.index==equity)].Currency.values[0]
    start = str(datetime.today() - pd.Timedelta(days=history)).split()[0]
    end = str(datetime.today()).split()[0]
    data =  web.DataReader(symbol,data_source='yahoo', end = end, start = start)
    data['Symbol']=symbol
    data['Currency']=currency
    data['start']=start
    data['end']=end
    data['equity']=equity
    
    return data.reset_index().to_dict('records')

    
    

@app.callback(
    Output('yahoo','children'),
    [

     Input("yahoo_data", "data")]

    
)
def plot_data(data):
    

    
        if data is None:
            raise PreventUpdate
        else:
            data = pd.DataFrame(data).set_index('Date')
    

        
        symbol = data.Symbol.values[0]
        currency = data.Currency.values[0]
        start = data.start.values[0]
        end=data.end.values[0]
        equity=data.equity.values[0]
    
        hovertemplate = ['<br><br>    Open: {}<br>    High: {}<br>    Low: {}<br><b>    Close</b>: {}<br>    Adj Close: {}<br>    Volume: {}'.format(
                                                                                                                                               round(data.iloc[i].Open,2),
        round(data.iloc[i].High,2),
        round(data.iloc[i].Low,2),
        round(data.iloc[i].Close,2),
        round(data.iloc[i]['Adj Close'],2),
        '{:,}'.format(int(data.iloc[i].Volume)).replace(',',' ')) for i in range(len(data))]
        
        fill = 'tonexty'
        
        #fill = None if data.Close.isna().sum()>0 else 'tozeroy'
        
        
        return html.Div(children=[

                                  dcc.Graph(figure = go.Figure(data = [
                                    go.Scatter(x = data.index,
                                              y = data.Close,
                                               connectgaps=False,
                                              name = '<b>'+equity+'</b><br>Currency in '+currency,
                                              marker = dict(color='blue'),
                                               fill=fill,
                                               hovertemplate=hovertemplate
                                              )
                                              ],
                                             layout = go.Layout(height=700,template="seaborn",xaxis = dict(title = 'Date'),
                                                               yaxis = dict(title='Closing Price ('+currency+')'),
                                                                hovermode="x unified",
                                                                title = dict(font=dict(family='Arial',size=23),text = '{} Daily Closing Price ({}) <br>from {} to {}'.format(equity,currency,pd.to_datetime(data.index.min()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th'),pd.to_datetime(data.index.max()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th')),
                                                                             x=0.5
                                                                            )
                                                               )
                                           )
                        ),
                                  html.Label(['Check '+equity+' on ',
                                             html.A('Yahoo Finance.',
                                                    href= 'https://finance.yahoo.com/quote/'+symbol+'/',
                                                    target="_blank"
                                                   
                                                   )
                                             ],
                                            
                                              style=dict(textAlign='center',fontSize=22, fontFamily='Arial')),
                                  html.Br(),
                                  
                                    

                                                                            
                                  html.Div(className='row',children=[
                                                  html.H3('6. Select the number of preceding days for training.'),
                                                  dcc.Slider(id='past',
                                                              
                                                              min = 1,
                                                              max=60,
                                                             step=1,
                                                             value=14,
                                                              marks={
                                                                  1:'one day',
                                                                  
                                                                  14: 'two weeks',
                                                                  
                                                                  30: 'one month',
                                                                  
                                                                  40: '40 days',
                                                                  
                                                                  60: 'two months'
                                                                  
                                                                #  90: 'three months',
                                                                     
                                                                  #  120: 'four months'
                                                                    } 
                                                            ),
                                                  html.Div(id='slider-container')
                                      ]),
                                                  

                                          html.Div(className='row',
                                                   style={'width':'88%', 'margin':20, 'textAlign': 'center'},
                                                   children=[
                                                        html.Button( '7. Preprocess data', 
                                                                    id = 'train',
                                                                    n_clicks=0)
                                                       
                                                   ])
                                     ])
                                      

@app.callback(
    Output('slider-container', 'children'),
    [Input('past', 'value')])
def update_output(value):
    return 'You have selected {} preceding days.'.format(value)


@app.callback(
    Output('history-container', 'children'),
    [Input('history', 'value')])
def update_output(value):
    return 'You have selected {} days ({} years) from the past.'.format('{:,}'.format(value).replace(',',' '),round(value/(12*30),1))


@app.callback(
    Output('forecast_length_container', 'children'),
    [Input('forecast_length', 'value')])
def update_output(value):
    return 'You have selected forecasting for {} days ({} months) to the future.'.format('{:,}'.format(value).replace(',',' '),round(value/(30),1))


def arrange(data,past):
    dfs=[]
#     for index in tqdm(range(past,len(data))):

#         dff= pd.concat([pd.DataFrame(data.iloc[index]).T.reset_index(), data.iloc[index-past:index,:].T.reset_index()
#     .drop('index',axis=1)],axis=1).set_index(['index','Close'])
#         dff.columns = ['day '+str(i+1) for i in range(len(dff.columns))]
#         dfs.append(dff)


#     dff = pd.concat(dfs)
#     dff=dff.reset_index().set_index('index')
#     dff.index.name='Date'
#     dff['Change'] =dff.Close-dff[dff.columns[-1]]
#     dff = dff[[c for c in dff.columns if 'day' in c]+['Close','Change']]
    
# This is a bit faster yet less elegant.

    for index in tqdm(range(past, len(data))):

        goal = pd.DataFrame(data.iloc[index]).T 

        d=data.iloc[index-past:index,:].T
        d.columns=['day '+str(i+1) for i in range(len(d.columns))]

        d['Date'] = goal.index[0]
        d['Close'] = goal[goal.columns[0]].values[0]
        d['Change'] = d.Close - d['day '+str(past)]
        d=d.set_index('Date')
        dfs.append(d)
    

    dff=pd.concat(dfs).sort_index()
    dff=dff[[c for c in dff.columns if 'day' in c]+['Close','Change']]
    
   
    
    return dff




@app.callback(
    ServersideOutput('prepro_data','data'),
    [Input('train', 'n_clicks')],
    [State('yahoo_data','data'),
     State('past','value')

    ]
)
def preprocess(n_clicks, data,  past):
    
    if n_clicks > 0:
    
        if data is None:
            
            raise PreventUpdate
        else:
            dff = pd.DataFrame(data).set_index('Date')
            
            symbol = dff.Symbol.values[0]
            currency = dff.Currency.values[0]
            equity=dff.equity.values[0]
            dff=dff[['Close']]
            start=time.time()
            try:
                df = arrange(data=dff,past=past)

                df['Currency']=currency
                df['Equity']=equity
                df['Symbol']=symbol

            except:
                return html.Div('Try again with less data.',style={'width':'88%', 'margin':20, 'textAlign': 'center','fontSize':22})

            end = time.time()
            print('Arranging took {} seconds'.format(end-start))
            return df.reset_index().to_dict('records')


@app.callback(
    Output('rearrange','children'),
  
    [Input('prepro_data','data'),

    ]
)
                                 
def rearrange(data):
  

        if data is None:
            raise PreventUpdate
        else:
            data = pd.DataFrame(data).set_index('Date')


        symbol = data.Symbol.values[0]


        currency = data.Currency.values[0]
        equity=data.Equity.values[0]
        
        data=data[['Close']]



        
        return html.Div(
                        id = 'test_section',
                        children=[
                                  html.P('Ready!',style={'width':'88%', 'margin':20, 'textAlign': 'center','fontSize':15}),
                                  
                                  html.Br(),
                                  
                                  html.Div(className='row',children = [
                                  
                                          html.Div(className='four columns',children=[
                                                  html.H3('8. Select a model.'),
                                                  dcc.Dropdown(id='model',
                                                              multi = False,
                                                              options = model_names,
                                                              value='Lasso')
                                          ]),
                                      
                                          html.Div(className='four columns',children=[
                                                  html.H3('9. Select feature scaling.'),
                                                  dcc.RadioItems(id='scaler',
                                                              
                                                              options = scaler_names,
                                                              value='Standard',
                                                              labelStyle={'display': 'inline-block'} )
                                          ]),
                                              
                                          html.Div(className= 'four columns', 
                                                   children=[
                                          
                                                      html.H3('10. Select test size.'),
                                                      dcc.RadioItems(id='test_size',                                                              
                                    options = [{'label':str(val)+' days', 'value':str(val)+' days'} for val in [10,20,40]],
                                                              value='10 days',
                                                              labelStyle={'display': 'inline-block'} )
                                                  ])
                                   ]),
                                 
    
                                       

                            html.Div(
                                     style={'width':'88%', 'margin':20, 'textAlign': 'center'},
                                     children=[
                                         html.Button( '11. Test', id = 'test',n_clicks=0)
                            ])
                        ])

                                 
    

@app.callback(
    Output('test_predict','children'),
    [Input('test', 'n_clicks')],
    [State('prepro_data','data'),
    State('model','value'),
     State('scaler','value'),
     State('test_size','value')

    ]
)
                                 
def test(n_clicks, data, 
             model, 
             scaler,
             
             test_size):                                  
    if n_clicks > 0:
        
        if data is None:
            raise PreventUpdate
        else:
            df = pd.DataFrame(data).set_index('Date')
            currency = df.Currency.values[0]
            equity = df.Equity.values[0]
            symbol = df.Symbol.values[0]
            df.drop(['Currency','Equity','Symbol'],axis=1,inplace=True)
        
        
           
        features = list(df.drop(['Close','Change'],axis=1).columns)
        predicted_label = ['Change']

        test_size = int(test_size.split(' days')[0])
        
        df_split = -test_size
        
        
        scl = scalers[scaler]
        model = models[model]

        train_data = df.iloc[:df_split,:]
        x_train = train_data[features]
        X_train =scl.fit_transform(x_train)


        y_train = train_data[predicted_label]

        test_data = df.iloc[df_split:,:]
        x_test = test_data[features]
        X_test =scl.transform(x_test)
        y_test = test_data[['Close']]


        start = time.time()
        model.fit(X_train, y_train.values.ravel())
        end=time.time()
        print('Fitting took {} seconds.'.format(end-start))



        sim_data = train_data.iloc[-1:,:].copy()


        start = time.time()
        while len(sim_data) <= len(test_data):

            d_last = sim_data.iloc[-1:,1:].copy().drop('Change',axis=1)
            d_last.columns=features
            d_last['Change'] = model.predict(scl.transform(d_last[features]))
            d_last['Close'] = np.maximum(0, d_last[features[-1]] + d_last.Change)
            sim_data = pd.concat([sim_data,d_last])


        end=time.time()
        print('Simulation took {} seconds.'.format(end-start))

        sim_data = sim_data.iloc[1:,:]
        sim_data.index = test_data.index
        
        ar_data = sim_data.copy()
        ar_data.Close = AR(train_data.Close.values).fit().predict(start=len(train_data),end=len(train_data)+len(test_data)-1, dynamic=False)

        ar_mae = mean_absolute_error(y_test.Close,ar_data.Close)
        ar_week_mae = mean_absolute_error(y_test.iloc[:8,:].Close,ar_data.iloc[:8,:].Close)
        ar_error = ar_data.iloc[0,:].Close-y_test.iloc[0,:].Close

        mae = mean_absolute_error(y_test.Close,sim_data.Close)
        nmae = mae/y_test.Close.std()
        r2 = 100*r2_score(y_test.Close,sim_data.Close)
        rmse = math.sqrt(mean_squared_error(y_test.Close,sim_data.Close))
        nrmse = rmse/y_test.Close.std()
        std = (y_test.Close.values-sim_data.Close.values).std()

        firsterror= sim_data.iloc[0,:].Close-y_test.iloc[0,:].Close
        lasterror = y_test.iloc[-1,:].Close-sim_data.iloc[-1,:].Close
        week_mae = mean_absolute_error(y_test.iloc[:8,:].Close,sim_data.iloc[:8,:].Close)


        first_error_percentage = np.absolute(firsterror/y_test.iloc[0,:].Close)

        week_errors = y_test.iloc[:8,:].Close-sim_data.iloc[:8,:].Close

        errors = y_test.Close-sim_data.Close

        week_mape = (np.absolute(week_errors)/np.absolute(y_test.iloc[:8,:].Close)).sum()/len(y_test.iloc[:8,:])
        whole_mape = (np.absolute(errors)/np.absolute(y_test.Close)).sum()/len(y_test)



        if len(test_data) >= 30:

            month_errors = y_test.iloc[:31,:].Close-sim_data.iloc[:31,:].Close
            month_mae = mean_absolute_error(y_test.iloc[:31,:].Close,sim_data.iloc[:31,:].Close)
            month_mape = (np.absolute(month_errors)/np.absolute(y_test.iloc[:31,:].Close)).sum()/len(y_test.iloc[:31,:])

            ar_month_mae = mean_absolute_error(y_test.iloc[:31,:].Close,ar_data.iloc[:31,:].Close)

            if month_mape <=.1:
                month_color='green'
            elif month_mape <=.5:
                month_color='orange'

            else:
                month_color='red'


            month_div= html.Div(className='three columns', children=[html.H5("First 30 days' average prediction error ("+currency+')',
                                                                                style=dict(textAlign='left')),
                                                            daq.LEDDisplay(value=round(month_mae,2),size=70,color=month_color,backgroundColor='black')
                                                                         , html.P('Average AR Error: '+str(round(ar_month_mae,2)))
                                        ])
        else:
            month_div=html.Div()



        if first_error_percentage <=.1:
            first_error_color='green'
        elif first_error_percentage <=.5:
            first_error_color='orange'
        else:
            first_error_color='red'


        if week_mape <=.1:
            mae_color='green'
        elif week_mape <=.5:
            mae_color='orange'
        else:
            mae_color='red'



        if whole_mape <=.1:
            color='green'
        elif whole_mape <=.5:
            color='orange'
        else:
            color='red'




        return html.Div(id = 'test_section', 
                            children=[
                                


                                 html.Div(children=[
                                         dcc.Graph(figure = go.Figure(data = [
                                                                      go.Scatter(x = train_data.iloc[int(-.4*len(train_data)):,:].index,
                                                                                              y = train_data.iloc[int(-.4*len(train_data)):,:].Close,
                                                                                              name = 'Train data',
                                                                                              marker = dict(color='blue')),

                                                                      go.Scatter(x = test_data.index,
                                                                                              y = test_data.Close,
                                                                                              name='Actual Value',
                                                                                              marker = dict(color='green')),
                                                                      go.Scatter(x = sim_data.index,
                                                                                              y = sim_data.Close,
                                                                                              name='Predicted Value',
                                                                                              marker = dict(color='red')),
                                                                     go.Scatter(x = ar_data.index,
                                                                                  y = ar_data.Close,
                                                                                  name='Autoregression',

                                                                                  marker = dict(color='#ffc233'))

                                                                                   ],
                                                                      layout=go.Layout(height=700,template="plotly_dark",legend=dict(
                                                                                                orientation="h",
                                                                                                yanchor="top",
                                                                                                y=0.99,
                                                                                                xanchor="center",
                                                                                                x=0.5,
                                                                                                title_font_family = 'Arial',
                                                                                               font = dict(family='Arial',
                                                                                                           size=18,
                                                                                                           color='black'),
                                                                                               bgcolor = 'white',
                                                                                               bordercolor = 'Black',
                                                                                               borderwidth=2),
                                                                                        xaxis = dict(showgrid=False,title = 'Date'),
                                                                                       hovermode="x unified",
                                                                                       yaxis = dict(showgrid=False,title='Closing Price ('+currency+')'),
              title = dict(font=dict(family='Arial',size=23),text = '{} Daily Closing Price ( {} )<br>from {} to {}'.format(equity,currency,pd.to_datetime(sim_data.index.min()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th'),pd.to_datetime(sim_data.index.max()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th')),x=0.5 )))),


                                    html.Div(className='row',children=[
                                        html.Div(className='three columns',children=[
                                            html.H5("First day's prediction error ("+currency+')',style=dict(textAlign='left')),
                                            daq.LEDDisplay(value=round(firsterror,2),size=70,color=first_error_color,backgroundColor='black')
                                            ,html.P('AR Error: '+str(round(ar_error,2)))
                                        ]),
                                        html.Div(className='three columns', children=[
                                            html.H5("First 7 days' average prediction error ("+currency+')',style=dict(textAlign='left')),
                                                            daq.LEDDisplay(value=round(week_mae,2),size=70,color=mae_color,backgroundColor='black')
                                            ,html.P('Average AR Error: '+str(round(ar_week_mae,2)))
                                        ]),

                                        month_div,


                           html.Div(className='three columns', children=[
                                            html.H5("Overall average prediction error ("+currency+')',style=dict(textAlign='left')),
                                                            daq.LEDDisplay(value=round(mae,2),size=70,color=color,backgroundColor='black')
                               ,html.P('Average AR Error: '+str(round(ar_mae,2)))
                                        ])
                                    ]),




                           html.H2('11. Select future days to forecast.'),
                           dcc.Slider(id = 'forecast_length',
                                      min = 1,
                                      max = 90,
                                      value = 21,
                                      marks ={
                                          1: 'one day',
                                          
                                          #7: 'one week',
                                          
                                          14: 'two weeks',
                                          
                                          30:'one month',

                                          90:'three months'

                                          #180:'six months',
                                        #  270: 'nine months',
                                          #12*30:'one year'
#                                           2*12*30:'two years',
#                                           3*12*30: 'three years',
#                                           4*12*30:'four years',
#                                           5*12*30:'five years'


                                      }),
                           html.Div(id='forecast_length_container'),
                           html.Div(style={'width':'88%', 'margin':20, 'textAlign': 'center'},
                                    className='row',
                                    children=[

                                   html.Button('12. Forecast future values.',id='forecast_button',n_clicks=0)

                           ])
                            ])
                            ])

        
        


                                  
                                  
    
@app.callback(
    Output('forecast','children'),
    [Input('forecast_button', 'n_clicks')],
    [State('prepro_data','data'),
    State('model','value'),
     State('scaler','value'),

     State('forecast_length','value')
    ]
)
                                 
def forecast(n_clicks, data, model, scaler, prediction_length):
    
    
    if n_clicks > 0:
    
        if data is None:
            raise PreventUpdate
        else:
            df = pd.DataFrame(data).set_index('Date')
            df.index = pd.to_datetime(df.index)
            currency = df.Currency.values[0]
            equity = df.Equity.values[0]
            symbol = df.Symbol.values[0]
            df.drop(['Currency','Equity','Symbol'],axis=1,inplace=True)
        
        
       
        features = list(df.drop(['Close','Change'],axis=1).columns)
        predicted_label = ['Change']


        scl = scalers[scaler]
        model = models[model]
        dfs = []
        
        x_train = df[features]
        X_train = scl.fit_transform(x_train)
        y_train = df[predicted_label]

        model.fit(X_train, y_train.values.ravel())
        
        
        last_index = df.index.max()
        
        df_predict = df.iloc[-1:,:].copy()
        
        

        for i in range(prediction_length):

            d_last = df_predict.iloc[-1:,1:].copy().drop('Change',axis=1)
            
            d_last.columns=features
            
            
            d_last.index+=pd.Timedelta(days=1)

            d_last['Change'] = model.predict(scl.transform(d_last[features]))
            d_last['Close'] = np.maximum(0, d_last[features[-1]] + d_last.Change)
            
           

            df_predict = pd.concat([df_predict,d_last])

        
        df_predict = df_predict.iloc[1:,:]
        
        
        ar_data = df_predict.copy()
        ar_data.Close = AR(df.Close.values).fit().predict(start=len(df),end=len(df)+len(df_predict)-1, dynamic=False)

  
        return html.Div(children=[
        
                dcc.Graph(figure=go.Figure(data = [
            go.Scatter(x = df.iloc[int(-.2*len(df)):,:].index,
                      y = df.iloc[int(-.2*len(df)):,:].Close,
                      name = 'Actual',
                       fill='tozeroy',
                      marker = dict(color='green')),

            go.Scatter(x = df_predict.index,
                      y = df_predict.Close,
                      name='Forecast',
                      fill='tozeroy',
                      marker = dict(color='red')),
                    
            go.Scatter(x = ar_data.index,
                      y = ar_data.Close,
                      name='Autoregression',
                 
                      marker = dict(color='#ffc233'))
                    

        ],
                       layout=go.Layout(template="ggplot2",
                                       height=700,
                                        legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=0.97,
                                        xanchor="center",
                                        x=0.5,
                                        title_font_family = 'Arial',
                                       font = dict(family='Arial',size=15,color='black'),
                                       bgcolor = 'white',
                                       bordercolor = 'black',
                                       borderwidth=2),
                           hovermode="x unified",
                                        xaxis = dict(title = 'Date'),
                                         yaxis = dict(title='Closing Price ('+currency+')'),
                                         title = dict(font=dict(family='Arial',size=23),x=0.5,text='{} Daily Closing Price ( {} ) <br>from {} to {}'.format(equity,currency,pd.to_datetime(df_predict.index.min()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th'),pd.to_datetime(df_predict.index.max()).strftime("%A %B %dth, %Y").replace(' 0',' ').replace('1th','1st').replace('2th','2nd').replace('3th','3rd').replace('11st','11th').replace('12nd','12th').replace('13rd','13th')))
                                          )
                         )
                         )
        ])
                                                     

app.config['suppress_callback_exceptions']=True  
app.layout = serve_layout
# Run app.
if __name__ == '__main__':
    app.run_server(debug=False)   
