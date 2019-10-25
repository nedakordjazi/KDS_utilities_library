#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:22:39 2019

@author: nkordjazi@ibm.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import math
import itertools
from sklearn.decomposition import PCA


import os, sys
sns.set()

class EdaUtils():

    def __init__(self,  path_to_data):
        
        '''
        path_to_data: the path to the data folder on Box, ending in "CPT Canada - Cognitive & Analytics Practice/Nutrien/data/" 
        functions with visualize_ at the biggining create visualizations
        
        '''
        
                 
        self.path_to_data = path_to_data
        
    def bloomberg_price_flatten_and_merge(self):
        
        '''
        flattens the C1, S1, RR1, 1 and KO1 sheets
        '''
        

        bbp_c1 = pd.read_excel(self.path_to_data + 'Bloomberg/bloomberg-prices.xlsx', 'Corn')
        bbp_c1.rename(columns={'Settlement Price':'Settlement_Price_C1'} , inplace  = True)
        
        bbp_s1 = pd.read_excel(self.path_to_data + 'Bloomberg/bloomberg-prices.xlsx', 'Soy')
        bbp_s1.rename(columns={'Settlement Price':'Settlement_Price_S1'} , inplace  = True)
        
        bbp_w1 = pd.read_excel(self.path_to_data + 'Bloomberg/bloomberg-prices.xlsx', 'Wheat')
        bbp_w1.rename(columns={'Settlement Price':'Settlement_Price_W1'} , inplace  = True)
        
        bbp_rr1 = pd.read_excel(self.path_to_data + 'Bloomberg/bloomberg-prices.xlsx', 'Rice')
        bbp_rr1.rename(columns={'Settlement Price':'Settlement_Price_RR1'} , inplace  = True)
        
        bbp_ko1 = pd.read_excel(self.path_to_data + 'Bloomberg/bloomberg-prices.xlsx', 'Palm Oil')
        bbp_ko1.rename(columns={'Settlement Price':'Settlement_Price_KO1'} , inplace  = True)
        
        
        df = bbp_c1.merge(bbp_s1 , on = 'Date', suffixes=('_c1', '_s1'))
        df = df.merge(bbp_w1 , on = 'Date')
        df = df.merge(bbp_rr1 , on = 'Date')
        df = df.merge(bbp_ko1 , on = 'Date')
        
        #df= df.loc[: , ['Date' , 'Settlement_Price_W1' , 'Settlement_Price_S1' , 'Settlement_Price_C1' , 'Settlement_Price_RR1' , 'Settlement_Price_KO1']]
        df= df.loc[: , ['Date' , 'Settlement_Price_S1' , 'Settlement_Price_C1' ]]
        
        return df
    
    def visualize_us_import_quantity_trade(self):

        MOP_trade= pd.read_excel(self.path_to_data + 'CRU/usa-net-potash-import_clean.xlsx')
        MOP_trade = MOP_trade.T
        MOP_trade.rename(columns = {0:'Canada' , 1:'World'} , inplace = True)
        MOP_trade = MOP_trade.drop(['Import_from' , '2018 YTD'])
        sns.set()
        fig, ax = plt.subplots()
        plt.plot(MOP_trade['World'] , 'o-', label = 'Global - Canada Included') 
        plt.plot(MOP_trade['Canada'] , 'o-', label = 'Canada') 
        plt.ylabel('Tonnes' , fontsize = 18)
        plt.title('Net Import Quantity of Potash (US)' ,  fontsize = 24)
        plt.legend()
        
        return MOP_trade
    
    def visualize_calculate_us_per_area_affodability(self):

        csb_price= pd.read_excel(self.path_to_data + 'Other/us-wheat-corn-soy-prices-2010-to.xlsx', 'Price')
        csb_price = csb_price[csb_price.date >= pd.to_datetime('2010-01-07 00:00:00')]
        #df = csb_price.groupby([pd.Grouper(key = 'date' , freq='W-MON')]).mean()
        df = csb_price.groupby([pd.TimeGrouper(key='date', freq='7D')]).mean()
        
        MOP_price= pd.read_excel(self.path_to_data + 'CRU/us-weekly-potash-fob-spot-price_clean.xlsx')
        #MOP_price = MOP_price.dropna()
        MOP_price.rename(columns = {'PriceDate':'date' } , inplace = True)
        df = df.merge(MOP_price , on = 'date')
        #sns.set()
        #fig, ax = plt.subplots()
        #plt.plot(df.Soybeans , df['Potash Granular Bulk FOB US Midwest East Spot'] , 'o-', label = 'Potash Granular Bulk FOB US Midwest East Spot') 
        #plt.plot(df.Corn , df['Potash Granular Bulk FOB US Midwest East Spot'] , 'o-', label = 'Potash Granular Bulk FOB US Midwest East Spot') 
        #plt.plot(df.Wheat , df['Potash Granular Bulk FOB US Midwest East Spot'] , 'o-', label = 'Potash Granular Bulk FOB US Midwest East Spot') 
        
        
        # acre harvest about 40 bushels of crop pr acre
        # fertilizer requirement about 150 pounds pr acre
        
        df['crop_mix_price'] = (df.Soybeans + df.Corn + df.Wheat)/3
        #plt.plot(df.crop_mix_price , df['Potash Granular Bulk FOB US Midwest East Spot'] , 'o-', label = 'Potash Granular Bulk FOB US Midwest East Spot') 
        df['affordibility_MWE'] = (df['crop_mix_price']*60)/(df['Potash Granular Bulk FOB US Midwest East Spot']*(150/2000))
        df['affordibility_MWW'] = (df['crop_mix_price']*60)/(df['Potash Granular Bulk FOB US Midwest West Spot']*(150/2000))
        df['affordibility_S'] = (df['crop_mix_price']*60)/(df['Potash Granular Bulk FOB US South Spot']*(150/2000))
        
        fig, ax = plt.subplots()
        plt.plot(df.date , df['affordibility_MWE'] , 'o-', label = 'Affordibility - US Midwest East') 
        plt.plot(df.date , df['affordibility_MWW'] , 'o-', label = 'Affordibility - US Midwest West') 
        plt.plot(df.date , df['affordibility_S'] , 'o-', label = 'Affordibility - US South') 
        plt.ylabel('Affordibility' , fontsize = 18)
        plt.title('Affordibility Across the US Market' ,  fontsize = 24)
        plt.legend()
        
        return df

    def visualize_us_affordibiity_demand_lagged_correlation(self):
        
        df = self.visualize_calculate_us_per_area_affodability()
        MOP_trade = self.visualize_us_import_quantity_trade()
        
        df_ann = df.groupby([pd.TimeGrouper('M',key='date')]).mean()
        
        MOP_trade['Year'] = [a.year for a in MOP_trade.index]
        MOP_trade['Month'] = [a.month for a in MOP_trade.index]
        df_demand_ann = df_ann.merge(MOP_trade , on = ['Year' , 'Month'])
        C= []
        for sh in range(0,12):
            cc = np.corrcoef(df_demand_ann.World.astype('float')[sh:] , df_demand_ann['affordibility_MWE'].shift(sh).astype('float')[sh:])[0,1]
            C = C+[cc]
        
        sns.set()
        fig, ax = plt.subplots()
        plt.plot(C , 'o-', label = 'Correlation Value' , linewidth = 3) 
        plt.ylabel('Correlation Coef.' , fontsize = 16)
        plt.xlabel('Lag Duration (months)' , fontsize = 16)
        plt.xticks(range(0,13))
        plt.title('Lagged Correlation Between Affordibility and Net Import Quantity - US Midwest East' ,  fontsize = 16)
        plt.legend()



    def visualize_us_affordibiity_demand_elasticity(self):
        
        df = self.visualize_calculate_us_per_area_affodability()
        MOP_trade = self.visualize_us_import_quantity_trade()
        
        df_ann = df.groupby([pd.TimeGrouper('M',key='date')]).mean()
        
        MOP_trade['Year'] = [a.year for a in MOP_trade.index]
        MOP_trade['Month'] = [a.month for a in MOP_trade.index]
        df_demand_ann = df_ann.merge(MOP_trade , on = ['Year' , 'Month'])
        
        df_demand_ann['delt_affordability'] = df_demand_ann['affordibility_MWE'].diff()/df_demand_ann['affordibility_MWE'].shift(1)
        df_demand_ann['delt_fert_imp'] = df_demand_ann['World'].diff()/df_demand_ann['World'].shift(1)
        df_demand_ann['elasticity_demand_affordability']= df_demand_ann['delt_affordability']/df_demand_ann['delt_fert_imp']
        
        
        c = []
        for a,b in zip(df_demand_ann.Year.tolist(),df_demand_ann.Month.tolist()):
            c = c + [str(a)+'/'+str(b)]
        df_demand_ann['date'] = pd.to_datetime(c)
        fig, ax = plt.subplots()    
        plt.plot(df_demand_ann.date , df_demand_ann['elasticity_demand_affordability'] , 'o-', label = 'Demand Elasticity of Affordability' )
        plt.ylabel('Elasticity' , fontsize  = 16)
        plt.title('Demand Elasticity of Affordability - US Midwest East' , fontsize = 16)
        plt.legend()
        
    
    def weather_clean_and_save(self):
        
        prec_hist= pd.read_csv(self.path_to_data + 'Weather/WeatherData/Historical/precipitation-monthly-Jan2014-July2019_allregions.csv')
        prec_hist.rename(columns = {'Value':'Raw_Percipitation'} , inplace = True)
        prec_hist['Timestamp'] = pd.to_datetime(prec_hist['Timestamp']) 
        prec_hist['Month'] = [a.month for a in prec_hist['Timestamp']]
        avg_prec= pd.DataFrame(prec_hist.groupby(['Region' , 'Month'])['Raw_Percipitation'].mean())
        avg_prec.rename(columns = {'Raw_Percipitation':'5yr_avg_perc'} , inplace = True)
        avg_prec['Region'] = [a[0] for a in avg_prec.index.tolist()]
        avg_prec['Month'] = [a[1] for a in avg_prec.index.tolist()]
        avg_prec.index = range(len(avg_prec))
        prec_hist = prec_hist.merge(avg_prec , on = ['Region' , 'Month'])
        prec_hist['prec_dev_from_avg'] = prec_hist.Raw_Percipitation - prec_hist['5yr_avg_perc']
        prec_hist.to_csv(self.path_to_data + 'Weather/percipitation-data-clean.csv')
        
        temp_hist= pd.read_csv(self.path_to_data + 'Weather/WeatherData/Historical/temperature-monthly-Jan2014-July2019_allregions.csv')
        temp_hist.rename(columns ={'Value':'Raw_Temp'} , inplace =True)
        temp_hist['Timestamp'] = pd.to_datetime(temp_hist['Timestamp']) 
        temp_hist['Month'] = [a.month for a in temp_hist['Timestamp']]
        avg_temp= pd.DataFrame(temp_hist.groupby(['Region' , 'Month'])['Raw_Temp'].mean())
        avg_temp.rename(columns = {'Raw_Temp':'5yr_avg_temp'} , inplace = True)
        avg_temp['Region'] = [a[0] for a in avg_temp.index.tolist()]
        avg_temp['Month'] = [a[1] for a in avg_temp.index.tolist()]
        avg_temp.index = range(len(avg_temp))
        temp_hist = temp_hist.merge(avg_temp , on = ['Region' , 'Month'])
        temp_hist['temp_dev_from_avg'] = temp_hist.Raw_Temp - temp_hist['5yr_avg_temp']
        temp_hist.to_csv(self.path_to_data + 'Weather/temprature-data-clean.csv')

#        weather = temp_hist.merge(prec_hist , on = ['Region' , 'Timestamp'])
#        oct_forec= pd.read_csv('/Users/nkordjazi@ibm.com/Box/CPT Canada - Cognitive & Analytics Practice/Nutrien/data/WeatherData/October2019Forecast/precipitation-forecast-6-months-forward_allregions.csv')

################################ WEATHER VISUALIZATION
    def visualize_weather_deviation_from_average(self , regions_of_interest = []):
        
        '''
        Visualizes weather in regions of interest - if [] visualizes all regions  e.g. regions_of_interest = ['USNorthCentral','USSouth']
        '''
        
        temp_hist = pd.read_csv(self.path_to_data + 'Weather/temprature-data-clean.csv')
        prec_hist = pd.read_csv(self.path_to_data + 'Weather/percipitation-data-clean.csv')
        
        if regions_of_interest == []:
            regions_of_interest = temp_hist.Region.unique()
        
        
        sns.set()
        fig, ax = plt.subplots()
        for r  in regions_of_interest:
            exec(r+ "= pd.DataFrame(temp_hist[temp_hist.Region == '"+r+"'].groupby(['Region' , 'Timestamp'])['temp_dev_from_avg'].mean())")
            exec(r+"['Time'] = [a[1] for a in "+r+".index.tolist()]")
            exec("plt.plot("+r+".Time , "+r+".temp_dev_from_avg , label = '"+r+"')")
        
        plt.xticks(rotation=90)
        plt.legend()
        plt.ylabel('Deviation' , fontsize = 16)
        plt.title('Temprature Deviation from the 5-Year Average in Regions of Interest' ,  fontsize = 18)
        
        
        sns.set()
        fig, ax = plt.subplots()
        for r  in regions_of_interest:
            exec(r+ "= pd.DataFrame(prec_hist[prec_hist.Region == '"+r+"'].groupby(['Region' , 'Timestamp'])['prec_dev_from_avg'].mean())")
            exec(r+"['Time'] = [a[1] for a in "+r+".index.tolist()]")
            exec("plt.plot("+r+".Time , "+r+".prec_dev_from_avg , label = '"+r+"')")
        
        plt.xticks(rotation=90)
        plt.legend()
        plt.ylabel('Deviation' , fontsize = 16)
        plt.title('Precipitation Deviation from the 5-Year Average in Regions of Interest' ,  fontsize = 18)


    def visualize_us_harvest_months_and_yield(self , Commodity_Description = ['Corn']):
        
        '''
        Commodity_Description : subset of ['Corn' , 'Meal, Soybean' , 'Oil, Palm' , 'Oil, Palm Kernel']
        '''
        
        
        us_harvest= pd.read_excel(self.path_to_data + 'USDA/us-harvest-yeild-production-by-crop.xlsx')
        us_harvest_idx = us_harvest.groupby(['Commodity_Description','Market_Year','Attribute_Description'])['Value'].idxmax()
        
        H = us_harvest[(us_harvest.Attribute_Description == 'Area Harvested') & (us_harvest.Market_Year>=2014) & (us_harvest.Commodity_Description.isin(Commodity_Description))]
        
        c = []
        for a,b in zip(H.Market_Year.tolist(),H.Month.tolist()):
            c = c + [str(a)+'/'+str(b)]
        H.index = H.Market_Year
        H = H.loc[:, ['Month']]
        
        
        sns.set()
        
        H.plot.bar()
        plt.xlabel('Market Year' , fontsize = 16)
        plt.ylabel('Month' , fontsize = 16)
        plt.title('Recorded Harvest Month')


        Y = us_harvest[(us_harvest.Attribute_Description == 'Yield') & (us_harvest.Market_Year>=2014) & (us_harvest.Commodity_Description.isin(Commodity_Description))]
        
        c = []
        for a,b in zip(Y.Market_Year.tolist(),Y.Month.tolist()):
            c = c + [str(a)+'/'+str(b)]
        Y.index = Y.Market_Year
        Y = Y.loc[:, ['Value']]
        
        
        sns.set()
        
        Y.plot.bar()
        plt.xlabel('Market Year' , fontsize = 16)
        plt.ylabel('(1000 Ha)' , fontsize = 16)
        plt.title('Harvested Area')




