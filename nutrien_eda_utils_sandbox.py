#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:42:16 2019

@author: nkordjazi@ibm.com
"""


import os
import sys

full_path = os.path.abspath("").split("/kds-utilities-library")[0] + "/kds-utilities-library/"
sys.path.append(full_path)
from nutrien_eda_utils import EdaUtils
eda = EdaUtils(path_to_data = '/Users/nkordjazi@ibm.com/Box/CPT Canada - Cognitive & Analytics Practice/Nutrien/data/')


eda.visualize_calculate_us_per_area_affodability()
eda.visualize_us_affordibiity_demand_lagged_correlation()
eda.visualize_us_affordibiity_demand_elasticity()
eda.visualize_weather_deviation_from_average(regions_of_interest = [])    # All regions
eda.visualize_weather_deviation_from_average(regions_of_interest = ['USSouth' , 'USNorthCentral'])   # Just US Regions