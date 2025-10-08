# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 01:09:50 2025

@author: leona
"""

import pyodbc;
print('DRIVERS:\n' + '\n'.join(pyodbc.drivers()));
print('\nDSNs:\n' + '\n'.join(f'{k}: {v}' for k,v in pyodbc.dataSources().items()))