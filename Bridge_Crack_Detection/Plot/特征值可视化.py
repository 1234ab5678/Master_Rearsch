# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    #columa_names所有列表的名称
    column_names = ['crack']
    data = pd.read_csv(file_path,header = None, names = column_names)
    return data
dataset = read_data('./半径.txt')
xValue=list(range(1,51))
yValue1=dataset['crack']

dataset = read_data('./nc半径.txt')
yValue2=dataset['crack']

plt.scatter(xValue,yValue1, marker = 'x',color = 'red', s = 40 ,label = 'First')
plt.scatter(xValue,yValue2, marker = 'o',color = 'blue', s = 40 ,label = 'Second')

plt.axhline(y=60, color='k', linestyle='-')
plt.show()