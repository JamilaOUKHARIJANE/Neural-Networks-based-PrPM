"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path

version = "_mean"
ascii_offset = 161
beam_size = 3
th_reduction_factor = 1
One_hot_encoding=False
combined_Act_res = False
useProb_reduction = False
use_modulator = False
use_train_test_logs = False
BK_end = True
declare_BK = False
root_folder = Path.cwd() #/ 'implementation_real_logs'
data_folder = root_folder / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'
output_folder2 = data_folder / 'output2'

declare_folder = input_folder / 'declare_models'
xes_log_folder =  input_folder / 'log_xes'
log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

w = 0.9
epochs = 100
folds = 3
train_ratio = 0.8
variant_split = 0.9
validation_split = 0.2


log_list = [
   #'helpdesk.xes',
    # 'BPI2011.xes',
    'BPI2012.xes' ,
    'BPI2013.xes',
    'BPI2017.xes'
]

synthetic_log_list = [
    #'BPI2011.xes',
    #'BPI2012.xes',
    #'BPI2017.xes',
    'helpdesk.xes'
    #'sepsis_cases.xes'
    #'BPI2013.xes'
]
method_marker = {'baseline': 'x', 'beamsearch (beam size = 3)': '1', 'beamsearch (beam size = 5)': '.', 'beamsearch (beam size = 10)': '',
                 'beamsearch with BK (beam size = 10)': '+', 'beamsearch with BK (beam size = 5)':'*', 'frequency':'+', 'beamsearch with BK (beam size = 3)': '.'}
method_color = {'baseline': 'mediumpurple', 'beamsearch (beam size = 3)': 'deepskyblue', 'beamsearch (beam size = 5)': 'orange',
                'beamsearch (beam size = 10)': 'purple', 'beamsearch with BK (beam size = 10)': 'brown',  'beamsearch with BK (beam size = 5)':'red',
                'beamsearch with BK (beam size = 3)': 'green'}
