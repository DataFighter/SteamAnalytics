#!/usr/bin/env python

'''
The code script is to run the lstm with different dataset in parallel in Spark.
'''

import os, sys, inspect, math, json

this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
lstm_dir = os.path.realpath( os.path.abspath( os.path.join( this_dir, "../models/")))
lstm_params_dir = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/params/")))
lstm_data_dir = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/data/")))
lstm_code_dir = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/scode/")))
csv_dir = os.path.realpath(os.path.abspath(os.path.join(this_dir, "../Synapsify_data")))


if lstm_dir not in sys.path:
    sys.path.insert(0, lstm_dir)
    sys.path.insert(0, lstm_code_dir)

### from load_params import Load_LSTM_Params
from lstm_class import LSTM as lstm

# param_file = 'orig_params.json'
param_file = 'ruofan_params.json'
data_file  = ''

json_path = os.path.join(lstm_params_dir, param_file)
'''
jsonFile = open(json_path, "r")
tmp_data = json.load(jsonFile)
jsonFile.close()


tmp = tmp_data['data_file']
tmp_data['data_file'] = "Annotated_Comments_for_Always_Discreet_1.csv"
tmp_data['train_size'] = 1524
tmp_data['test_size'] = 1533

jsonFile = open(json_path, "w+")
jsonFile.write(json.dumps(tmp_data))
jsonFile.close()

'''
### PD_list is a list of instances of "PD"
PD_list = {}
index = 0
for ff in os.listdir(csv_dir):
    if ff.endswith(".csv"):

        ### Open the original Json file, and load the parameters.
        jsonFile = open(json_path, "r")
        tmp_data = json.load(jsonFile)
        jsonFile.close()

        ### Update the values of Json parameters: "data_file", "train_size", "test_size".
        tmp_data['data_file'] = ff
        ### Compute the train and testing dataset size
        csv_path = os.path.realpath(os.path.abspath(os.path.join(csv_dir, ff)))
        num_lines = sum(1 for line in open(csv_path))
        num_instances = num_lines - 1   ### The first line shouldn't be considered.
        train_size = int(math.ceil(num_lines / 2.0))
        test_size = int(num_instances - train_size)
        tmp_data['train_size'] = train_size
        tmp_data['test_size'] = test_size

        '''
        ### Create the Json file name as well as the path of the json file, and store the updated information into our Json file
        new_json_name = ff
        new_json_name = new_json_name.strip('.csv') ### Delete ".csv"
        new_json_name += ".json"
        new_json_path = os.path.join(lstm_params_dir, new_json_name)
        '''

        jsonFile = open(json_path, "w+")
        jsonFile.write(json.dumps(tmp_data))
        jsonFile.close()

       ### temp = Load_LSTM_Params(lstm_params_dir, param_file)

        temp = lstm(params_dir=lstm_params_dir, param_file=param_file)
        print temp.model_options
        ##temp.preprocess()
        ##temp.update_options()

        PD_list[str(index)] = temp
        index += 1

for item in PD_list:
    print item.model_options
    pass



'''
PD = Load_LSTM_Params(lstm_params_dir, param_file)
PD.preprocess()
PD.update_options()
print PD.model_options
# Here I can pickle the PD object for use later. Good if the data is HUGE

LSTM = lstm(PD)
LSTM.build_model()
LSTM.train_model().test_model()

# IN.gen_sent_tvt(0,5,100,100)
'''