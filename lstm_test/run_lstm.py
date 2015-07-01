"""
This code is used to test different dataset, which is embedded in a "for" loop.
"""

import os
import inspect
import lstm1

def main():

    ### Get the pkl file directory
    current_file_name = os.path.split(inspect.getfile(inspect.currentframe()))[0]
    pkl_dir_name = os.path.realpath(os.path.abspath(os.path.join(current_file_name, "../Synapsify_pkl_data/")))

    for ff in os.listdir(pkl_dir_name):
        if (not ff.endswith(".dict.pkl")):      ### Select the file that is not dictionaries
                filename = os.path.realpath(os.path.abspath(os.path.join(pkl_dir_name, ff)))
                lstm1.train_lstm(
                    #reload_model="lstm_model.npz",
                    max_epochs = 100,
                    test_size = 500,
                    data_file_name = filename)




if __name__ == '__main__':
    # Run Lstm code
    main()

