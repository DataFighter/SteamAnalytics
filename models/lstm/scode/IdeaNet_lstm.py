"""
The calling function for loading models and parameters, building models,
    and running the lstm model again some dataset.
"""

import os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

# from Synapsify.loadCleanly import sheets as sh

class IdeaNet():

    def __init__(self, directory, filename):
        self._directory = directory
        self._filename  = filename
        self._param_file = os.join(directory,filename)

    #====================================================================
    # LOAD MODEL OPTIONS LOAD MODEL OPTIONS LOAD MODEL OPTIONS
    #====================================================================
    def gen_sent_tvt(self, textcol, sentcol, train_size, test_size):
        # directory = "/Users/dogfish/Documents/Sean/Gonzalez_Associates_LLC/Synapsify/git/IdeaNets/Synapsify_data"
        # filename = "Annotated_Comments_for_Always_Discreet_1.csv"
        self._textcol = textcol
        self._sentcol = sentcol
        self._train_size = train_size
        self._test_size  = test_size

        SP = synapsify_preprocess(self._directory, self._filename, textcol, sentcol, train_size, test_size)
        TT = SP.main() #assume it has 'valid' fields.

        self._params, self._model_options = load_params_data(TT,self._param_file)

    #====================================================================
    # LOAD OR BUILD MODEL LOAD OR BUILD MODEL LOAD OR BUILD MODEL
    #====================================================================
    def gen_model(self):

        if load_model:
            load_model()
        else:
            Model = build_model(self._params, self._model_options)
            lstm_model = Model.main()


    #====================================================================
    # TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM
    #====================================================================
    def train(self):

        LSTM = train_lstm(lstm_model) # Object?

    #====================================================================
    # COMPARE LSTM COMPARE LSTM COMPARE LSTM COMPARE LSTM COMPARE LSTM
    #====================================================================

    tagged_data = LSTM(tag_this_data?)
    calc_lstm_stats(tagged_data)
