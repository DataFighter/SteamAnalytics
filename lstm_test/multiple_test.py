
import os, inspect
import math
import synapsify_generatepkl_function
### import synapsify_generatepkl_function_shuffled

###  The code is used to generate multiple datasets.
def main():
    ### Get the directory of Synapsify data
    directory = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "../Synapsify_data")))

    ### Get the comment column, and we just assume the first column are all comments
    textcol = 0

    ### Get the sentiment column, and we assume the sixth column are all sentiments
    sentcol = 5


    ### Iteratively get the dataset file name from the directory "Synapsify_data", and preprocess the data
    for ff in os.listdir(directory):
        if ff.endswith(".csv"):

            ### Get how many lines of the dataset, so we can get seperate both training and testing data
            getfile = os.path.join(directory, ff)
            num_lines = sum(1 for line in open(getfile))

            ### Get the training and testing dataset
            num_instances = num_lines - 1
            train_size = int(math.ceil(num_lines / 2.0))
            test_size = int(num_instances - train_size)

            ### Use the original dataset order without shuffling
            synapsify_generatepkl_function.preprocess(directory, ff, textcol, sentcol, train_size, test_size)
            ### Shuffled dataset:
            ###synapsify_generatepkl_function_shuffled.preprocess(directory, ff, textcol, sentcol, train_size, test_size)


if __name__ == "__main__":
    main()