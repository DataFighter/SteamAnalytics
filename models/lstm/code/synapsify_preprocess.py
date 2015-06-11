"""
This file is designed to ingest Synapsify standard tagged data sets and convert them to LSTM input format

Input:
    1. Directory and filename of tagged dataset to be converted
    2. Token dictionary - text file where each row is a new word

Output:
    LSTM intput file structure - 2xN array
        Columns:
            2x1 vector
        Rows:
            1st row: vector of indices to token dictionary
            2nd row: total sentiment of that vector
"""

from Synapsify.loadCleanly import sheets as sh

def main(directory, filename):

    header,rows = sh.get_spreadsheet_rows(directory+filename,0)

if __name__ == '__main__':
    directory = sys.argv[0]
    filename  = sys.argv[1]
    main(directory, filename)