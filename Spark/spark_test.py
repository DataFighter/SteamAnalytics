import os, sys, inspect, csv, math
from StringIO import StringIO

### Note: Please set-up the environment variables before running the code:
### AWS_SECRET_ACCESS_KEY=...
### AWS_ACCESS_KEY_ID=...

### Current directory path.
curr_dir = os.path.split(inspect.getfile(inspect.currentframe()))[0]

### Setup the environment variables
spark_home_dir = os.path.realpath(os.path.abspath(os.path.join(curr_dir, "../../../../AWS_Tutorial/spark-1.4.0/")))
python_dir = os.path.realpath(os.path.abspath(os.path.join(spark_home_dir, "./python")))
os.environ["SPARK_HOME"] = spark_home_dir
os.environ["PYTHONPATH"] = python_dir

### Setup pyspark directory path
pyspark_dir = os.path.realpath(os.path.abspath(os.path.join(spark_home_dir, "./python")))
sys.path.append(pyspark_dir)

### Setup the scode directory
scode_dir = os.path.realpath(os.path.abspath(os.path.join(curr_dir, "../models/lstm/scode")))
sys.path.append(scode_dir)

### Setup the Synapsify directory
synapsify_dir = os.path.realpath(os.path.abspath(os.path.join(curr_dir, "../../Synapsify")))
sys.path.append(synapsify_dir)

### from load_params import Load_LSTM_Params
from lstm_class import LSTM as lstm

### Import the pyspark
from pyspark import SparkConf, SparkContext

### myfunc is to print the frist row for testing purpose.
def myfunc(path, content):
  ### Convert the string to the file object, and we need to import StringIO in the code.
  data = StringIO(content)    
  
  cr = csv.reader(data)
  num_lines = sum(1 for line in cr)
  num_instances = num_lines - 1   ### The first line shouldn't be considered.
  train_size = int(math.ceil(num_lines / 2.0))
  test_size = int(num_instances - train_size)
  print "The total lines of ", path, " is: ", num_lines
  print "The training size is ", train_size
  print "The testing size is ", test_size

  for row in cr:
    #print "The first row of ", path, " is: ", row
    break

def lstm_test(path, content):
  data = StringIO(content)

  ### Read data from S3.
  cr = csv.reader(data)
  num_lines = sum(1 for line in cr)
  num_instances = num_lines - 1   ### The first line shouldn't be considered.
  train_size = int(math.ceil(num_lines / 2.0))
  test_size = int(num_instances - train_size)

  ### Create an instance of lstm class
  params = {}
  params['raw_rows'] = content  ### Update the lstm
  params['train_size'] = train_size
  params['test_size'] = test_size

  run_lstm = lstm(params=params)
  run_lstm.build_model()
  run_lstm.train_model()
  run_lstm.test_model()


def main():
  ### Initialize the SparkConf and SparkContext

  ### Locations of Python files.
  sheets_loc = '/home/ying/Deep_Learning/Synapsify/Synapsify/loadCleanly/sheets.py'
  lstm_class_loc = '/home/ying/Deep_Learning/IdeaNets/IdeaNets/models/lstm/scode/lstm_class.py'
  load_params_loc = '/home/ying/Deep_Learning/IdeaNets/IdeaNets/models/lstm/scode/load_params.py'
  preprocess_loc = '/home/ying/Deep_Learning/IdeaNets/IdeaNets/models/lstm/scode/synapsify_preprocess.py'

  ### Pass Python files to Spark.
  pyFiles = []
  pyFiles.append(sheets_loc)
  pyFiles.append(lstm_class_loc)
  pyFiles.append(load_params_loc)
  pyFiles.append(preprocess_loc)


  ### Initialize the spark configuration.
  conf = SparkConf().setAppName("ruofan").setMaster("local")
  sc = SparkContext(conf = conf, pyFiles=pyFiles)

  ### Add non-python files passing to Spark.
  sc.addFile('/home/ying//AWS_Tutorial/spark-1.4.0/bin/en.zip')
  sc.addFile('/home/ying/Deep_Learning/IdeaNets/IdeaNets/models/lstm/scode/tokenizer.perl')
  sc.addFile('/home/ying/Deep_Learning/Synapsify/Synapsify/loadCleanly/stopwords.txt')
  sc.addFile('/home/ying/Deep_Learning/Synapsify/Synapsify/loadCleanly/prepositions.txt')


  datafile = sc.wholeTextFiles("s3n://synapsify-ruofan/Synapsify_data", use_unicode=False) ### Read data directory from S3 storage.

  ### Sent the application in each of the slave node
  datafile.foreach(lambda (path, content): lstm_test(path, content))

if __name__ == "__main__":
  main()
