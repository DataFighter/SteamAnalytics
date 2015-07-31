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
  ### Convert the string to the file object, and we need to import StringIO in the code.
  data = StringIO(content)

  ### Read data from S3.
  cr = csv.reader(data)
  num_lines = sum(1 for line in cr)
  num_instances = num_lines - 1   ### The first line shouldn't be considered.
  train_size = int(math.ceil(num_lines / 2.0))
  test_size = int(num_instances - train_size)

  ### Create an instance of lstm class
  run_lstm = lstm()
  run_lstm.params['data_file'] = data  ### Update the lstm
  run_lstm.params['train_size'] = train_size
  run_lstm.params['test_size'] = test_size

  run_lstm.build_model()
  run_lstm.train_model()
  run_lstm.test_model()


def main():
  ### Initialize the SparkConf and SparkContext

  '''
  conf = SparkConf().setAppName("ruofan").setMaster("local")
  sc = SparkContext(conf = conf)
  datafile = sc.wholeTextFiles("s3n://synapsify-ruofan/Synapsify_data", use_unicode=False) ### Read data directory from S3 storage.

  ### Sent the application in each of the slave node
  temp = datafile.foreach(lambda (path, content): myfunc(path, content))
  '''
  run_lstm = lstm()
  run_lstm.build_model()
  run_lstm.train_model()
  run_lstm.test_model()

if __name__ == "__main__":
  main()
