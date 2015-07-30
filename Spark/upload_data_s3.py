import os, inspect
import boto
from boto.s3.connection import Location

### This code is to upload the testing data onto AWS S3.

def main():
	### Connect to S3.
	s3 = boto.connect_s3()
	bucket = s3.get_bucket("synapsify-ruofan")	### Put the bucket name here.

	### Current directory path, and specify the data directory.
	curr_dir = os.path.split(inspect.getfile(inspect.currentframe()))[0]
	data_dir = os.path.realpath(os.path.abspath(os.path.join(curr_dir, "../../../Synapsify_data/")))

	#for filename in nameList:
	for ff in os.listdir(data_dir):
		# sourcePath = os.path.join(sourceDir, filename)
		# destPath = os.path.join("data/", filename)
		sourcePath = os.path.join(data_dir, ff)
		destPath = os.path.join("Synapsify_data/", ff)
		k = boto.s3.key.Key(bucket)
		k.key = destPath
		k.set_contents_from_filename(sourcePath)

if __name__ == "__main__":
	main()