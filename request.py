import boto3
import botocore
import re

BUCKET_NAME = 'linearaccelerationdata' # replace with your bucket name
KEY = 'public/2019-01-25T12-52-58' # replace with your object key

s3 = boto3.resource('s3')


my_bucket = s3.Bucket(BUCKET_NAME)
file_names = []
files_to_be_downloaded = []
for object in my_bucket.objects.all():
    file=re.findall('\S*2019-01-25\S*',object.key)
    name=re.findall('2019-01-25\S*',object.key)
    if(file):
    	files_to_be_downloaded.append(file)
    	file_names.append(name)


for index in range(5,18):
	try:
	    s3.Bucket(BUCKET_NAME).download_file(''.join(files_to_be_downloaded[index]), ''.join(file_names[index]))
	except botocore.exceptions.ClientError as e:
	    if e.response['Error']['Code'] == "404":
	        print("The object does not exist.")
	    else:
	        raise