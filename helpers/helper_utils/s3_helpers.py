import boto3
import os

def download_from_s3(s3_bucket,source_key,destination_path):
    s3_client = boto3.client('s3')
    s3_client.download_file(s3_bucket, source_key, destination_path)


def upload_to_s3(bucket_name,destination_key,source_file_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(source_file_path, bucket_name, destination_key)

def download_folder_contents_from_s3(bucket_name,source_key,destination_directory):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=source_key):
        if obj.key[-2:] == "pt":
            destination_file = os.path.join(destination_directory,obj.key.split("/")[-1])
            bucket.download_file(obj.key,destination_file)

def sync_to_S3_command(source,destination_bucket,destination_key):
    return "aws s3 sync {} s3://{}/{}".format(source,destination_bucket,destination_key)

