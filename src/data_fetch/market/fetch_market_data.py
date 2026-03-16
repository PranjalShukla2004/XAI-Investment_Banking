import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import date, timedelta

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='ef80de00-cdfa-4db5-bf5d-52394ce27b2a',
  aws_secret_access_key='jGEmpBQg4nzN0bpfWRMhB9DSmxtXpA6x',
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.massive.com',
  config=Config(signature_version='s3v4'),
)

# Specify the bucket name
bucket_name = 'flatfiles'

start_date = date(2021, 2, 1)
end_date = date(2022, 3, 1)
current_date = start_date

while current_date <= end_date:
  # Specify the S3 object key name
  object_key = (
    f"flatfiles/us_stocks_sip/day_aggs_v1/"
    f"{current_date:%Y}/{current_date:%m}/{current_date:%Y-%m-%d}.csv.gz"
  )

  # Remove the bucket name (e.g. 'flatfiles/') prefix if present in object_key
  if object_key.startswith(bucket_name + '/'):
    object_key = object_key[len(bucket_name + '/'):]

  # Specify the local file name and path to save the downloaded file
  local_file_name = object_key.split('/')[-1]  # e.g., '2025-06-12.csv.gz'
  local_file_path = './' + local_file_name

  # Print the file being downloaded
  print(f"Downloading file '{object_key}' from bucket '{bucket_name}'...")

  # Download the file
  try:
    s3.download_file(bucket_name, object_key, local_file_path)
  except ClientError as exc:
    error_code = str(exc.response.get('Error', {}).get('Code', ''))
    if error_code in {'404', 'NoSuchKey', 'NotFound'}:
      print(f"Skipping missing file '{object_key}'")
      current_date += timedelta(days=1)
      continue
    raise
  current_date += timedelta(days=1)
