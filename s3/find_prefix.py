import boto3
import botocore
import os

# --- Configuration ---
# Replace these placeholders with your actual S3 details.

S3_ENDPOINT_URL = 'https://your-s3-endpoint.example.com'
S3_ACCESS_KEY = 'YOUR_ACCESS_KEY'
S3_SECRET_KEY = 'YOUR_SECRET_KEY'
S3_BUCKET_NAME = 'your-bucket-name'
S3_REGION = 'your-region'
# Path to your custom CA certificate bundle file.
# If you don't need a custom CA, set this to None.
CA_BUNDLE_PATH = '/path/to/your/ca.crt' 

def get_all_s3_prefixes(bucket_name, s3_client):
    """
    Retrieves all unique prefixes from a given S3 bucket.

    In S3, prefixes are the "directory paths" to your objects. For an object with
    the key 'level1/level2/file.txt', this function will identify 'level1/' and
    'level1/level2/' as prefixes.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_client: An initialized boto3 S3 client.

    Returns:
        list: A sorted list of all unique prefixes in the bucket.
    """
    all_prefixes = set()
    
    try:
        # Paginators are used to handle buckets with a large number of objects
        # by making multiple API calls in the background.
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        print("Fetching object list from the bucket. This might take a while for large buckets...")

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # If a key contains a slash, it has at least one prefix.
                    if '/' in key:
                        # Split the key into parts and reconstruct the prefixes.
                        parts = key.split('/')
                        # We iterate up to the last part (the filename)
                        for i in range(1, len(parts)):
                            prefix = '/'.join(parts[:i]) + '/'
                            all_prefixes.add(prefix)

        print("Successfully fetched and processed all objects.")
        return sorted(list(all_prefixes))

    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'NoSuchBucket':
            print(f"Error: The bucket '{bucket_name}' does not exist.")
        elif error_code == 'InvalidAccessKeyId':
            print("Error: The AWS Access Key ID you provided does not exist in our records.")
        elif error_code == 'SignatureDoesNotMatch':
            print("Error: The request signature we calculated does not match the signature you provided. Check your Secret Access Key.")
        else:
            print(f"An unexpected error occurred: {e}")
        return None
    except botocore.exceptions.EndpointConnectionError as e:
        print(f"Error connecting to the endpoint URL '{S3_ENDPOINT_URL}'. Please check the endpoint URL and your network connection.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == '__main__':
    # Initialize the S3 client
    try:
        s3_client_args = {
            'service_name': 's3',
            'aws_access_key_id': S3_ACCESS_KEY,
            'aws_secret_access_key': S3_SECRET_KEY,
            'endpoint_url': S3_ENDPOINT_URL,
            'region_name': S3_REGION,
        }

        # If a CA bundle path is provided, add it to the client arguments for verification.
        if CA_BUNDLE_PATH:
            if not os.path.exists(CA_BUNDLE_PATH):
                 print(f"Error: The specified CA bundle file does not exist at '{CA_BUNDLE_PATH}'")
                 exit()
            s3_client_args['verify'] = CA_BUNDLE_PATH

        s3_client = boto3.client(**s3_client_args)

        # Get and print all prefixes
        prefixes = get_all_s3_prefixes(S3_BUCKET_NAME, s3_client)

        if prefixes is not None:
            if prefixes:
                print("\n--- Found Prefixes ---")
                for prefix in prefixes:
                    print(prefix)
                print(f"\nTotal unique prefixes found: {len(prefixes)}")
            else:
                print("\nNo prefixes found in the bucket.")

    except Exception as e:
        print(f"Failed to initialize S3 client: {e}")
