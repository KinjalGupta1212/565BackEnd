import boto3
import os

s3 = boto3.client("s3")

def upload_directory(directory, bucket):
    for root, _, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, directory)

            s3.upload_file(local_path, bucket, s3_path)

if __name__ == "__main__":
    upload_directory("./chroma_db", "rag-chatbot-bucket-565")