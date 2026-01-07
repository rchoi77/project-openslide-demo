import boto3

def get_filenames_from_s3(bucket: str, prefix: str, output_file: str):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    with open(output_file, 'w') as f:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                f.write(obj['Key'].replace('morphle-tiff/', '') + '\n')

if __name__ == "__main__":
    get_filenames_from_s3(bucket='morphle-epivara-wsi', prefix='morphle-tiff/', output_file='slides_to_process.txt')
    print(f"Filenames saved to slides_to_process.txt")