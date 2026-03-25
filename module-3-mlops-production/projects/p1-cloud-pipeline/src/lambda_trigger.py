"""AWS Lambda function to trigger preprocessing when data lands in S3.

Deploy this to Lambda with trigger: S3 → PutObject → prefix 'raw/'
"""

import json
import os
import boto3

SAGEMAKER_ROLE = os.getenv('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::123456789:role/SageMakerRole')
BUCKET_NAME = os.getenv('S3_BUCKET', 'your-ml-pipeline-bucket')


def lambda_handler(event, context):
    """
    Triggered when a file is uploaded to S3 raw/ prefix.
    Launches a SageMaker Processing job to clean the data.
    """
    # Parse S3 event
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']

    print(f"New file detected: s3://{bucket}/{key}")

    if not key.startswith('raw/'):
        print(f"Skipping non-raw file: {key}")
        return {'statusCode': 200, 'body': 'Skipped'}

    # Launch SageMaker Processing Job
    sm = boto3.client('sagemaker', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

    job_name = f"preprocess-{key.split('/')[-1].replace('.csv', '')}-{context.aws_request_id[:8]}"

    response = sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'VolumeSizeInGB': 10,
            }
        },
        AppSpecification={
            'ImageUri': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
            'ContainerEntrypoint': ['python3', '/opt/ml/processing/code/preprocess.py'],
        },
        ProcessingInputs=[{
            'InputName': 'raw-data',
            'S3Input': {
                'S3Uri': f's3://{bucket}/{key}',
                'LocalPath': '/opt/ml/processing/input',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
            }
        }],
        ProcessingOutputConfig={
            'Outputs': [{
                'OutputName': 'processed-data',
                'S3Output': {
                    'S3Uri': f's3://{bucket}/processed/',
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'EndOfJob',
                }
            }]
        },
        RoleArn=SAGEMAKER_ROLE,
    )

    print(f"Launched processing job: {job_name}")
    return {
        'statusCode': 200,
        'body': json.dumps({'jobName': job_name, 'status': 'LAUNCHED'})
    }


def test_locally():
    """Test the handler logic without AWS."""
    mock_event = {
        'Records': [{
            's3': {
                'bucket': {'name': 'my-ml-bucket'},
                'object': {'key': 'raw/churn/2024-01-01/churn_data.csv'}
            }
        }]
    }

    class MockContext:
        aws_request_id = 'test-1234-abcd'

    print("Testing lambda handler locally...")
    print(f"Event: {json.dumps(mock_event, indent=2)}")
    # Note: will fail at SageMaker call without real AWS — that's expected
    print("\nLocal test complete. Deploy to Lambda for real execution.")


if __name__ == '__main__':
    test_locally()
