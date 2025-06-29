import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model = PyTorchModel(
    model_data="s3://your-bucket/your-model.tar.gz",
    role=role,
    entry_point="ecog_eeg_dl_classifier/sagemaker/train_script.py",
    framework_version="1.12",
    py_version="py38"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("Model deployed to endpoint:", predictor.endpoint_name)
