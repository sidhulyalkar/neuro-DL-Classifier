from setuptools import setup, find_packages

setup(
    name="ecog-eeg-dl-classifier",
    version="0.1.0",
    description="Deep learning toolkit for ECoG/EEG signal classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "boto3",
        "sagemaker"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "generate-synthetic-data=ecog_eeg_dl_classifier.data.simulators.synthetic_signal:save_synthetic_dataset"
        ]
    }
)
