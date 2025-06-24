install:
	pip install -e .

test:
	pytest tests/

simulate-data:
	python -c "from ecog_eeg_dl_classifier.data.simulators.synthetic_signal import save_synthetic_dataset; save_synthetic_dataset(n_samples=1000)"

train:
	python ecog_eeg_dl_classifier/sagemaker/train_script.py

export:
	python model_export.py

deploy:
	python ecog_eeg_dl_classifier/sagemaker/deploy_model.py

notebook:
	jupyter notebook notebooks/

clean:
	rm -rf __pycache__ *.pyc *.pyo *.egg-info build dist model/export/*.pth model/export/*.tar.gz
