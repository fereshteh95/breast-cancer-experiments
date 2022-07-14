"""

    put the entire evaluation pipeline here
    this file uses the config.yaml to create the model, prepare and load data
    and all the other necessary steps to start an evaluation job and report all the necessary metrics
    these metrics are used to create the model-card corresponding to each training-job

"""
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf

from data_pipeline import DataLoader
from utils import setup_mlflow_active_run

from evaluation import Evaluator


def main():
    config_file_path = Path('../config.yaml')
    config = OmegaConf.load(config_file_path)
    evaluator = Evaluator(config)

    model = tf.keras.models.load_model(config.general_info.best_weights_path)
    data_loader = DataLoader(config)
    test_data_gen, _ = data_loader.create_test_generator()
    active_run = setup_mlflow_active_run(config_path=config_file_path, is_evaluation=True)

    evaluator.evaluate(model=model,
                       test_data_gen=test_data_gen,
                       active_run=active_run
                       )


if __name__ == '__main__':
    main()
