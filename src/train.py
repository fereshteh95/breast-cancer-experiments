"""

    put the entire training pipeline here
    this file uses the config.yaml to create the model, prepare data, create data generators
    and all the other necessary steps to start a training job
    Note: All training jobs should be logged my mlflow

"""
from pathlib import Path
from omegaconf import OmegaConf

from model_factory import PatchModelBuilder
from data_pipeline import DataLoader
from utils import setup_mlflow_active_run


from training import Trainer


def main():
    config_file_path = Path('../config.yaml')
    config = OmegaConf.load(config_file_path)
    run_dir = Path('../run')
    exported_dir = Path('../exported')
    trainer = Trainer(config, run_dir, exported_dir)

    model_builder = PatchModelBuilder(config, phase='train')
    compiled_model = model_builder.get_model()
    if config.info_training.use_best_weights:
        compiled_model.load_weights(config.general_info.best_weights_path)
    callbacks = model_builder.get_callbacks()

    data_loader = DataLoader(config)
    train_data_gen, n_iter_train = data_loader.create_training_generator()
    val_data_gen, n_iter_val = data_loader.create_validation_generator()
    class_weight = data_loader.get_class_weight()
    setup_mlflow_active_run(config_path=config_file_path, is_evaluation=False)

    trainer.train(model=compiled_model,
                  train_data_gen=train_data_gen,
                  n_iter_train=n_iter_train,
                  val_data_gen=val_data_gen,
                  n_iter_val=n_iter_val,
                  class_weight=class_weight,
                  callbacks=callbacks)


if __name__ == '__main__':
    main()
