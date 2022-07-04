from pathlib import Path
import pandas as pd
import click
import shutil
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


@click.command()
@click.argument('path_to_data', type=str, default='')
@click.argument('path_to_labels', type=str, default='')
@click.argument('test_size', type=float, default=0.2)
def main(path_to_data: str, path_to_labels: str, test_size: float):
    config_file_path = Path('../config.yaml')
    config = OmegaConf.load(config_file_path)

    random_seed = config.general_info.random_seed
    class_names = config.general_info.classes
    data_dir = '../data/'+path_to_data.split('/')[-1].split('.')[0]
    train_csv_file = config.data_sequence.train_csv_file
    validation_csv_file = config.data_sequence.validation_csv_file

    shutil.unpack_archive(path_to_data, data_dir)

    df = pd.read_csv(path_to_labels)
    x = list(df.index)
    y = df[class_names].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)

    train_df = df.iloc[x_train]
    val_df = df.iloc[x_test]

    train_df.to_csv(train_csv_file, index=False)
    val_df.to_csv(validation_csv_file, index=False)


if __name__ == '__main__':
    main()


