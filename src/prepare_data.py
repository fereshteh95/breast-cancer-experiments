from pathlib import Path
import glob
import ast
import pandas as pd
import numpy as np
import click
import shutil
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


@click.command()
@click.argument('test_size', type=float, default=0.3)
def main(test_size: float):
    config_file_path = Path('../config.yaml')
    config = OmegaConf.load(config_file_path)

    random_seed = config.general_info.random_seed
    class_names = config.general_info.classes
    view = config.data.view
    path_to_data_1 = config.data.path_to_data_1
    path_to_data_2 = config.data.path_to_data_2
    path_to_labels_1 = config.data.path_to_labels_1
    path_to_labels_2 = config.data.path_to_labels_2

    data_dir_1 = '../data/' + path_to_data_1.split('/')[-1].split('.')[0]
    data_dir_2 = '../data/' + path_to_data_2.split('/')[-1].split('.')[0]
    train_csv_file = config.data_sequence.train_csv_file
    validation_csv_file = config.data_sequence.validation_csv_file
    test_csv_file = config.data_sequence.test_csv_file

    shutil.unpack_archive(path_to_data_1, data_dir_1)
    shutil.unpack_archive(path_to_data_2, data_dir_2)

    label_dict = {
        'Benign': 0,
        'Malignant': 1,
        # 'Benign mass': [0, 0, 1, 0],
        # 'Malignant mass': [0, 0, 0, 1]
    }
    df_1 = pd.read_csv(path_to_labels_1)
    df_2 = pd.read_csv(path_to_labels_2)
    df = df_1.append(df_2, ignore_index=True)
    # df = df.loc[df['Type'] != 'both']
    df = df.loc[(df['Type'] == 'mass') | (df['Type'] == 'both')]
    df = df.loc[df['Training_Tag'] == 'Train'].reset_index()

    label = []

    for i in range(len(df)):
        label.append(label_dict[df['Pathology'].values[i]])

    df[class_names] = label

    x = list(df.index)
    y = df[class_names].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)

    columns = ['imgfile']
    columns.extend(class_names)

    if len(class_names) == 1:
        label_columns = class_names[0]
    else:
        label_columns = columns[1:]

    x_names = []
    labels = []
    for i in range(len(x_train)):
        df_i = df.iloc[x_train[i]]

        x = df_i[view]
        y = df_i[class_names]

        x_names.append(x)
        labels.append(y)

    train_df = pd.DataFrame(index=np.arange(len(x_names)), columns=columns)
    train_df['imgfile'] = x_names
    train_df[label_columns] = labels

    x_names = []
    labels = []
    for i in range(len(x_test)):
        df_i = df.iloc[x_test[i]]

        x = df_i[view]
        y = df_i[class_names]

        x_names.append(x)
        labels.append(y)

    val_df = pd.DataFrame(index=np.arange(len(x_names)), columns=columns)
    val_df['imgfile'] = x_names
    val_df[label_columns] = labels

    df_1 = pd.read_csv(path_to_labels_1)
    df_2 = pd.read_csv(path_to_labels_2)
    df = df_1.append(df_2, ignore_index=True)
    # df = df.loc[df['Type'] != 'both']
    df = df.loc[(df['Type'] == 'mass') | (df['Type'] == 'both')]
    df = df.loc[df['Training_Tag'] == 'Evaluation'].reset_index()

    label = []

    for i in range(len(df)):
        label.append(label_dict[df['Pathology'].values[i]])

    df[class_names] = label

    x_names = []
    labels = []
    for i in range(len(df)):
        df_i = df.iloc[i]

        x = df_i[view]
        y = df_i[class_names]

        x_names.append(x)
        labels.append(y)

    test_df = pd.DataFrame(index=np.arange(len(x_names)), columns=columns)
    test_df['imgfile'] = x_names
    test_df[label_columns] = labels

    train_df.to_csv(train_csv_file, index=False)
    val_df.to_csv(validation_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)


if __name__ == '__main__':
    main()
