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
@click.argument('path_to_data', type=str, default='')
@click.argument('path_to_labels', type=str, default='')
@click.argument('test_size', type=float, default=0.3)
def main(path_to_data: str, path_to_labels: str, test_size: float):
    config_file_path = Path('../config.yaml')
    config = OmegaConf.load(config_file_path)

    random_seed = config.general_info.random_seed
    class_names = config.general_info.classes
    data_dir = '../data/'+path_to_data.split('/')[-1].split('.')[0]
    train_csv_file = config.data_sequence.train_csv_file
    validation_csv_file = config.data_sequence.validation_csv_file
    test_csv_file = config.data_sequence.test_csv_file

    shutil.unpack_archive(path_to_data, data_dir)

    label_dict = {
        'background': [1, 0, 0, 0, 0],
        'Benign calcification': [0, 1, 0, 0, 0],
        'Malignant calcification': [0, 0, 1, 0, 0],
        'Benign mass': [0, 0, 0, 1, 0],
        'Malignant mass': [0, 0, 0, 0, 1]
    }
    df = pd.read_csv(path_to_labels)
    df = df.loc[df['Training_Tag'] == 'Train'].reset_index()

    x = list(df.index)
    y = df['Label'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)

    columns = ['imgfile']
    columns.extend(class_names)

    train_imgs = []
    train_labels = []
    for i in range(len(x_train)):
        df_i = df.iloc[x_train[i]]

        abn = ast.literal_eval(df_i['PatchFolder_abn'])
        img = []
        for j in range(len(abn)):
            img.extend(glob.glob('../data/' + abn[j] + '*.jpg'))
        label = [label_dict[df_i['Label']] for _ in range(len(img))]
        train_imgs.extend(img)
        train_labels.extend(label)

        bck = ast.literal_eval(df_i['PatchFolder_bck'])
        img = []
        for j in range(len(bck)):
            img.extend(glob.glob('../data/' + bck[j] + '*.jpg'))
        label = [label_dict['background'] for _ in range(len(img))]

        train_imgs.extend(img)
        train_labels.extend(label)

    train_df = pd.DataFrame(index=np.arange(len(train_imgs)), columns=columns)
    train_df['imgfile'] = train_imgs
    train_df[columns[1:]] = train_labels

    val_imgs = []
    val_labels = []
    for i in range(len(x_test)):
        df_i = df.iloc[x_test[i]]

        abn = ast.literal_eval(df_i['PatchFolder_abn'])
        img = []
        for j in range(len(abn)):
            img.extend(glob.glob('../data/' + abn[j] + '*.jpg'))
        label = [label_dict[df_i['Label']] for _ in range(len(img))]
        val_imgs.extend(img)
        val_labels.extend(label)

        bck = ast.literal_eval(df_i['PatchFolder_bck'])
        img = []
        for j in range(len(bck)):
            img.extend(glob.glob('../data/' + bck[j] + '*.jpg'))
        label = [label_dict['background'] for _ in range(len(img))]

        val_imgs.extend(img)
        val_labels.extend(label)

    val_df = pd.DataFrame(index=np.arange(len(val_imgs)), columns=columns)
    val_df['imgfile'] = val_imgs
    val_df[columns[1:]] = val_labels

    df = pd.read_csv(path_to_labels)
    df = df.loc[df['Training_Tag'] == 'Evaluation'].reset_index()

    columns = ['imgfile']
    columns.extend(class_names)

    test_imgs = []
    test_labels = []
    for i in range(len(df)):
        df_i = df.iloc[i]

        abn = ast.literal_eval(df_i['PatchFolder_abn'])
        img = []
        for j in range(len(abn)):
            img.extend(glob.glob('../data/' + abn[j] + '*.jpg'))
        label = [label_dict[df_i['Label']] for _ in range(len(img))]
        test_imgs.extend(img)
        test_labels.extend(label)

        bck = ast.literal_eval(df_i['PatchFolder_bck'])
        img = []
        for j in range(len(bck)):
            img.extend(glob.glob('../data/' + bck[j] + '*.jpg'))
        label = [label_dict['background'] for _ in range(len(img))]

        test_imgs.extend(img)
        test_labels.extend(label)

    test_df = pd.DataFrame(index=np.arange(len(test_imgs)), columns=columns)
    test_df['imgfile'] = test_imgs
    test_df[columns[1:]] = test_labels

    train_df.to_csv(train_csv_file, index=False)
    val_df.to_csv(validation_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)


if __name__ == '__main__':
    main()


