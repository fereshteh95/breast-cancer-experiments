from pathlib import Path

import yaml
import mlflow
from mlflow.entities import ViewType
from git import Repo


MLFLOW_TRACKING_URI = "http://185.110.190.127:7080/"


def setup_mlflow_active_run(config_path: Path,
                            is_evaluation: bool):
    """setup the MLFlow"""

    repo = Repo('../')
    root = Path(repo.working_tree_dir).name
    experiment_name = root + '/' + repo.active_branch.name

    mlflow.end_run()
    active_run = _setup_mlflow(mlflow_experiment_name=experiment_name)

    if is_evaluation:
        sess_type = 'evaluation'
    else:
        sess_type = 'training'

    mlflow.set_tag("session_type", sess_type)  # ['hpo', 'evaluation', 'training']
    try:
        # config = load_config_as_dict(path=config_path)
        # _add_config_file_to_mlflow(config)
        mlflow.log_artifact(str(config_path))
    except Exception as e:
        print(f'exception when logging config file to mlflow: {e}')
    try:
        mlflow.log_param('project name', experiment_name)
    except Exception as e:
        print(f'exception when logging project name to mlflow: {e}')

    return active_run


def _setup_mlflow(mlflow_experiment_name: str,
                  mlflow_tracking_uri: str = MLFLOW_TRACKING_URI) -> mlflow.ActiveRun:
    """Sets up mlflow and returns an ``active_run`` object.

    tracking_uri/
        experiment_id/
            run1
            run2
            ...

    Args:
        mlflow_tracking_uri: ``tracking_uri`` for mlflow
        mlflow_experiment_name: ``experiment_name`` for mlflow, use the same ``experiment_name`` for all experiments
        related to the same task, i.e. the repository name.

    Returns:
        active_run: an ``active_run`` object to use for mlflow logging.

    """

    client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)
    experiments = client.list_experiments(view_type=ViewType.ALL)
    if mlflow_experiment_name not in [i.name for i in experiments]:
        print(f'creating a new experiment: {mlflow_experiment_name}')
        experiment = client.create_experiment(name=mlflow_experiment_name)
    else:
        experiment = [i for i in experiments if i.name == mlflow_experiment_name][0]
        if experiment.lifecycle_stage != 'active':
            print(f'experiment {mlflow_experiment_name} exists but is not active, restoring ...')
            client.restore_experiment(experiment.experiment_id)
            print(f'restored {mlflow_experiment_name}')

    print(f'Exp ID: {experiment.experiment_id}')
    print(f'Exp Name: {experiment.name}')
    print(f'Exp Artifact Location: {experiment.artifact_location}')
    print(f'Exp Tags: {experiment.tags}')
    print(f'Exp Lifecycle Stage: {experiment.lifecycle_stage}')
    # if experiment is not None:
    #     experiment_id = experiment.experiment_id
    # else:
    #     experiment_id = mlflow.create_experiment(mlflow_experiment_name)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    active_run = mlflow.start_run(experiment_id=experiment.experiment_id)
    # active_run = mlflow.start_run(experiment_id=experiment.experiment_id)

    return active_run


def _add_config_file_to_mlflow(config_dict: dict):
    """Adds parameters from config file to mlflow.

    Args:
        config_dict: config file as nested dictionary
    """

    def param_extractor(dictionary):

        """Returns a list of each item formatted like 'trainer.mlflow.tracking_uri: /tracking/uri' """

        values = []
        if dictionary is None:
            return values

        for key, value in dictionary.items():
            if isinstance(value, dict):
                items_list = param_extractor(value)
                for i in items_list:
                    values.append(f'{key}.{i}')
            else:
                values.append(f'{key}: {value}')
        return values

    fields_to_ignore = ['model_details', 'model_parameters', 'considerations']
    new_config = {k: v for k, v in config_dict.items() if k not in fields_to_ignore}
    str_params = param_extractor(new_config)
    params = {}
    for item in str_params:
        name = f"config_{item.split(':')[0]}"
        item_value = item.split(': ')[-1]

        params[name] = item_value

    mlflow.log_params(params)


def load_config_as_dict(path: Path) -> dict:
    """
    loads the ``yaml`` config file and returns a dictionary

    Args:
        path: path to json config file

    Returns:
        a nested object in which parameters are accessible using dot notations, for example ``config.model.optimizer.lr``

    """

    with open(path) as f:
        data_map = yaml.safe_load(f)
    return data_map
