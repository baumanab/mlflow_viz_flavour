"""
The `mlflow_vega` module provides an API for logging and loading Vega models. This module
exports Vega models with the following flavors:

Vega (native) format
    This is the main flavor that can be loaded back into Vega.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
# Standard Libraries
import logging
import os
import shutil
import json
import importlib
import pkgutil

# External Libraries
import yaml
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow import pyfunc
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.utils.annotations import keyword_only
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import DIRECTORY_NOT_EMPTY
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


# Internal Libraries
import mlflow_vega
import mlflow_vega.styles


FLAVOR_NAME = 'mlflow_vega'

_SERIALIZED_VEGA_MODEL_FILE_NAME = ''
_PICKLE_MODULE_INFO_FILE_NAME = 'pickle_module_info.txt'

_logger = logging.getLogger(__name__)


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


_discovered_styles = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(mlflow_vega.styles)
}


def discovered_styles():
    return _discovered_styles


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=['altair==4.1.0', *_discovered_styles.keys()],
        additional_conda_channels=None,
    )


@keyword_only
def log_model(
    vega_saved_model_dir,
    artifact_path,
    conda_env=None,
    signature=None,
    input_example=None,
    registered_model_name=None,
):
    """

    :return:
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_vega,
        vega_saved_model_dir=vega_saved_model_dir,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )


@keyword_only
def save_model(
    vega_saved_model_dir,
    path,
    mlflow_model=None,
    conda_env=None,
    signature=None,
    input_example=None,
):
    """
    """
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path), DIRECTORY_NOT_EMPTY)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        # _save_example(mlflow_model, input_example, path)
        pass
    root_relative_path = _copy_file_or_tree(src=vega_saved_model_dir, dst=path, dst_dir=None)
    model_dir_subpath = 'vegamodel'
    shutil.move(os.path.join(path, root_relative_path), os.path.join(path, model_dir_subpath))

    conda_env_subpath = 'conda.yaml'
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, 'r') as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        saved_model_dir=model_dir_subpath,
    )
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow_vega", env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(model_uri):
    """
    """
    import altair.vegalite.v4 as vg

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    local_spec_path = os.path.join(local_model_path, 'vegamodel', 'spec.json')
    local_conf_path = os.path.join(local_model_path, 'vegamodel', 'config.json')
    with open(local_spec_path, 'r') as file:
        spec = json.load(file)

    with open(local_conf_path, 'r') as file:
        conf = json.load(file)

    # initial vegalite style
    return vg.VegaLite(spec)


"""
# Command Line:

export MLFLOW_TRACKING_URI='http://localhost:5000'
mlflow ui --backend-store-uri sqlite:///mlflow.db
"""


"""
# Standard Libraries
import os
import uuid

# External Libraries
import mlflow
import altair_viewer

# Internal Libraries
import mlflow_vega


os.environ['MLFLOW_TRACKING_URI']='http://localhost:5000'
try:
    mlflow.create_experiment('example')
except Exception as error:
    print(error)

experiment = mlflow.get_experiment_by_name('example')
run_id = str(uuid.uuid4())
with mlflow.start_run(run_name=f'example_{run_id}', experiment_id=experiment.experiment_id) as run:
    model = mlflow_vega.log_model(vega_saved_model_dir='spec', artifact_path='model')


model = mlflow_vega.load_model(os.path.join(run.info.artifact_uri, 'model'))
altair_viewer.display(model.spec)
"""



"""
# Idea #1: declare a style in the flavor?

def load_viz(style):
    styles = {
        'vega': 'altair',
        # ...
    }
    
    try:
        import styles[style]
    except ImportError as error:
        pass
"""

"""
# Idea #2: saving a loading closure

save_model(model, pickle_able_func_to_load_model):
    # ...
"""
