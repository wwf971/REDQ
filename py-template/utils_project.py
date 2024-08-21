from __future__ import annotations
import sys, os, pathlib
DirPathProject = os.path.dirname(os.path.realpath(__file__)) + "/"
DirPathCurrent = os.path.dirname(os.path.realpath(__file__)) + "/"
DirPathParent = pathlib.Path(DirPathCurrent).parent.absolute().__str__() + "/"
DirPathGrandParent = pathlib.Path(DirPathParent).parent.absolute().__str__() + "/"
DirPathGreatGrandParent = pathlib.Path(DirPathGrandParent).parent.absolute().__str__() + "/"
sys.path += [
    DirPathCurrent, DirPathParent, DirPathGrandParent, DirPathGreatGrandParent
]

from _utils_import import DLUtils
import _utils_project

def create_experiment_dir_path(config):
    """
    data of different experiment instance will be saved under project_dir_path.
    data of each experiment instance will be contained in one specific folder under project_dir_path
    """
    from template_py.utils_project import DirPathProject
    experiment_dir_path = DirPathProject + "0-experiments/"
    DLUtils.file.EnsureDir(experiment_dir_path)
    _utils_project.CleanEmptySubDir(experiment_dir_path)
    config.experiment_dir_path = experiment_dir_path

def get_experiment_dir_path(config):
    if not hasattr(config, "experiment_dir_path"):
        create_experiment_dir_path(config)
    return config.experiment_dir_path

def create_experiment_instance_dir_path(config):
    """
    data of current experiment instance will be saved in config.experiment_instance_dir_path
    """
    from _utils_import import DLUtils
    import datetime
    experiment_dir_path = get_experiment_dir_path(config)
    experiment_instance_dir_path = experiment_dir_path + config.sample_id + "/"\
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    DLUtils.file.EnsureDir(experiment_instance_dir_path)
    config.experiment_instance_dir_path = experiment_instance_dir_path
    return experiment_instance_dir_path

def get_experiment_instance_dir_path(config):
    if not hasattr(config, "experiment_instance_dir_path"):
        create_experiment_instance_dir_path(config)
    return config.experiment_instance_dir_path

def get_save_file_path(config, episode_index):
    experiment_instance_dir_path = get_experiment_instance_dir_path(config)
    return experiment_instance_dir_path + f"model-episode-{episode_index}.dat"