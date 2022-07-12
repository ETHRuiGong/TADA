import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet
from data.cityscapes_loader_annotationadaptation import cityscapesLoaderAnnotateAdapt
from data.synthia_dataset_annotationadaptation import SynthiaDataSetAnnotateAdapt
from data.cityscapes_loader_annotationadaptation_fewshot import cityscapesLoader_annotationadaptation_fewshot
from data.cityscapes_loader_annotationadaptation_fewshot_numsample import cityscapesLoader_annotationadaptation_fewshot_numsample
from data.cityscapes_loader_annotationadaptation_fewshot_numsample_adjustnumsamples import cityscapesLoader_annotationadaptation_fewshot_numsample_adjustnumsamples
from data.cityscapes_loader_annotationadaptation_fewshot_numsample_classspecificsample import cityscapesLoader_annotationadaptation_fewshot_numsample_classspecific
from data.gta5_dataset_annotationcoarsetofine import GTA5DataSet_Coarsetofine
from data.cityscapes_loader_coarsetofine_fewshot_numsample_classspecificsample import cityscapesLoader_coarsetofine_fewshot_numsample_classspecific
from data.synscapes_dataset_conflictclass import SynscapesDataSet_Conflictclass
from data.cityscapes_loader_conflictclass_fewshot_numsample_classspecificsample import cityscapesLoader_conflictclass_fewshot_numsample_classspecific
from data.cityscapes_loader_16classes import cityscapesLoader_16classes

from data.cityscapes_loader_conflictclass_fewshot_numsample_classspecificsample_full import cityscapesLoader_conflictclass_fewshot_numsample_classspecific_full
from data.cityscapes_loader_coarsetofine_fewshot_numsample_classspecificsample_full import cityscapesLoader_coarsetofine_fewshot_numsample_classspecific_full
from data.cityscapes_loader_annotationadaptation_fewshot_numsample_classspecificsample_full import cityscapesLoader_annotationadaptation_fewshot_numsample_classspecific_full

from data.cityscapes_loader_coarsetofine_fewshot_numsample_classspecificsample_extremecase import cityscapesLoader_coarsetofine_fewshot_numsample_classspecific_extremecase
from data.gta5_dataset_annotationcoarsetofine_extremecase import GTA5DataSet_Coarsetofine_Extremecase

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

def get_loader_annotationadaptation(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoaderAnnotateAdapt,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return './data/CityScapes/'
    if name == 'cityscapes_fewshot':
        return './data/CityScapes/'
    if name == 'gta' or name == 'gtaUniform':
        return './data/gta/'
    if name == 'synthia':
        return './data/RAND_CITYSCAPES'
    if name == 'synscapes':
        return './data/Synscapes'

def get_loader_annotationadaptation_fewshot(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_annotationadaptation_fewshot,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]

def get_loader_annotationadaptation_fewshot_numsample(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_annotationadaptation_fewshot_numsample,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]

def get_loader_annotationadaptation_fewshot_numsample_adjustnumsample(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_annotationadaptation_fewshot_numsample_adjustnumsamples,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]


def get_loader_annotationadaptation_fewshot_numsample_classspecific(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_annotationadaptation_fewshot_numsample_classspecific,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]


def get_loader_coarsetofine_fewshot_numsample_classspecific(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_coarsetofine_fewshot_numsample_classspecific,
        "gta": GTA5DataSet_Coarsetofine,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]


def get_loader_conflictclass_fewshot_numsample_classspecific(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_conflictclass_fewshot_numsample_classspecific,
        "synscapes": SynscapesDataSet_Conflictclass,
    }[name]


def get_loader_16classes(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes16classes": cityscapesLoader_16classes,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

def get_loader_annotationadaptation_fewshot_numsample_classspecific_full(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_annotationadaptation_fewshot_numsample_classspecific_full,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]

def get_loader_coarsetofine_fewshot_numsample_classspecific_full(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_coarsetofine_fewshot_numsample_classspecific_full,
        "gta": GTA5DataSet_Coarsetofine,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]

def get_loader_conflictclass_fewshot_numsample_classspecific_full(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_conflictclass_fewshot_numsample_classspecific_full,
        "synscapes": SynscapesDataSet_Conflictclass,
    }[name]


def get_loader_coarsetofine_fewshot_numsample_classspecific_extremecase(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        'cityscapes_fewshot': cityscapesLoader_coarsetofine_fewshot_numsample_classspecific_extremecase,
        "gta": GTA5DataSet_Coarsetofine_Extremecase,
        "synthia": SynthiaDataSetAnnotateAdapt
    }[name]