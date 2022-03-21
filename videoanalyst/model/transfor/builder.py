# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from videoanalyst.utils import merge_cfg_into_hps

from .transfor_base import TASK_TRANSFOR


def build(task: str, cfg: CfgNode, transmodel=None):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    basemodel:
        warp backbone into encoder if not None

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_TRANSFOR:
        modules = TASK_TRANSFOR[task]
    else:
        logger.error("no transfor-module for task {}".format(task))
        exit(-1)

    name = cfg.name
    assert name in modules, "transfor model {} not registered for {}!".format(
        name, task)

    if transmodel:
        module = modules[name](transmodel)
    else:
        # module = modules[name](192, 6, 1, 2)
        module = modules[name](pretrain_img_size=280)

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)   # get pre-trained model
    module.set_hps(hps)
    module.update_params()
    return module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]5
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}
    for cfg_name, module in TASK_TRANSFOR.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            transfor = module[name]
            hps = transfor.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
