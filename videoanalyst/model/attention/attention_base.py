# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

TRACK_ATTENTION = Registry('TRACK_ATTENTION')
VOS_ATTENTION = Registry('VOS_ATTENTION')

TASK_ATTENTION = dict(
    track=TRACK_ATTENTION,
    vos=VOS_ATTENTION,
)
