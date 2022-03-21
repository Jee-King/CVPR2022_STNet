# -*- coding: utf-8 -*
from videoanalyst.utils import Registry

TRACK_TRANSFOR = Registry('TRACK_TRANSFOR')
VOS_TRANSFOR = Registry('VOS_TRANSFOR')

TASK_TRANSFOR = dict(
    track=TRACK_TRANSFOR,
    vos=VOS_TRANSFOR,
)
