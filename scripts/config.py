#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# config.py
#

from enum import Enum


class DebugLevel(Enum):
    INFO = 1
    DETAILS = 2


class Config():
    IS_DEBUG = True
    DEBUG_LEVEL = DebugLevel.DETAILS
