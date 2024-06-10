from ast import *
import astunparse as au
import subprocess
import inspect
from subprocess import check_output
import os
import sys
from config import Keras as config
import time
import ntpath
from copy import deepcopy as copy

class RootNodeException(Exception):

    pass

def classname(cls):
    return cls.__class__.__name__.lower()

def get_body_nodes(root_node):
    ret_list = []
    try:
        body_node = getattr(root_node, 'body')
        ret_list.append(body_node)
        orelse_node = getattr(root_node, 'orelse')
        ret_list.append(orelse_node)
        finally_node = getattr(root_node, 'finalbody')
        ret_list.append(finally_node)
    except Exception:
        pass
    return ret_list
