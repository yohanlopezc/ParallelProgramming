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

def add_code(body, index, extra_code):
    for elem in extra_code:
        body.insert(index, elem)
        index += 1
    return index

def parse_code(filename, save_to_file):
    original_code = open(filename, 'r').read()
    root_node = parse(original_code)
    if save_to_file:
        filename = os.path.join('PARSED', f'parsed_{ntpath.basename(filename)}')
        file = open(filename, 'w')
        file.write(au.dump(root_node))
        file.close()
    return root_node

def recover_code_from_ast_file(filename, save_to_file):
    parsed_code = open(filename, 'r').read()
    global node
    code_to_exec = "node = " + parsed_code
    exec(code_to_exec, globals())
    return generate_code_from_ast(node, save_to_file)
