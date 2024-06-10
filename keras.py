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

ef generate_horovodized_code(root_node, filename):

    directory = os.path.join('PARALLEL', 'KERAS')
    filename = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    python_code = au.unparse(root_node)

    file = open(filename, 'w')
    file.write(python_code)
    file.close()
    try:
        import black
        out = check_output(["black", filename], stderr=subprocess.STDOUT)
    except ModuleNotFoundError:
        print("Could not format code! Run pip install black")
    return filename

def add_import(root_node, code):
    name = code.names[0].name
    last_idx = 0
    exists = False
    hvd = [code]
    body = get_body_nodes(root_node)[0]
    for idx, elem in enumerate(body):
        if classname(elem).startswith('import'):
            last_idx = idx
            if elem.names[0].name == name:
                if VERBOSE:
                    print(name + ' import already exists')
                exists = True
        else:
            if not exists:
                last_idx = add_code(body, idx, hvd)
            break

    return last_idx

def add_horovod_initialization(framework, root_node, idx):

    body = get_body_nodes(root_node)[0]
    if framework == 'tf1':
        return add_code(body, idx, copy(hvdconfig.configs_tf1))
    else:
        return add_code(body, idx, copy(hvdconfig.configs_tf2))

def gen_aux_fn(root_node, idx, code):
    body = get_body_nodes(root_node)[0]
    hvd = [code]
    return add_code(body, idx, hvd)
def find_wrappers(root_node):

    def find_wrappers_recursive(body, possible_names):
        for elem in body:
            try:
                if classname(elem) == 'classdef':
                    inheritances = elem.bases
                    for inh in inheritances:
                        if inh.id in possible_names:
                            possible_names.append(elem.name)
                            if VERBOSE:
                                print(f"The class {elem.name} inherits from {inh.id}!")
                else:
                    for body2 in get_body_nodes(elem):
                        find_model_names_recursive(body2, model_names)
            except AttributeError:
                pass

    possible_names = copy(hvdconfig.possible_model_names_keras)
    body = get_body_nodes(root_node)[0]
    find_wrappers_recursive(body, possible_names)
    return possible_names

def find_model_names(root_node, possible_names):

    def find_model_names_recursive(body, model_names):
        # possible_names = copy(hvdconfig.possible_model_names_keras)
        for elem in body:
            try:
                if classname(elem) == 'assign':
                    try:
                        if elem.value.func.id in possible_names:
                            model_names.append(elem.targets[0].id)
                    except AttributeError:
                        if elem.value.func.attr in possible_names:
                            model_names.append(elem.targets[0].id)
                else:
                    for body2 in get_body_nodes(elem):
                        find_model_names_recursive(body2, model_names)
            except AttributeError:
                pass

    model_names = list()
    body = get_body_nodes(root_node)[0]
    find_model_names_recursive(body, model_names)
    return model_names

def find_names_in_code(code):
    return [x for id, x in enumerate(code.split('\'')) if id%2 == 1]

def find_variables_in_code(var_names, body, idx1):
    elems = []
    idxs = []
    for idx, el in enumerate(body):
        if idx > idx1:
            names = find_names_in_code(au.dump(el))
            for n in var_names:
                if n in names:
                    elems.append(el)
                    names = []
                    idxs.append(idx)
    for i in reversed(idxs):
        del body[i]
    return elems

def adapt_model_compile(root_node, model_names):

    def adapt_model_compile_recursive(body, adapted, model_names):

        def adapt_keywords_model_compile(keywords, args):
            found = False
            for i, kw in enumerate(keywords):
                if kw.arg == 'optimizer':
                    found = True
                    opt_keyword = copy(hvdconfig.optimizer_keyword)
                    opt_keyword.value.args = [kw.value]
                    keywords[i] = opt_keyword
                    break
            if not found:
                opt_arg = copy(hvdconfig.optimizer_arg)
                opt_arg.args = [args[0]]
                args[0] = opt_arg

        found = False
        for elem in body:
            try:
                if classname(elem) in ['expr', 'return'] and elem.value.func.attr == 'compile' \
                    and (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    adapt_keywords_model_compile(elem.value.keywords, elem.value.args)
                    adapted = True
                else:
                    for body2 in get_body_nodes(elem):
                        adapted, found2 = adapt_model_compile_recursive(body2, adapted, model_names)
                        found = found or found2
            except AttributeError:
                pass

        return adapted, found

    adapted, found = adapt_model_compile_recursive(get_body_nodes(root_node)[0], False, model_names)
    if not adapted and VERBOSE:
        print("Could not adapt model.compile() function!")
        if not found:
            print("    --Function not found--")

def adapt_model_evaluate_and_predict(root_node, model_names):

    def adapt_model_evaluate_and_predict_recursive(body, adapted, model_names):
        found = False
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) in ['expr', 'assign', 'return'] and \
                    elem.value.func.attr in ['evaluate', 'predict'] and \
                        (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    node = copy(hvdconfig.if_rank_0)
                    elems = []
                    if classname(elem) == 'assign':
                        var_names = []
                        targets = elem.targets
                        if classname(targets[0]) == 'tuple':
                            for target in targets[0].elts:
                                var_names.append(target.id)
                        elif classname(targets) == 'list':
                            var_names.append(targets[0].id)
                        elems = find_variables_in_code(var_names, body, idx1)
                    node.body = [elem] + elems
                    body[idx1] = node
                    adapted = True
                else:
                    for body2 in get_body_nodes(elem):
                        adapted, found2 = adapt_model_evaluate_and_predict_recursive(body2, adapted, model_names)
                        found = found or found2
            except AttributeError:
                pass
        return adapted, found

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_model_evaluate_and_predict_recursive(body, False, model_names)
    if not adapted and VERBOSE:
        print("Could not adapt model.evaluate()/model.predict() function!")
        if not found:
            print("    --Function not found--")

def adapt_model_fit(root_node, model_names):

    def adapt_model_fit_recursive(body, adapted, model_names):

        def adapt_keywords_model_fit(keywords):

            adapted_callbacks = adapted_epochs = adapted_verbose = False

            for idx, kw in enumerate(keywords):
                if kw.arg == 'epochs' and not adapted_epochs:
                    epochs_keyword = copy(hvdconfig.epochs_keyword)
                    epochs_keyword.value.args = [kw.value]
                    keywords[idx] = epochs_keyword
                    adapted_epochs = True
                if kw.arg == 'callbacks' and not adapted_callbacks:
                    # cb_keyword.value.args[0].id = kw.value
                    cb_keyword = copy(hvdconfig.callbacks_keyword)
                    cb_keyword.value.args[0] = kw.value
                    keywords[idx] = cb_keyword
                    adapted_callbacks = True
                if kw.arg == 'verbose' and not adapted_verbose:
                    keywords[idx] = copy(hvdconfig.verbose_keyword)
                    adapted_verbose = True

            if not adapted_callbacks:
                keywords.append(copy(hvdconfig.callbacks_keyword))
                adapted_callbacks = True
            if not adapted_verbose:
                keywords.append(copy(hvdconfig.verbose_keyword))
                adapted_verbose = True

            return adapted_epochs and adapted_verbose and adapted_callbacks

        found = False
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) in ['expr', 'assign', 'return'] and elem.value.func.attr == 'fit' \
                    and (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    adapted = adapt_keywords_model_fit(elem.value.keywords)
                elif classname(elem) == 'call' and elem.func.attr == 'fit' and \
                    (elem.func.value.id in model_names or 'model' in elem.func.value.id):
                    found = True
                    adapted = adapt_keywords_model_fit(elem.keywords)
                else:
                    try:
                        for body2 in get_body_nodes(elem):
                            adapted, found2 = adapt_model_fit_recursive(body2, adapted, model_names)
                            found = found or found2
                    except RootNodeException:
                        pass
            except AttributeError:
                pass
        return adapted, found

    def adapt_model_fit_generator_recursive(body, adapted, model_names):

        def adapt_keywords_model_fit_generator(keywords):

            adapted_steps_per_epoch = adapted_validation_steps = False
            adapted_callbacks = adapted_verbose = adapted_epochs = False

            for idx, kw in enumerate(keywords):
                if kw.arg == 'epochs' and not adapted_epochs:
                    epochs_keyword = copy(hvdconfig.epochs_keyword)
                    epochs_keyword.value.args = [kw.value]
                    keywords[idx] = epochs_keyword
                    adapted_epochs = True
                if kw.arg == 'steps_per_epoch' and not adapted_steps_per_epoch:
                    spe_keyword = copy(hvdconfig.steps_per_epoch_keyword)
                    spe_keyword.value.args = [kw.value]
                    keywords[idx] = spe_keyword
                    adapted_steps_per_epoch = True
                if kw.arg == 'validation_steps' and not adapted_validation_steps:
                    vs_keyword = copy(hvdconfig.validation_steps_keyword)
                    vs_keyword.value.args = [kw.value]
                    keywords[idx] = vs_keyword
                    adapted_validation_steps = True
                if kw.arg == 'callbacks' and not adapted_callbacks:
                    # cb_keyword.value.args[0].id = kw.value
                    cb_keyword = copy(hvdconfig.callbacks_keyword)
                    cb_keyword.value.args[0] = kw.value
                    keywords[idx] = cb_keyword
                    adapted_callbacks = True
                if kw.arg == 'verbose' and not adapted_verbose:
                    keywords[idx] = copy(hvdconfig.verbose_keyword)
                    adapted_verbose = True

            if not adapted_callbacks:
                keywords.append(copy(hvdconfig.callbacks_keyword))
                adapted_callbacks = True
            if not adapted_verbose:
                keywords.append(copy(hvdconfig.verbose_keyword))
                adapted_verbose = True

            adapted = adapted_callbacks and adapted_verbose and \
                        adapted_validation_steps and (adapted_steps_per_epoch or adapted_epochs)

            return adapted

        found = False
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) in ['expr', 'assign', 'return'] and elem.value.func.attr == 'fit_generator' \
                    and (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    adapted = adapt_keywords_model_fit_generator(elem.value.keywords)
                elif classname(elem) == 'call' and elem.func.attr == 'fit_generator' and elem.func.value.id == model_name:
                    found = True
                    adapted = adapt_keywords_model_fit_generator(elem.keywords)
                else:
                    for body2 in get_body_nodes(elem):
                        adapted, found2 = adapt_model_fit_generator_recursive(body2, adapted, model_names)
                        found = found or found2
            except AttributeError:
                pass
        return adapted, found

    def check(name, adapted, found):
        if not adapted:
            if VERBOSE:
                print("Could not adapt " + name + " function!")
                if not found:
                    print("    --Function not found--")

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_model_fit_recursive(body, False, model_names)
    check('model.fit()', adapted, found)
    adapted, found = adapt_model_fit_generator_recursive(body, False, model_names)
    check('model.fit_generator()', adapted, found)
def adapt_model_save(root_node, model_names):

    def adapt_model_save_recursive(body, adapted, model_names):
        found = False
        for idx1, elem in enumerate(body):
            try:
                # print(au.dump(elem))
                if classname(elem) == 'expr' and \
                    elem.value.func.attr in ['save', 'save_weights'] and \
                        (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    node = copy(hvdconfig.if_rank_0)
                    node.body = [elem]
                    body[idx1] = node
                    adapted = True
                else:
                    for body2 in get_body_nodes(elem):
                        adapted, found2 = adapt_model_save_recursive(body2, adapted, model_names)
                        found = found or found2
            except AttributeError:
                pass
        return adapted, found

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_model_save_recursive(body, False, model_names)
    if not adapted and VERBOSE:
        print("Could not adapt model.save() function!")
        if not found:
            print("    --Function not found--")

def remove_block_comments(root_node):

    def remove_block_comments_recursive(body):
        sub_index_list = []
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) == 'expr' and classname(elem.value) == 'str':
                    sub_index_list.append(idx1)
                elif not classname(elem) == 'assign':
                    for body2 in get_body_nodes(elem):
                        remove_block_comments_recursive(body2)
            except AttributeError:
                pass
        for i in reversed(sub_index_list):
            del body[i]

    remove_block_comments_recursive(get_body_nodes(root_node)[0])

def add_auxiliar_functions(root_node, index):
    body = getattr(root_node, 'body')
    for aux_func in hvdconfig.aux_funcs:
        body.insert(index, aux_func)

def add_necessary_imports(node):
    last_idx = 0
    for imp in hvdconfig.imports:
        idx = add_import(node, imp)
        if idx > last_idx:
            last_idx = idx
    return last_idx

def horovodize(framework, path, v):
    time_ini = time.time()
    global VERBOSE
    VERBOSE = v
    # filename of horovod code
    name = f"hvd_{ntpath.basename(path)}"

    # Obtain AST from original code
    node = parse_code(filename=path, save_to_file=True)

    # Find wrappers: "class CustomModel(Sequential)"
    possible_names = find_wrappers(node)

    # Find model names first
    model_names = find_model_names(node, possible_names)
    if len(model_names) == 0:
        print("No models found! Aborting...")
        return

    # Remove block comments or docstrings
    remove_block_comments(node)
    print("✓ Removed unnecesary comments")

    # Add necessary imports
    last_idx = add_necessary_imports(node)
    print("✓ Added imports")

    # Add horovod config for tf1 or tf2
    add_horovod_initialization(framework, node, last_idx)
    print("✓ Added Horovod init")

    # Add auxiliar functions used by horovod
    add_auxiliar_functions(node, last_idx)
    print("✓ Added auxiliar functions")

    # Adapt existing code
    adapt_model_compile(node, model_names)
    adapt_model_fit(node, model_names)
    adapt_model_save(node, model_names)
    adapt_model_evaluate_and_predict(node, model_names)
    print("✓ Code completely adapted")

    # Generate new python code
    filename = generate_horovodized_code(root_node=node, filename=name)
    time_end = time.time()
    print(f"✓ File generated: {filename}")
    if VERBOSE or True:
        print(f"Compilation time: {round(time_end-time_ini, 2)} seconds")
