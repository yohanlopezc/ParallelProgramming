from ast import *

class Keras:

    ## Keras model names
    possible_model_names_keras=[
        'Model', 'Sequential', 'Xception',
        'VGG16', 'VGG19', 'ResNet50',
        'ResNet50V2', 'ResNet101', 'ResNet101V2',
        'ResNet152', 'ResNet152V2', 'InceptionV3',
        'InceptionResNetV2', 'MobileNet', 'DenseNet121',
        'DenseNet169', 'DenseNet201', 'NASNetLarge',
        'NASNetMobile', 'MobileNetV2'
    ]
mports = [

        Import(names=[alias(name='tensorflow',asname='tf')]),
        Import(names=[alias(name='horovod.tensorflow',asname='hvd')]),
        Import(names=[alias(name='horovod.tensorflow.keras',asname='hvd_keras')]),
        Import(names=[alias(name='math',asname=None)]),
        ImportFrom(module='tensorflow.keras.callbacks',names=[alias(name='ModelCheckpoint',asname=None),alias(name='ReduceLROnPlateau',asname=None)],level=0),
        ImportFrom(module='tensorflow.keras',names=[alias(name='backend',asname='K')],level=0),
        ImportFrom(module='tensorflow.keras.optimizers',names=[alias(name='get',asname='get_optimizer_by_name')],level=0)
    ]
configs_tf1 = [
        Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='init',ctx=Load()),args=[],keywords=[])),
        Assign(targets=[Name(id='config',ctx=Store())],value=Call(func=Attribute(value=Name(id='tf',ctx=Load()),attr='ConfigProto',ctx=Load()),args=[],keywords=[])),
        Assign(targets=[Attribute(value=Attribute(value=Name(id='config',ctx=Load()),attr='gpu_options',ctx=Load()),attr='allow_growth',ctx=Store())],value=NameConstant(value=True)),
        Assign(targets=[Attribute(value=Attribute(value=Name(id='config',ctx=Load()),attr='gpu_options',ctx=Load()),attr='visible_device_list',ctx=Store())],value=Call(func=Name(id='str',ctx=Load()),args=[Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='local_rank',ctx=Load()),args=[],keywords=[])],keywords=[])),
        Expr(value=Call(func=Attribute(value=Name(id='K',ctx=Load()),attr='set_session',ctx=Load()),args=[Call(func=Attribute(value=Name(id='tf',ctx=Load()),attr='Session',ctx=Load()),args=[],keywords=[keyword(arg='config',value=Name(id='config',ctx=Load()))])],keywords=[]))
    ]
configs_tf2 = [
        Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='init',ctx=Load()),args=[],keywords=[])),
        Assign(targets=[Name(id='gpus',ctx=Store())],value=Call(func=Attribute(value=Attribute(value=Attribute(value=Name(id='tf',ctx=Load()),attr='config',ctx=Load()),attr='experimental',ctx=Load()),attr='list_physical_devices',ctx=Load()),args=[Str(s='GPU')],keywords=[])),
        For(target=Name(id='gpu',ctx=Store()),iter=Name(id='gpus',ctx=Load()),body=[Expr(value=Call(func=Attribute(value=Attribute(value=Attribute(value=Name(id='tf',ctx=Load()),attr='config',ctx=Load()),attr='experimental',ctx=Load()),attr='set_memory_growth',ctx=Load()),args=[Name(id='gpu',ctx=Load()),NameConstant(value=True)],keywords=[]))],orelse=[]),
        If(test=Name(id='gpus',ctx=Load()),body=[Expr(value=Call(func=Attribute(value=Attribute(value=Attribute(value=Name(id='tf',ctx=Load()),attr='config',ctx=Load()),attr='experimental',ctx=Load()),attr='set_visible_devices',ctx=Load()),args=[Subscript(value=Name(id='gpus',ctx=Load()),slice=Index(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='local_rank',ctx=Load()),args=[],keywords=[])),ctx=Load()),Str(s='GPU')],keywords=[]))],orelse=[])
    ]

optimizer_arg=Call(func=Name(id='hvd_adapt_optimizer',ctx=Load()),args=[],keywords=[])
    optimizer_keyword=keyword(arg='optimizer',value=Call(func=Name(id='hvd_adapt_optimizer',ctx=Load()),args=[Name(id='opt',ctx=Load())],keywords=[]))
    callbacks_keyword=keyword(arg='callbacks',value=Call(func=Name(id='hvd_adapt_callbacks',ctx=Load()),args=[List(elts=[],ctx=Load()),NameConstant(value=True)],keywords=[]))
    epochs_keyword=keyword(arg='epochs',value=Call(func=Name(id='hvd_adapt_epochs',ctx=Load()),args=[Name(id='epochs',ctx=Load())],keywords=[]))
    verbose_keyword=keyword(arg='verbose',value=IfExp(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=Num(n=1),orelse=Num(n=0)))
    steps_per_epoch_keyword=keyword(arg='steps_per_epoch',value=Call(func=Name(id='hvd_adapt_steps',ctx=Load()),args=[],keywords=[]))
    validation_steps_keyword=keyword(arg='validation_steps',value=Call(func=Name(id='hvd_adapt_steps',ctx=Load()),args=[],keywords=[]))

if_rank_0=If(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=[],orelse=[])

    ## Función auxiliar
    aux_funcs = [
        FunctionDef(name='hvd_adapt_optimizer',args=arguments(args=[arg(arg='opt',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[If(test=Call(func=Name(id='isinstance',ctx=Load()),args=[Name(id='opt',ctx=Load()),Name(id='str',ctx=Load())],keywords=[]),body=[Assign(targets=[Name(id='opt',ctx=Store())],value=Call(func=Name(id='get_optimizer_by_name',ctx=Load()),args=[Name(id='opt',ctx=Load())],keywords=[]))],orelse=[]),Assign(targets=[Name(id='opt_config',ctx=Store())],value=Call(func=Attribute(value=Name(id='opt',ctx=Load()),attr='get_config',ctx=Load()),args=[],keywords=[])),Try(body=[AugAssign(target=Subscript(value=Name(id='opt_config',ctx=Load()),slice=Index(value=Str(s='learning_rate')),ctx=Store()),op=Mult(),value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='size',ctx=Load()),args=[],keywords=[]))],handlers=[ExceptHandler(type=Name(id='KeyError',ctx=Load()),name=None,body=[AugAssign(target=Subscript(value=Name(id='opt_config',ctx=Load()),slice=Index(value=Str(s='lr')),ctx=Store()),op=Mult(),value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='size',ctx=Load()),args=[],keywords=[]))])],orelse=[],finalbody=[]),Return(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='DistributedOptimizer',ctx=Load()),args=[Call(func=Attribute(value=Name(id='opt',ctx=Load()),attr='from_config',ctx=Load()),args=[Name(id='opt_config',ctx=Load())],keywords=[])],keywords=[]))],decorator_list=[],returns=None),
        FunctionDef(name='hvd_adapt_callbacks',args=arguments(args=[arg(arg='callbacks',annotation=None),arg(arg='save_checkpoints',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[Assign(targets=[Name(id='hvd_callbacks',ctx=Store())],value=List(elts=[Call(func=Attribute(value=Attribute(value=Name(id='hvd_keras',ctx=Load()),attr='callbacks',ctx=Load()),attr='BroadcastGlobalVariablesCallback',ctx=Load()),args=[Num(n=0)],keywords=[]),Call(func=Attribute(value=Attribute(value=Name(id='hvd_keras',ctx=Load()),attr='callbacks',ctx=Load()),attr='MetricAverageCallback',ctx=Load()),args=[],keywords=[])],ctx=Load())),If(test=BoolOp(op=And(),values=[Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),Name(id='save_checkpoints',ctx=Load())]),body=[Expr(value=Call(func=Attribute(value=Name(id='callbacks',ctx=Load()),attr='append',ctx=Load()),args=[Call(func=Name(id='ModelCheckpoint',ctx=Load()),args=[Str(s='./checkpoint-{epoch}.h5')],keywords=[])],keywords=[]))],orelse=[]),Return(value=BinOp(left=Name(id='hvd_callbacks',ctx=Load()),op=Add(),right=Name(id='callbacks',ctx=Load())))],decorator_list=[],returns=None),
        FunctionDef(name='hvd_adapt_epochs',args=arguments(args=[arg(arg='epochs',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[Return(value=Call(func=Name(id='max',ctx=Load()),args=[Num(n=1),Call(func=Attribute(value=Name(id='math',ctx=Load()),attr='ceil',ctx=Load()),args=[BinOp(left=Name(id='epochs',ctx=Load()),op=FloorDiv(),right=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='size',ctx=Load()),args=[],keywords=[]))],keywords=[])],keywords=[]))],decorator_list=[],returns=None),
        FunctionDef(name='hvd_adapt_steps',args=arguments(args=[arg(arg='steps',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[Return(value=Call(func=Name(id='max',ctx=Load()),args=[Num(n=1),Call(func=Attribute(value=Name(id='math',ctx=Load()),attr='ceil',ctx=Load()),args=[BinOp(left=Name(id='steps',ctx=Load()),op=FloorDiv(),right=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='size',ctx=Load()),args=[],keywords=[]))],keywords=[])],keywords=[]))],decorator_list=[],returns=None)
    ]
class Torch:

## Torchvision model names:
    possible_model_names_torchvision = [
    'resnet18',
    'alexnet',
    'vgg16',
    'squeezenet1_0',
    'densenet161',
    'inception_v3',
    'googlenet',
    'shufflenet_v2_x1_0',
    'mobilenet_v2',
    'resnext50_32x4d',
    'wide_resnet50_2',
    'mnasnet1_0'
    ]

## Torchvision optimizer names:
    possible_optim_names_torchvision = [
    	'Adadelta',
    	'Adagrad',
    	'Adam',
    	'Adamax',
    	'AdamW',
    	'ASGD',
    	'LBFGS',
    	'RMSprop',
    	'Rprop',
    	'SGD',
    	'SparseAdam'
    ]

## IMPORTS:
    imports=[
        Import(names=[alias(name='horovod.torch',asname='hvd')]),
        Import(names=[alias(name='torch',asname=None)])
    ]

## HOROVOD CONFIGS
    configs = [
        Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='init',ctx=Load()),args=[],keywords=[])),
        If(test=Call(func=Attribute(value=Attribute(value=Name(id='torch',ctx=Load()),attr='cuda',ctx=Load()),attr='is_available',ctx=Load()),args=[],keywords=[]),body=[Expr(value=Call(func=Attribute(value=Attribute(value=Name(id='torch',ctx=Load()),attr='cuda',ctx=Load()),attr='set_device',ctx=Load()),args=[Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='local_rank',ctx=Load()),args=[],keywords=[])],keywords=[]))],orelse=[])
    ]

##Adaptaciones del código

    # adapt data loaders:
    data_sampler=Assign( targets=[Name(id='hvd_sampler_', ctx=Store())], value=Call( func=Attribute( value=Attribute( value=Attribute( value=Attribute( value=Name( id='torch', ctx=Load()), attr='utils', ctx=Load()), attr='data', ctx=Load()), attr='distributed', ctx=Load()), attr='DistributedSampler', ctx=Load()), args=[], keywords=[ keyword( arg='dataset', value=Name( id='dataset', ctx=Load())), keyword( arg='num_replicas', value=Call( func=Attribute( value=Name( id='hvd', ctx=Load()), attr='size', ctx=Load()), args=[], keywords=[])), keyword( arg='rank', value=Call( func=Attribute( value=Name( id='hvd', ctx=Load()), attr='rank', ctx=Load()), args=[], keywords=[]))]))
    data_sampler_keyword=keyword(arg='sampler',value=Name(id='hvd_sampler_',ctx=Load()))


    model_to_cuda=Expr(value=Call(func=Attribute(value=Name(id='model',ctx=Load()),attr='cuda',ctx=Load()),args=[],keywords=[]))
    # adapt optimizer
    adapt_opt=Expr(value=Call(func=Name(id='hvd_adapt_optimizer',ctx=Load()),args=[Name(id='optimizer',ctx=Load()), Name(id='model',ctx=Load())],keywords=[]))
    # broadcast_parameters
    broadcast_parameters=Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='broadcast_parameters',ctx=Load()),args=[Call(func=Attribute(value=Name(id='model', ctx=Load()),attr='state_dict',ctx=Load()),args=[],keywords=[])],keywords=[keyword(arg='root_rank',value=Num(n=0))]))
    # adapt loss & accuracy
    adapt_loss=[
        AugAssign( target=Name( id='test_loss', ctx=Store()), op=Div(), value=Call( func=Name( id='len', ctx=Load()), args=[Name( id='test_sampler', ctx=Load())], keywords=[])),
        Assign( targets=[Name( id='test_loss', ctx=Store())], value=Call( func=Name( id='hvd_metric_average', ctx=Load()), args=[ Name( id='test_loss', ctx=Load()), Str(s='avg_loss')], keywords=[]))
    ]
    adapt_accuracy=[
        AugAssign( target=Name( id='test_accuracy', ctx=Store()), op=Div(), value=Call( func=Name( id='len', ctx=Load()), args=[Name( id='test_sampler', ctx=Load())], keywords=[])),
        Assign( targets=[Name( id='test_accuracy', ctx=Store())], value=Call( func=Name( id='hvd_metric_average', ctx=Load()), args=[ Name( id='test_accuracy', ctx=Load()), Str(s='avg_accuracy')], keywords=[]))
    ]

    if_rank_0=If(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=[],orelse=[])
    ## auxiliar functions
    aux_funcs = [
        FunctionDef(name='hvd_adapt_optimizer',args=arguments(args=[arg(arg='optimizer',annotation=None),arg(arg='model',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[AugAssign(target=Subscript(value=Subscript(value=Attribute(value=Name(id='optimizer',ctx=Load()),attr='param_groups',ctx=Load()),slice=Index(value=Num(n=0)),ctx=Load()),slice=Index(value=Str(s='lr')),ctx=Store()),op=Mult(),value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='size',ctx=Load()),args=[],keywords=[])),Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='broadcast_optimizer_state',ctx=Load()),args=[Name(id='optimizer',ctx=Load())],keywords=[keyword(arg='root_rank',value=Num(n=0))])),Assign(targets=[Name(id='optimizer',ctx=Store())],value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='DistributedOptimizer',ctx=Load()),args=[Name(id='optimizer',ctx=Load())],keywords=[keyword(arg='named_parameters',value=Call(func=Attribute(value=Name(id='model',ctx=Load()),attr='named_parameters',ctx=Load()),args=[],keywords=[]))])),Return(value=Name(id='optimizer',ctx=Load()))],decorator_list=[],returns=None),
        FunctionDef(name='hvd_metric_average',args=arguments(args=[arg(arg='val',annotation=None),arg(arg='name',annotation=None)],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[Assign(targets=[Name(id='tensor',ctx=Store())],value=Call(func=Attribute(value=Name(id='torch',ctx=Load()),attr='tensor',ctx=Load()),args=[Name(id='val',ctx=Load())],keywords=[])),Assign(targets=[Name(id='avg_tensor',ctx=Store())],value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='allreduce',ctx=Load()),args=[Name(id='tensor',ctx=Load())],keywords=[keyword(arg='name',value=Name(id='name',ctx=Load()))])),Return(value=Call(func=Attribute(value=Name(id='avg_tensor',ctx=Load()),attr='item',ctx=Load()),args=[],keywords=[]))],decorator_list=[],returns=None)
    ]

