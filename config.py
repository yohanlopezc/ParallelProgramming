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
