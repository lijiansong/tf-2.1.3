?2
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?"serve*1.14.02unknown8?'

NoOpNoOp
i
ConstConst"/device:CPU:0*%
valueB B


signatures
 *
dtype0*
_output_shapes
: 
R
serving_default_xPlaceholder*
shape: *
dtype0*
_output_shapes
: 
R
serving_default_yPlaceholder*
shape: *
dtype0*
_output_shapes
: 
?
PartitionedCallPartitionedCallserving_default_xserving_default_y*
Tout
2*)
_gradient_op_typePartitionedCall-20*
_output_shapes
: *
Tin
2*)
f$R"
 __inference_signature_wrapper_15**
config_proto

GPU 

CPU2J 8
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*$
fR
__inference__traced_save_37**
config_proto

GPU 

CPU2J 8*
Tout
2*)
_gradient_op_typePartitionedCall-38*
_output_shapes
: *
Tin
2
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tout
2*)
_gradient_op_typePartitionedCall-49*
_output_shapes
: *
Tin
2*'
f"R 
__inference__traced_restore_48**
config_proto

GPU 

CPU2J 8?
?
P
__inference__traced_restore_48
file_prefix

identity_1??	RestoreV2?
RestoreV2/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2*
_output_shapes
:2
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
_output_shapes
: *
T02

Identityd

Identity_1IdentityIdentity:output:0
^RestoreV2*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: 2
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix
?
>
 __inference_signature_wrapper_15
x
y
identity?
PartitionedCallPartitionedCallxy*$
fR
__inference_some_function_7**
config_proto

GPU 

CPU2J 8*
Tout
2*)
_gradient_op_typePartitionedCall-12*
Tin
2*
_output_shapes
: 2
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: : :! 

_user_specified_namex:!

_user_specified_namey
?
9
__inference_some_function_7
x
y
identity:
addAddV2xy*
T0*
_output_shapes
: 2
addJ
IdentityIdentityadd:z:0*
_output_shapes
: *
T02

Identity"
identityIdentity:output:0*
_input_shapes
: : :! 

_user_specified_namex:!

_user_specified_namey
?
q
__inference__traced_save_37
file_prefix
savev2_const

identity_1??MergeV2Checkpoints?SaveV2?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_e4377c37804f4f9f85394481272204bd/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
N2

StringJoinZ

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
dtypes
2*
_output_shapes
 2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
T0*
N*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix^SaveV2"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identityv

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2*
_output_shapes
: *
T02

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*?
serving_default?

y
serving_default_y:0 

x
serving_default_x:0 #
output_0
PartitionedCall:0 tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:?
A

signatures
some_function"
_generic_user_object
,
serving_default"
signature_map
?2?
__inference_some_function_7?
???
FullArgSpec
args?
jself
jx
jy
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
? 
? 
*B(
 __inference_signature_wrapper_15xyO
__inference_some_function_70%?"
?

?
x 

?
y 
? "? y
 __inference_signature_wrapper_15U/?,
? 
%?"

x
?
x 

y
?
y ""?

output_0?
output_0 