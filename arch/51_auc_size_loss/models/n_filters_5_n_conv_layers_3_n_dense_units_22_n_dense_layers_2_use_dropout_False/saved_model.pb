��'
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
:
OnesLike
x"T
y"T"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��$
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/dense_output/kernelVarHandleOp*
_output_shapes
: *+

debug_nameAdam/v/dense_output/kernel/*
dtype0*
shape
:*+
shared_nameAdam/v/dense_output/kernel
�
.Adam/v/dense_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_output/kernelVarHandleOp*
_output_shapes
: *+

debug_nameAdam/m/dense_output/kernel/*
dtype0*
shape
:*+
shared_nameAdam/m/dense_output/kernel
�
.Adam/m/dense_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_output/kernel*
_output_shapes

:*
dtype0
�
)Adam/v/dense1/batch_normalization_28/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/v/dense1/batch_normalization_28/beta/*
dtype0*
shape:*:
shared_name+)Adam/v/dense1/batch_normalization_28/beta
�
=Adam/v/dense1/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOp)Adam/v/dense1/batch_normalization_28/beta*
_output_shapes
:*
dtype0
�
)Adam/m/dense1/batch_normalization_28/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/m/dense1/batch_normalization_28/beta/*
dtype0*
shape:*:
shared_name+)Adam/m/dense1/batch_normalization_28/beta
�
=Adam/m/dense1/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOp)Adam/m/dense1/batch_normalization_28/beta*
_output_shapes
:*
dtype0
�
*Adam/v/dense1/batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/v/dense1/batch_normalization_28/gamma/*
dtype0*
shape:*;
shared_name,*Adam/v/dense1/batch_normalization_28/gamma
�
>Adam/v/dense1/batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOp*Adam/v/dense1/batch_normalization_28/gamma*
_output_shapes
:*
dtype0
�
*Adam/m/dense1/batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/m/dense1/batch_normalization_28/gamma/*
dtype0*
shape:*;
shared_name,*Adam/m/dense1/batch_normalization_28/gamma
�
>Adam/m/dense1/batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOp*Adam/m/dense1/batch_normalization_28/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense1/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/dense1/bias/*
dtype0*
shape:*#
shared_nameAdam/v/dense1/bias
u
&Adam/v/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense1/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense1/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/dense1/bias/*
dtype0*
shape:*#
shared_nameAdam/m/dense1/bias
u
&Adam/m/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense1/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense1/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense1/kernel/*
dtype0*
shape
:*%
shared_nameAdam/v/dense1/kernel
}
(Adam/v/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense1/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense1/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense1/kernel/*
dtype0*
shape
:*%
shared_nameAdam/m/dense1/kernel
}
(Adam/m/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense1/kernel*
_output_shapes

:*
dtype0
�
)Adam/v/dense0/batch_normalization_27/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/v/dense0/batch_normalization_27/beta/*
dtype0*
shape:*:
shared_name+)Adam/v/dense0/batch_normalization_27/beta
�
=Adam/v/dense0/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOp)Adam/v/dense0/batch_normalization_27/beta*
_output_shapes
:*
dtype0
�
)Adam/m/dense0/batch_normalization_27/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/m/dense0/batch_normalization_27/beta/*
dtype0*
shape:*:
shared_name+)Adam/m/dense0/batch_normalization_27/beta
�
=Adam/m/dense0/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOp)Adam/m/dense0/batch_normalization_27/beta*
_output_shapes
:*
dtype0
�
*Adam/v/dense0/batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/v/dense0/batch_normalization_27/gamma/*
dtype0*
shape:*;
shared_name,*Adam/v/dense0/batch_normalization_27/gamma
�
>Adam/v/dense0/batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOp*Adam/v/dense0/batch_normalization_27/gamma*
_output_shapes
:*
dtype0
�
*Adam/m/dense0/batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/m/dense0/batch_normalization_27/gamma/*
dtype0*
shape:*;
shared_name,*Adam/m/dense0/batch_normalization_27/gamma
�
>Adam/m/dense0/batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOp*Adam/m/dense0/batch_normalization_27/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense0/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/dense0/bias/*
dtype0*
shape:*#
shared_nameAdam/v/dense0/bias
u
&Adam/v/dense0/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense0/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense0/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/dense0/bias/*
dtype0*
shape:*#
shared_nameAdam/m/dense0/bias
u
&Adam/m/dense0/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense0/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense0/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense0/kernel/*
dtype0*
shape
:
*%
shared_nameAdam/v/dense0/kernel
}
(Adam/v/dense0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense0/kernel*
_output_shapes

:
*
dtype0
�
Adam/m/dense0/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense0/kernel/*
dtype0*
shape
:
*%
shared_nameAdam/m/dense0/kernel
}
(Adam/m/dense0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense0/kernel*
_output_shapes

:
*
dtype0
�
Adam/v/conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/conv2/kernel/*
dtype0*
shape:*$
shared_nameAdam/v/conv2/kernel
�
'Adam/v/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/conv2/kernel/*
dtype0*
shape:*$
shared_nameAdam/m/conv2/kernel
�
'Adam/m/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/conv1/kernel/*
dtype0*
shape:*$
shared_nameAdam/v/conv1/kernel
�
'Adam/v/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/conv1/kernel/*
dtype0*
shape:*$
shared_nameAdam/m/conv1/kernel
�
'Adam/m/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/conv0/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/conv0/kernel/*
dtype0*
shape:*$
shared_nameAdam/v/conv0/kernel
�
'Adam/v/conv0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv0/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/conv0/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/conv0/kernel/*
dtype0*
shape:*$
shared_nameAdam/m/conv0/kernel
�
'Adam/m/conv0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv0/kernel*&
_output_shapes
:*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
-dense1/batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *>

debug_name0.dense1/batch_normalization_28/moving_variance/*
dtype0*
shape:*>
shared_name/-dense1/batch_normalization_28/moving_variance
�
Adense1/batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp-dense1/batch_normalization_28/moving_variance*
_output_shapes
:*
dtype0
�
)dense1/batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *:

debug_name,*dense1/batch_normalization_28/moving_mean/*
dtype0*
shape:*:
shared_name+)dense1/batch_normalization_28/moving_mean
�
=dense1/batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp)dense1/batch_normalization_28/moving_mean*
_output_shapes
:*
dtype0
�
"dense1/batch_normalization_28/betaVarHandleOp*
_output_shapes
: *3

debug_name%#dense1/batch_normalization_28/beta/*
dtype0*
shape:*3
shared_name$"dense1/batch_normalization_28/beta
�
6dense1/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOp"dense1/batch_normalization_28/beta*
_output_shapes
:*
dtype0
�
#dense1/batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$dense1/batch_normalization_28/gamma/*
dtype0*
shape:*4
shared_name%#dense1/batch_normalization_28/gamma
�
7dense1/batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOp#dense1/batch_normalization_28/gamma*
_output_shapes
:*
dtype0
�
-dense0/batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *>

debug_name0.dense0/batch_normalization_27/moving_variance/*
dtype0*
shape:*>
shared_name/-dense0/batch_normalization_27/moving_variance
�
Adense0/batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp-dense0/batch_normalization_27/moving_variance*
_output_shapes
:*
dtype0
�
)dense0/batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *:

debug_name,*dense0/batch_normalization_27/moving_mean/*
dtype0*
shape:*:
shared_name+)dense0/batch_normalization_27/moving_mean
�
=dense0/batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp)dense0/batch_normalization_27/moving_mean*
_output_shapes
:*
dtype0
�
"dense0/batch_normalization_27/betaVarHandleOp*
_output_shapes
: *3

debug_name%#dense0/batch_normalization_27/beta/*
dtype0*
shape:*3
shared_name$"dense0/batch_normalization_27/beta
�
6dense0/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOp"dense0/batch_normalization_27/beta*
_output_shapes
:*
dtype0
�
#dense0/batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$dense0/batch_normalization_27/gamma/*
dtype0*
shape:*4
shared_name%#dense0/batch_normalization_27/gamma
�
7dense0/batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOp#dense0/batch_normalization_27/gamma*
_output_shapes
:*
dtype0
�
dense_output/kernelVarHandleOp*
_output_shapes
: *$

debug_namedense_output/kernel/*
dtype0*
shape
:*$
shared_namedense_output/kernel
{
'dense_output/kernel/Read/ReadVariableOpReadVariableOpdense_output/kernel*
_output_shapes

:*
dtype0
�
dense1/iterationVarHandleOp*
_output_shapes
: *!

debug_namedense1/iteration/*
dtype0	*
shape: *!
shared_namedense1/iteration
m
$dense1/iteration/Read/ReadVariableOpReadVariableOpdense1/iteration*
_output_shapes
: *
dtype0	
�
dense1/biasVarHandleOp*
_output_shapes
: *

debug_namedense1/bias/*
dtype0*
shape:*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:*
dtype0
�
dense1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense1/kernel/*
dtype0*
shape
:*
shared_namedense1/kernel
o
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes

:*
dtype0
�
dense0/iterationVarHandleOp*
_output_shapes
: *!

debug_namedense0/iteration/*
dtype0	*
shape: *!
shared_namedense0/iteration
m
$dense0/iteration/Read/ReadVariableOpReadVariableOpdense0/iteration*
_output_shapes
: *
dtype0	
�
dense0/biasVarHandleOp*
_output_shapes
: *

debug_namedense0/bias/*
dtype0*
shape:*
shared_namedense0/bias
g
dense0/bias/Read/ReadVariableOpReadVariableOpdense0/bias*
_output_shapes
:*
dtype0
�
dense0/kernelVarHandleOp*
_output_shapes
: *

debug_namedense0/kernel/*
dtype0*
shape
:
*
shared_namedense0/kernel
o
!dense0/kernel/Read/ReadVariableOpReadVariableOpdense0/kernel*
_output_shapes

:
*
dtype0
�
conv2/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv2/kernel/*
dtype0*
shape:*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:*
dtype0
�
conv1/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv1/kernel/*
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
�
conv0/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv0/kernel/*
dtype0*
shape:*
shared_nameconv0/kernel
u
 conv0/kernel/Read/ReadVariableOpReadVariableOpconv0/kernel*&
_output_shapes
:*
dtype0
z
serving_default_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv0/kernelconv1/kernelconv2/kerneldense0/kerneldense0/bias-dense0/batch_normalization_27/moving_variance#dense0/batch_normalization_27/gamma)dense0/batch_normalization_27/moving_mean"dense0/batch_normalization_27/betadense0/iterationdense1/kerneldense1/bias-dense1/batch_normalization_28/moving_variance#dense1/batch_normalization_28/gamma)dense1/batch_normalization_28/moving_mean"dense1/batch_normalization_28/betadense1/iterationdense_output/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2100417

NoOpNoOp
ӂ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$kernel_quantizer
$kernel_quantizer_internal
%
quantizers

&kernel
 '_jit_compiled_convolution_op*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.	quantizer* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5kernel_quantizer
5kernel_quantizer_internal
6
quantizers

7kernel
 8_jit_compiled_convolution_op*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?	quantizer* 
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Fkernel_quantizer
Fkernel_quantizer_internal
G
quantizers

Hkernel
 I_jit_compiled_convolution_op*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P	quantizer* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]kernel_quantizer
^bias_quantizer
]kernel_quantizer_internal
^bias_quantizer_internal
_
quantizers
`	batchnorm

akernel
bbias
c
_iteration*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j	quantizer* 
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qkernel_quantizer
rbias_quantizer
qkernel_quantizer_internal
rbias_quantizer_internal
s
quantizers
t	batchnorm

ukernel
vbias
w
_iteration*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~	quantizer* 
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�kernel_quantizer_internal
�
quantizers
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�	quantizer* 
�
&0
71
H2
a3
b4
�5
�6
c7
�8
�9
u10
v11
�12
�13
w14
�15
�16
�17*
_
&0
71
H2
a3
b4
�5
�6
u7
v8
�9
�10
�11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

&0*

&0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
	
$0* 
\V
VARIABLE_VALUEconv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

70*

70*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
	
50* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

H0*

H0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
	
F0* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
9
a0
b1
�2
�3
c4
�5
�6*
"
a0
b1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

]0
^1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
]W
VARIABLE_VALUEdense0/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense0/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEdense0/iteration:layer_with_weights-3/_iteration/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
9
u0
v1
�2
�3
w4
�5
�6*
"
u0
v1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

q0
r1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEdense1/iteration:layer_with_weights-4/_iteration/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 


�0* 
c]
VARIABLE_VALUEdense_output/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
c]
VARIABLE_VALUE#dense0/batch_normalization_27/gamma&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"dense0/batch_normalization_27/beta&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)dense0/batch_normalization_27/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-dense0/batch_normalization_27/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#dense1/batch_normalization_28/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"dense1/batch_normalization_28/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)dense1/batch_normalization_28/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-dense1/batch_normalization_28/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
2
c0
�1
�2
w3
�4
�5*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

�0*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

c0
�1
�2*

`0*
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 

w0
�1
�2*

t0*
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
^X
VARIABLE_VALUEAdam/m/conv0/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv0/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv1/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense0/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense0/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense0/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense0/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/dense0/batch_normalization_27/gamma2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/dense0/batch_normalization_27/gamma2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/dense0/batch_normalization_27/beta2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/dense0/batch_normalization_27/beta2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense1/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense1/kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense1/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense1/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/dense1/batch_normalization_28/gamma2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/dense1/batch_normalization_28/gamma2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/m/dense1/batch_normalization_28/beta2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/dense1/batch_normalization_28/beta2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/dense_output/kernel2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/dense_output/kernel2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv0/kernelconv1/kernelconv2/kerneldense0/kerneldense0/biasdense0/iterationdense1/kerneldense1/biasdense1/iterationdense_output/kernel#dense0/batch_normalization_27/gamma"dense0/batch_normalization_27/beta)dense0/batch_normalization_27/moving_mean-dense0/batch_normalization_27/moving_variance#dense1/batch_normalization_28/gamma"dense1/batch_normalization_28/beta)dense1/batch_normalization_28/moving_mean-dense1/batch_normalization_28/moving_variance	iterationlearning_rateAdam/m/conv0/kernelAdam/v/conv0/kernelAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/dense0/kernelAdam/v/dense0/kernelAdam/m/dense0/biasAdam/v/dense0/bias*Adam/m/dense0/batch_normalization_27/gamma*Adam/v/dense0/batch_normalization_27/gamma)Adam/m/dense0/batch_normalization_27/beta)Adam/v/dense0/batch_normalization_27/betaAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/bias*Adam/m/dense1/batch_normalization_28/gamma*Adam/v/dense1/batch_normalization_28/gamma)Adam/m/dense1/batch_normalization_28/beta)Adam/v/dense1/batch_normalization_28/betaAdam/m/dense_output/kernelAdam/v/dense_output/kerneltotalcountConst*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2102023
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv0/kernelconv1/kernelconv2/kerneldense0/kerneldense0/biasdense0/iterationdense1/kerneldense1/biasdense1/iterationdense_output/kernel#dense0/batch_normalization_27/gamma"dense0/batch_normalization_27/beta)dense0/batch_normalization_27/moving_mean-dense0/batch_normalization_27/moving_variance#dense1/batch_normalization_28/gamma"dense1/batch_normalization_28/beta)dense1/batch_normalization_28/moving_mean-dense1/batch_normalization_28/moving_variance	iterationlearning_rateAdam/m/conv0/kernelAdam/v/conv0/kernelAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/dense0/kernelAdam/v/dense0/kernelAdam/m/dense0/biasAdam/v/dense0/bias*Adam/m/dense0/batch_normalization_27/gamma*Adam/v/dense0/batch_normalization_27/gamma)Adam/m/dense0/batch_normalization_27/beta)Adam/v/dense0/batch_normalization_27/betaAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/bias*Adam/m/dense1/batch_normalization_28/gamma*Adam/v/dense1/batch_normalization_28/gamma)Adam/m/dense1/batch_normalization_28/beta)Adam/v/dense1/batch_normalization_28/betaAdam/m/dense_output/kernelAdam/v/dense_output/kerneltotalcount*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2102170��"
�
�
C__inference_dense0_layer_call_and_return_conditional_losses_2099482

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:L
>batch_normalization_27_assignmovingavg_readvariableop_resource:N
@batch_normalization_27_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_27_batchnorm_mul_readvariableop_resource:F
8batch_normalization_27_batchnorm_readvariableop_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add_2/ReadVariableOp�&batch_normalization_27/AssignMovingAvg�5batch_normalization_27/AssignMovingAvg/ReadVariableOp�(batch_normalization_27/AssignMovingAvg_1�7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_27/batchnorm/ReadVariableOp�3batch_normalization_27/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_27/moments/meanMeanBiasAdd:output:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_27/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:05batch_normalization_27/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_27/AssignMovingAvgAssignSubVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_27/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:07batch_normalization_27/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_27/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource0batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/subSub7batch_normalization_27/batchnorm/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R{
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource)^batch_normalization_27/AssignMovingAvg_1*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0c
subSubReadVariableOp_1:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:E
mul_2Mul	mul_1:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:
Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:
@
NegNegtruediv:z:0*
T0*
_output_shapes

:
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:
K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:
P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:
[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:
6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:
g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:
S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:
Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:
M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:
L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:
R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:
[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:
L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_13AddV2moments/Squeeze_1:output:0add_13/y:output:0*
T0*
_output_shapes
:A
Rsqrt_2Rsqrt
add_13:z:0*
T0*
_output_shapes
:�
ReadVariableOp_2ReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource)^batch_normalization_27/AssignMovingAvg_1*
_output_shapes
:*
dtype0M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_14AddV2ReadVariableOp_2:value:0add_14/y:output:0*
T0*
_output_shapes
:=
SqrtSqrt
add_14:z:0*
T0*
_output_shapes
:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_15AddV2moments/Squeeze_1:output:0add_15/y:output:0*
T0*
_output_shapes
:A
Rsqrt_3Rsqrt
add_15:z:0*
T0*
_output_shapes
:I
mul_14MulSqrt:y:0Rsqrt_3:y:0*
T0*
_output_shapes
:_
Mul_15MulMatMul_1:product:0
mul_14:z:0*
T0*'
_output_shapes
:���������^
	BiasAdd_1BiasAdd
Mul_15:z:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp'^batch_normalization_27/AssignMovingAvg6^batch_normalization_27/AssignMovingAvg/ReadVariableOp)^batch_normalization_27/AssignMovingAvg_18^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp4^batch_normalization_27/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_27/AssignMovingAvg_1(batch_normalization_27/AssignMovingAvg_12P
&batch_normalization_27/AssignMovingAvg&batch_normalization_27/AssignMovingAvg2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
E
)__inference_reshape_layer_call_fn_2100422

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2099051h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_relu3_layer_call_and_return_conditional_losses_2101106

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�l
�
C__inference_dense0_layer_call_and_return_conditional_losses_2099987

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:F
8batch_normalization_27_batchnorm_readvariableop_resource:J
<batch_normalization_27_batchnorm_mul_readvariableop_resource:H
:batch_normalization_27_batchnorm_readvariableop_1_resource:H
:batch_normalization_27_batchnorm_readvariableop_2_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�add_2/ReadVariableOp�/batch_normalization_27/batchnorm/ReadVariableOp�1batch_normalization_27/batchnorm/ReadVariableOp_1�1batch_normalization_27/batchnorm/ReadVariableOp_2�3batch_normalization_27/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOp�sub/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_27/batchnorm/addAddV27batch_normalization_27/batchnorm/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_27/batchnorm/mul_2Mul9batch_normalization_27/batchnorm/ReadVariableOp_1:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/subSub9batch_normalization_27/batchnorm/ReadVariableOp_2:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sub/ReadVariableOpReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0e
subSubReadVariableOp_1:value:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:C
mul_2Mulmul:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:
Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:
@
NegNegtruediv:z:0*
T0*
_output_shapes

:
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:
K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:
P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:
[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:
6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:
g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:
S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:
Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:
M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:
L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:
R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:
[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:
L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������f
	BiasAdd_1BiasAddMatMul_1:product:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^add_2/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp2^batch_normalization_27/batchnorm/ReadVariableOp_12^batch_normalization_27/batchnorm/ReadVariableOp_24^batch_normalization_27/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp^sub/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2f
1batch_normalization_27/batchnorm/ReadVariableOp_11batch_normalization_27/batchnorm/ReadVariableOp_12f
1batch_normalization_27/batchnorm/ReadVariableOp_21batch_normalization_27/batchnorm/ReadVariableOp_22b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_28_layer_call_fn_2101671

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2099009o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101667:'#
!
_user_specified_name	2101665:'#
!
_user_specified_name	2101663:'#
!
_user_specified_name	2101661:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101645

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv1_layer_call_and_return_conditional_losses_2099185

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������	: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
C
'__inference_relu3_layer_call_fn_2101059

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu3_layer_call_and_return_conditional_losses_2099544`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�l
�
C__inference_dense1_layer_call_and_return_conditional_losses_2100125

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:H
:batch_normalization_28_batchnorm_readvariableop_1_resource:H
:batch_normalization_28_batchnorm_readvariableop_2_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�add_2/ReadVariableOp�/batch_normalization_28/batchnorm/ReadVariableOp�1batch_normalization_28/batchnorm/ReadVariableOp_1�1batch_normalization_28/batchnorm/ReadVariableOp_2�3batch_normalization_28/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOp�sub/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_28/batchnorm/addAddV27batch_normalization_28/batchnorm/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_28/batchnorm/mul_2Mul9batch_normalization_28/batchnorm/ReadVariableOp_1:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/subSub9batch_normalization_28/batchnorm/ReadVariableOp_2:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sub/ReadVariableOpReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0e
subSubReadVariableOp_1:value:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:C
mul_2Mulmul:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������f
	BiasAdd_1BiasAddMatMul_1:product:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^add_2/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp2^batch_normalization_28/batchnorm/ReadVariableOp_12^batch_normalization_28/batchnorm/ReadVariableOp_24^batch_normalization_28/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp^sub/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2f
1batch_normalization_28/batchnorm/ReadVariableOp_11batch_normalization_28/batchnorm/ReadVariableOp_12f
1batch_normalization_28/batchnorm/ReadVariableOp_21batch_normalization_28/batchnorm/ReadVariableOp_22b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_dense0_layer_call_fn_2100785

inputs
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense0_layer_call_and_return_conditional_losses_2099987o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100781:'#
!
_user_specified_name	2100779:'#
!
_user_specified_name	2100777:'#
!
_user_specified_name	2100775:'#
!
_user_specified_name	2100773:'#
!
_user_specified_name	2100771:'#
!
_user_specified_name	2100769:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�,
 __inference__traced_save_2102023
file_prefix=
#read_disablecopyonread_conv0_kernel:?
%read_1_disablecopyonread_conv1_kernel:?
%read_2_disablecopyonread_conv2_kernel:8
&read_3_disablecopyonread_dense0_kernel:
2
$read_4_disablecopyonread_dense0_bias:3
)read_5_disablecopyonread_dense0_iteration:	 8
&read_6_disablecopyonread_dense1_kernel:2
$read_7_disablecopyonread_dense1_bias:3
)read_8_disablecopyonread_dense1_iteration:	 >
,read_9_disablecopyonread_dense_output_kernel:K
=read_10_disablecopyonread_dense0_batch_normalization_27_gamma:J
<read_11_disablecopyonread_dense0_batch_normalization_27_beta:Q
Cread_12_disablecopyonread_dense0_batch_normalization_27_moving_mean:U
Gread_13_disablecopyonread_dense0_batch_normalization_27_moving_variance:K
=read_14_disablecopyonread_dense1_batch_normalization_28_gamma:J
<read_15_disablecopyonread_dense1_batch_normalization_28_beta:Q
Cread_16_disablecopyonread_dense1_batch_normalization_28_moving_mean:U
Gread_17_disablecopyonread_dense1_batch_normalization_28_moving_variance:-
#read_18_disablecopyonread_iteration:	 1
'read_19_disablecopyonread_learning_rate: G
-read_20_disablecopyonread_adam_m_conv0_kernel:G
-read_21_disablecopyonread_adam_v_conv0_kernel:G
-read_22_disablecopyonread_adam_m_conv1_kernel:G
-read_23_disablecopyonread_adam_v_conv1_kernel:G
-read_24_disablecopyonread_adam_m_conv2_kernel:G
-read_25_disablecopyonread_adam_v_conv2_kernel:@
.read_26_disablecopyonread_adam_m_dense0_kernel:
@
.read_27_disablecopyonread_adam_v_dense0_kernel:
:
,read_28_disablecopyonread_adam_m_dense0_bias::
,read_29_disablecopyonread_adam_v_dense0_bias:R
Dread_30_disablecopyonread_adam_m_dense0_batch_normalization_27_gamma:R
Dread_31_disablecopyonread_adam_v_dense0_batch_normalization_27_gamma:Q
Cread_32_disablecopyonread_adam_m_dense0_batch_normalization_27_beta:Q
Cread_33_disablecopyonread_adam_v_dense0_batch_normalization_27_beta:@
.read_34_disablecopyonread_adam_m_dense1_kernel:@
.read_35_disablecopyonread_adam_v_dense1_kernel::
,read_36_disablecopyonread_adam_m_dense1_bias::
,read_37_disablecopyonread_adam_v_dense1_bias:R
Dread_38_disablecopyonread_adam_m_dense1_batch_normalization_28_gamma:R
Dread_39_disablecopyonread_adam_v_dense1_batch_normalization_28_gamma:Q
Cread_40_disablecopyonread_adam_m_dense1_batch_normalization_28_beta:Q
Cread_41_disablecopyonread_adam_v_dense1_batch_normalization_28_beta:F
4read_42_disablecopyonread_adam_m_dense_output_kernel:F
4read_43_disablecopyonread_adam_v_dense_output_kernel:)
read_44_disablecopyonread_total: )
read_45_disablecopyonread_count: 
savev2_const
identity_93��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_conv0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_conv0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_conv1_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*&
_output_shapes
:y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_conv2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense0_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense0_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:
x
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_dense0_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_dense0_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_5/DisableCopyOnReadDisableCopyOnRead)read_5_disablecopyonread_dense0_iteration"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp)read_5_disablecopyonread_dense0_iteration^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_dense1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_dense1_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_dense1_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead,read_9_disablecopyonread_dense_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp,read_9_disablecopyonread_dense_output_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_10/DisableCopyOnReadDisableCopyOnRead=read_10_disablecopyonread_dense0_batch_normalization_27_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp=read_10_disablecopyonread_dense0_batch_normalization_27_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_dense0_batch_normalization_27_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_dense0_batch_normalization_27_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnReadCread_12_disablecopyonread_dense0_batch_normalization_27_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpCread_12_disablecopyonread_dense0_batch_normalization_27_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnReadGread_13_disablecopyonread_dense0_batch_normalization_27_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpGread_13_disablecopyonread_dense0_batch_normalization_27_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead=read_14_disablecopyonread_dense1_batch_normalization_28_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp=read_14_disablecopyonread_dense1_batch_normalization_28_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead<read_15_disablecopyonread_dense1_batch_normalization_28_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp<read_15_disablecopyonread_dense1_batch_normalization_28_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnReadCread_16_disablecopyonread_dense1_batch_normalization_28_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpCread_16_disablecopyonread_dense1_batch_normalization_28_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnReadGread_17_disablecopyonread_dense1_batch_normalization_28_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpGread_17_disablecopyonread_dense1_batch_normalization_28_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_iteration^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_learning_rate^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_m_conv0_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_m_conv0_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_v_conv0_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_v_conv0_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_adam_m_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_adam_m_conv1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead-read_23_disablecopyonread_adam_v_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp-read_23_disablecopyonread_adam_v_conv1_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_m_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_m_conv2_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_adam_v_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_adam_v_conv2_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_m_dense0_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_m_dense0_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_v_dense0_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_v_dense0_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_28/DisableCopyOnReadDisableCopyOnRead,read_28_disablecopyonread_adam_m_dense0_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp,read_28_disablecopyonread_adam_m_dense0_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_29/DisableCopyOnReadDisableCopyOnRead,read_29_disablecopyonread_adam_v_dense0_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp,read_29_disablecopyonread_adam_v_dense0_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_30/DisableCopyOnReadDisableCopyOnReadDread_30_disablecopyonread_adam_m_dense0_batch_normalization_27_gamma"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpDread_30_disablecopyonread_adam_m_dense0_batch_normalization_27_gamma^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnReadDread_31_disablecopyonread_adam_v_dense0_batch_normalization_27_gamma"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpDread_31_disablecopyonread_adam_v_dense0_batch_normalization_27_gamma^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_32/DisableCopyOnReadDisableCopyOnReadCread_32_disablecopyonread_adam_m_dense0_batch_normalization_27_beta"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpCread_32_disablecopyonread_adam_m_dense0_batch_normalization_27_beta^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnReadCread_33_disablecopyonread_adam_v_dense0_batch_normalization_27_beta"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpCread_33_disablecopyonread_adam_v_dense0_batch_normalization_27_beta^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_m_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_m_dense1_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_35/DisableCopyOnReadDisableCopyOnRead.read_35_disablecopyonread_adam_v_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp.read_35_disablecopyonread_adam_v_dense1_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_36/DisableCopyOnReadDisableCopyOnRead,read_36_disablecopyonread_adam_m_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp,read_36_disablecopyonread_adam_m_dense1_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead,read_37_disablecopyonread_adam_v_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp,read_37_disablecopyonread_adam_v_dense1_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_38/DisableCopyOnReadDisableCopyOnReadDread_38_disablecopyonread_adam_m_dense1_batch_normalization_28_gamma"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpDread_38_disablecopyonread_adam_m_dense1_batch_normalization_28_gamma^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnReadDread_39_disablecopyonread_adam_v_dense1_batch_normalization_28_gamma"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpDread_39_disablecopyonread_adam_v_dense1_batch_normalization_28_gamma^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_40/DisableCopyOnReadDisableCopyOnReadCread_40_disablecopyonread_adam_m_dense1_batch_normalization_28_beta"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpCread_40_disablecopyonread_adam_m_dense1_batch_normalization_28_beta^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnReadCread_41_disablecopyonread_adam_v_dense1_batch_normalization_28_beta"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpCread_41_disablecopyonread_adam_v_dense1_batch_normalization_28_beta^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnRead4read_42_disablecopyonread_adam_m_dense_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp4read_42_disablecopyonread_adam_m_dense_output_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_43/DisableCopyOnReadDisableCopyOnRead4read_43_disablecopyonread_adam_v_dense_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp4read_43_disablecopyonread_adam_v_dense_output_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:t
Read_44/DisableCopyOnReadDisableCopyOnReadread_44_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpread_44_disablecopyonread_total^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_45/DisableCopyOnReadDisableCopyOnReadread_45_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpread_45_disablecopyonread_count^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/_iteration/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/_iteration/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *=
dtypes3
12/			�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_92Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_93IdentityIdentity_92:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_93Identity_93:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=/9

_output_shapes
: 

_user_specified_nameConst:%.!

_user_specified_namecount:%-!

_user_specified_nametotal::,6
4
_user_specified_nameAdam/v/dense_output/kernel::+6
4
_user_specified_nameAdam/m/dense_output/kernel:I*E
C
_user_specified_name+)Adam/v/dense1/batch_normalization_28/beta:I)E
C
_user_specified_name+)Adam/m/dense1/batch_normalization_28/beta:J(F
D
_user_specified_name,*Adam/v/dense1/batch_normalization_28/gamma:J'F
D
_user_specified_name,*Adam/m/dense1/batch_normalization_28/gamma:2&.
,
_user_specified_nameAdam/v/dense1/bias:2%.
,
_user_specified_nameAdam/m/dense1/bias:4$0
.
_user_specified_nameAdam/v/dense1/kernel:4#0
.
_user_specified_nameAdam/m/dense1/kernel:I"E
C
_user_specified_name+)Adam/v/dense0/batch_normalization_27/beta:I!E
C
_user_specified_name+)Adam/m/dense0/batch_normalization_27/beta:J F
D
_user_specified_name,*Adam/v/dense0/batch_normalization_27/gamma:JF
D
_user_specified_name,*Adam/m/dense0/batch_normalization_27/gamma:2.
,
_user_specified_nameAdam/v/dense0/bias:2.
,
_user_specified_nameAdam/m/dense0/bias:40
.
_user_specified_nameAdam/v/dense0/kernel:40
.
_user_specified_nameAdam/m/dense0/kernel:3/
-
_user_specified_nameAdam/v/conv2/kernel:3/
-
_user_specified_nameAdam/m/conv2/kernel:3/
-
_user_specified_nameAdam/v/conv1/kernel:3/
-
_user_specified_nameAdam/m/conv1/kernel:3/
-
_user_specified_nameAdam/v/conv0/kernel:3/
-
_user_specified_nameAdam/m/conv0/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:MI
G
_user_specified_name/-dense1/batch_normalization_28/moving_variance:IE
C
_user_specified_name+)dense1/batch_normalization_28/moving_mean:B>
<
_user_specified_name$"dense1/batch_normalization_28/beta:C?
=
_user_specified_name%#dense1/batch_normalization_28/gamma:MI
G
_user_specified_name/-dense0/batch_normalization_27/moving_variance:IE
C
_user_specified_name+)dense0/batch_normalization_27/moving_mean:B>
<
_user_specified_name$"dense0/batch_normalization_27/beta:C?
=
_user_specified_name%#dense0/batch_normalization_27/gamma:3
/
-
_user_specified_namedense_output/kernel:0	,
*
_user_specified_namedense1/iteration:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:0,
*
_user_specified_namedense0/iteration:+'
%
_user_specified_namedense0/bias:-)
'
_user_specified_namedense0/kernel:,(
&
_user_specified_nameconv2/kernel:,(
&
_user_specified_nameconv1/kernel:,(
&
_user_specified_nameconv0/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2099009

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu2_layer_call_and_return_conditional_losses_2100736

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������N
ReluReluinputs*
T0*/
_output_shapes
:���������W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101705

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100229	
input!
unknown:#
	unknown_0:#
	unknown_1:
	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:	 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *�
f�R�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2100147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100225:'#
!
_user_specified_name	2100223:'#
!
_user_specified_name	2100221:'#
!
_user_specified_name	2100219:'#
!
_user_specified_name	2100217:'#
!
_user_specified_name	2100215:'#
!
_user_specified_name	2100213:'#
!
_user_specified_name	2100211:'
#
!
_user_specified_name	2100209:'	#
!
_user_specified_name	2100207:'#
!
_user_specified_name	2100205:'#
!
_user_specified_name	2100203:'#
!
_user_specified_name	2100201:'#
!
_user_specified_name	2100199:'#
!
_user_specified_name	2100197:'#
!
_user_specified_name	2100195:'#
!
_user_specified_name	2100193:'#
!
_user_specified_name	2100191:O K
(
_output_shapes
:����������

_user_specified_nameinput
�&
�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2098989

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_output_layer_call_fn_2101518

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_2099846`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�l
�
C__inference_dense0_layer_call_and_return_conditional_losses_2101054

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:F
8batch_normalization_27_batchnorm_readvariableop_resource:J
<batch_normalization_27_batchnorm_mul_readvariableop_resource:H
:batch_normalization_27_batchnorm_readvariableop_1_resource:H
:batch_normalization_27_batchnorm_readvariableop_2_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�add_2/ReadVariableOp�/batch_normalization_27/batchnorm/ReadVariableOp�1batch_normalization_27/batchnorm/ReadVariableOp_1�1batch_normalization_27/batchnorm/ReadVariableOp_2�3batch_normalization_27/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOp�sub/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_27/batchnorm/addAddV27batch_normalization_27/batchnorm/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_27/batchnorm/mul_2Mul9batch_normalization_27/batchnorm/ReadVariableOp_1:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/subSub9batch_normalization_27/batchnorm/ReadVariableOp_2:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sub/ReadVariableOpReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0e
subSubReadVariableOp_1:value:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:C
mul_2Mulmul:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:
Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:
@
NegNegtruediv:z:0*
T0*
_output_shapes

:
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:
K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:
P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:
[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:
6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:
g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:
S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:
Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:
M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:
L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:
R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:
[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:
L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������f
	BiasAdd_1BiasAddMatMul_1:product:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^add_2/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp2^batch_normalization_27/batchnorm/ReadVariableOp_12^batch_normalization_27/batchnorm/ReadVariableOp_24^batch_normalization_27/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp^sub/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2f
1batch_normalization_27/batchnorm/ReadVariableOp_11batch_normalization_27/batchnorm/ReadVariableOp_12f
1batch_normalization_27/batchnorm/ReadVariableOp_21batch_normalization_27/batchnorm/ReadVariableOp_22b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
`
D__inference_reshape_layer_call_and_return_conditional_losses_2100436

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv2_layer_call_and_return_conditional_losses_2099277

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense0_layer_call_and_return_conditional_losses_2100932

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:L
>batch_normalization_27_assignmovingavg_readvariableop_resource:N
@batch_normalization_27_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_27_batchnorm_mul_readvariableop_resource:F
8batch_normalization_27_batchnorm_readvariableop_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add_2/ReadVariableOp�&batch_normalization_27/AssignMovingAvg�5batch_normalization_27/AssignMovingAvg/ReadVariableOp�(batch_normalization_27/AssignMovingAvg_1�7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_27/batchnorm/ReadVariableOp�3batch_normalization_27/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_27/moments/meanMeanBiasAdd:output:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_27/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:05batch_normalization_27/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_27/AssignMovingAvgAssignSubVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_27/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:07batch_normalization_27/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_27/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource0batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_27/batchnorm/subSub7batch_normalization_27/batchnorm/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R{
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource)^batch_normalization_27/AssignMovingAvg_1*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0c
subSubReadVariableOp_1:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:E
mul_2Mul	mul_1:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:
Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:
@
NegNegtruediv:z:0*
T0*
_output_shapes

:
D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:
K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:
P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:
[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:
6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:
g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:
S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:
Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:
@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:
M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:
L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:
R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:
[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:
L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_13AddV2moments/Squeeze_1:output:0add_13/y:output:0*
T0*
_output_shapes
:A
Rsqrt_2Rsqrt
add_13:z:0*
T0*
_output_shapes
:�
ReadVariableOp_2ReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource)^batch_normalization_27/AssignMovingAvg_1*
_output_shapes
:*
dtype0M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_14AddV2ReadVariableOp_2:value:0add_14/y:output:0*
T0*
_output_shapes
:=
SqrtSqrt
add_14:z:0*
T0*
_output_shapes
:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_15AddV2moments/Squeeze_1:output:0add_15/y:output:0*
T0*
_output_shapes
:A
Rsqrt_3Rsqrt
add_15:z:0*
T0*
_output_shapes
:I
mul_14MulSqrt:y:0Rsqrt_3:y:0*
T0*
_output_shapes
:_
Mul_15MulMatMul_1:product:0
mul_14:z:0*
T0*'
_output_shapes
:���������^
	BiasAdd_1BiasAdd
Mul_15:z:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp'^batch_normalization_27/AssignMovingAvg6^batch_normalization_27/AssignMovingAvg/ReadVariableOp)^batch_normalization_27/AssignMovingAvg_18^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp4^batch_normalization_27/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_27/AssignMovingAvg_1(batch_normalization_27/AssignMovingAvg_12P
&batch_normalization_27/AssignMovingAvg&batch_normalization_27/AssignMovingAvg2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
I__inference_dense_output_layer_call_and_return_conditional_losses_2099796

inputs)
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0T
mulMulReadVariableOp:value:0Pow:z:0*
T0*
_output_shapes

:O
truedivRealDivmul:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: ]
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*
_output_shapes

:S
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_2NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101625

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense1_layer_call_and_return_conditional_losses_2099692

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:L
>batch_normalization_28_assignmovingavg_readvariableop_resource:N
@batch_normalization_28_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add_2/ReadVariableOp�&batch_normalization_28/AssignMovingAvg�5batch_normalization_28/AssignMovingAvg/ReadVariableOp�(batch_normalization_28/AssignMovingAvg_1�7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_28/batchnorm/ReadVariableOp�3batch_normalization_28/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_28/moments/meanMeanBiasAdd:output:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_28/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:05batch_normalization_28/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_28/AssignMovingAvgAssignSubVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_28/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:07batch_normalization_28/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_28/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource0batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/subSub7batch_normalization_28/batchnorm/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R{
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource)^batch_normalization_28/AssignMovingAvg_1*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0c
subSubReadVariableOp_1:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:E
mul_2Mul	mul_1:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_13AddV2moments/Squeeze_1:output:0add_13/y:output:0*
T0*
_output_shapes
:A
Rsqrt_2Rsqrt
add_13:z:0*
T0*
_output_shapes
:�
ReadVariableOp_2ReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource)^batch_normalization_28/AssignMovingAvg_1*
_output_shapes
:*
dtype0M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_14AddV2ReadVariableOp_2:value:0add_14/y:output:0*
T0*
_output_shapes
:=
SqrtSqrt
add_14:z:0*
T0*
_output_shapes
:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_15AddV2moments/Squeeze_1:output:0add_15/y:output:0*
T0*
_output_shapes
:A
Rsqrt_3Rsqrt
add_15:z:0*
T0*
_output_shapes
:I
mul_14MulSqrt:y:0Rsqrt_3:y:0*
T0*
_output_shapes
:_
Mul_15MulMatMul_1:product:0
mul_14:z:0*
T0*'
_output_shapes
:���������^
	BiasAdd_1BiasAdd
Mul_15:z:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp'^batch_normalization_28/AssignMovingAvg6^batch_normalization_28/AssignMovingAvg/ReadVariableOp)^batch_normalization_28/AssignMovingAvg_18^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp4^batch_normalization_28/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_28/AssignMovingAvg_1(batch_normalization_28/AssignMovingAvg_12P
&batch_normalization_28/AssignMovingAvg&batch_normalization_28/AssignMovingAvg2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_28_layer_call_fn_2101658

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2098989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101654:'#
!
_user_specified_name	2101652:'#
!
_user_specified_name	2101650:'#
!
_user_specified_name	2101648:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101725

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2100147	
input'
conv0_2099853:'
conv1_2099857:'
conv2_2099861: 
dense0_2099988:

dense0_2099990:
dense0_2099992:
dense0_2099994:
dense0_2099996:
dense0_2099998:
dense0_2100000:	  
dense1_2100126:
dense1_2100128:
dense1_2100130:
dense1_2100132:
dense1_2100134:
dense1_2100136:
dense1_2100138:	 &
dense_output_2100142:
identity��conv0/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�dense0/StatefulPartitionedCall�dense1/StatefulPartitionedCall�$dense_output/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2099051�
conv0/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv0_2099853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_2099093�
relu0/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu0_layer_call_and_return_conditional_losses_2099143�
conv1/StatefulPartitionedCallStatefulPartitionedCallrelu0/PartitionedCall:output:0conv1_2099857*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_2099185�
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_2099235�
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_2099861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_2099277�
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_2099327�
flatten/PartitionedCallPartitionedCallrelu2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2099334�
dense0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense0_2099988dense0_2099990dense0_2099992dense0_2099994dense0_2099996dense0_2099998dense0_2100000*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense0_layer_call_and_return_conditional_losses_2099987�
relu3/PartitionedCallPartitionedCall'dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu3_layer_call_and_return_conditional_losses_2099544�
dense1/StatefulPartitionedCallStatefulPartitionedCallrelu3/PartitionedCall:output:0dense1_2100126dense1_2100128dense1_2100130dense1_2100132dense1_2100134dense1_2100136dense1_2100138*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_2100125�
relu4/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu4_layer_call_and_return_conditional_losses_2099754�
$dense_output/StatefulPartitionedCallStatefulPartitionedCallrelu4/PartitionedCall:output:0dense_output_2100142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_output_layer_call_and_return_conditional_losses_2099796�
output/PartitionedCallPartitionedCall-dense_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_2099846n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense0/StatefulPartitionedCall^dense1/StatefulPartitionedCall%^dense_output/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense0/StatefulPartitionedCalldense0/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall:'#
!
_user_specified_name	2100142:'#
!
_user_specified_name	2100138:'#
!
_user_specified_name	2100136:'#
!
_user_specified_name	2100134:'#
!
_user_specified_name	2100132:'#
!
_user_specified_name	2100130:'#
!
_user_specified_name	2100128:'#
!
_user_specified_name	2100126:'
#
!
_user_specified_name	2100000:'	#
!
_user_specified_name	2099998:'#
!
_user_specified_name	2099996:'#
!
_user_specified_name	2099994:'#
!
_user_specified_name	2099992:'#
!
_user_specified_name	2099990:'#
!
_user_specified_name	2099988:'#
!
_user_specified_name	2099861:'#
!
_user_specified_name	2099857:'#
!
_user_specified_name	2099853:O K
(
_output_shapes
:����������

_user_specified_nameinput
�

�
(__inference_dense0_layer_call_fn_2100766

inputs
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense0_layer_call_and_return_conditional_losses_2099482o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������
: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100762:'#
!
_user_specified_name	2100760:'#
!
_user_specified_name	2100758:'#
!
_user_specified_name	2100756:'#
!
_user_specified_name	2100754:'#
!
_user_specified_name	2100752:'#
!
_user_specified_name	2100750:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
_
C__inference_output_layer_call_and_return_conditional_losses_2101565

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_relu4_layer_call_fn_2101418

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu4_layer_call_and_return_conditional_losses_2099754`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv0_layer_call_and_return_conditional_losses_2100484

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������	Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_relu4_layer_call_and_return_conditional_losses_2101465

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu0_layer_call_and_return_conditional_losses_2100536

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������	N
ReluReluinputs*
T0*/
_output_shapes
:���������	W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������	D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������	z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������	X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������	c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������	Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������	U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������	Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������	_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������	l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������	c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������	P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������	e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������	Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������	^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������	L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������	c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������	t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������	Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100188	
input!
unknown:#
	unknown_0:#
	unknown_1:
	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:	 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *�
f�R�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2099849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100184:'#
!
_user_specified_name	2100182:'#
!
_user_specified_name	2100180:'#
!
_user_specified_name	2100178:'#
!
_user_specified_name	2100176:'#
!
_user_specified_name	2100174:'#
!
_user_specified_name	2100172:'#
!
_user_specified_name	2100170:'
#
!
_user_specified_name	2100168:'	#
!
_user_specified_name	2100166:'#
!
_user_specified_name	2100164:'#
!
_user_specified_name	2100162:'#
!
_user_specified_name	2100160:'#
!
_user_specified_name	2100158:'#
!
_user_specified_name	2100156:'#
!
_user_specified_name	2100154:'#
!
_user_specified_name	2100152:'#
!
_user_specified_name	2100150:O K
(
_output_shapes
:����������

_user_specified_nameinput
�
�
%__inference_signature_wrapper_2100417	
input!
unknown:#
	unknown_0:#
	unknown_1:
	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:	 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	 

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2098875o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100413:'#
!
_user_specified_name	2100411:'#
!
_user_specified_name	2100409:'#
!
_user_specified_name	2100407:'#
!
_user_specified_name	2100405:'#
!
_user_specified_name	2100403:'#
!
_user_specified_name	2100401:'#
!
_user_specified_name	2100399:'
#
!
_user_specified_name	2100397:'	#
!
_user_specified_name	2100395:'#
!
_user_specified_name	2100393:'#
!
_user_specified_name	2100391:'#
!
_user_specified_name	2100389:'#
!
_user_specified_name	2100387:'#
!
_user_specified_name	2100385:'#
!
_user_specified_name	2100383:'#
!
_user_specified_name	2100381:'#
!
_user_specified_name	2100379:O K
(
_output_shapes
:����������

_user_specified_nameinput
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2099334

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu0_layer_call_and_return_conditional_losses_2099143

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������	N
ReluReluinputs*
T0*/
_output_shapes
:���������	W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������	D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������	z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������	X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������	c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������	Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������	U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������	Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������	_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������	l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������	c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������	P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������	e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������	Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������	^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������	L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������	c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������	t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������	Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
`
D__inference_reshape_layer_call_and_return_conditional_losses_2099051

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�8
"__inference__wrapped_model_2098875	
input�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv0_readvariableop_resource:�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv1_readvariableop_resource:�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv2_readvariableop_resource:�
un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_matmul_readvariableop_resource:
�
vn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_biasadd_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_mul_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_1_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_2_resource:}
sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_assignaddvariableop_resource:	 �
un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_matmul_readvariableop_resource:�
vn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_biasadd_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_mul_readvariableop_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_1_resource:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_2_resource:}
sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_assignaddvariableop_resource:	 �
tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense_output_readvariableop_resource:
identity��dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_1�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_2�dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_1�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_2�dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_1�fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_2�jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/AssignAddVariableOp�mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOp�ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOp�en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp�gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_1�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOp��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_1��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_2��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOp�in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOp�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOp�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOp�in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOp�jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/AssignAddVariableOp�mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOp�ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOp�en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp�gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_1�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOp��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_1��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_2��n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOp�in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOp�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOp�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOp�in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOp�kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp�mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_1�mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_2�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/ShapeShapeinput*
T0*
_output_shapes
::���
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_sliceStridedSlicefn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Shape:output:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stack:output:0vn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stack_1:output:0vn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shapePacknn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/strided_slice:output:0pn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/1:output:0pn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/2:output:0pn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/ReshapeReshapeinputnn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1/y:output:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOpReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv0_readvariableop_resource*&
_output_shapes
:*
dtype0�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mulMulln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp:value:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truedivRealDiv]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truediv:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truediv:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Round:y:0*
T0*&
_output_shapes
:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/StopGradient:output:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Neg_1Neg]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow:z:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Neg_1:y:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_2/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_1Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_1/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_2:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/subSub]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow:z:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/sub/y:output:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/clip_by_value/MinimumMinimum_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_1:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/sub:z:0*
T0*&
_output_shapes
:�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/clip_by_value/Minimum:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_1:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_2Mul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow_1:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/clip_by_value:z:0*
T0*&
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_2:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/truediv_1:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_1ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv0_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Neg_2Negnn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_3AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/Neg_2:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_3:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_4Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_4/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_3:z:0*
T0*&
_output_shapes
:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/mul_4:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_2ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv0_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_4AddV2nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_2:value:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/StopGradient_1:output:0*
T0*&
_output_shapes
:�
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/convolutionConv2Dhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/reshape/Reshape:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/add_4:z:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow/y:output:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/CastCast]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_1Cast_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_2Castgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/subSub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_2:y:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_2Powdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Const:output:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_1Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/LessEqual	LessEqualjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/convolution:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_1:z:0*
T0*/
_output_shapes
:���������	�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/ReluRelujn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/convolution:output:0*
T0*/
_output_shapes
:���������	�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/ones_likeOnesLikejn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/convolution:output:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Pow_2:z:0*
T0*
_output_shapes
: �
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mulMulcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/ones_like:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_2:z:0*
T0*/
_output_shapes
:���������	�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/SelectV2SelectV2cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/LessEqual:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Relu:activations:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul:z:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_1Muljn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/convolution:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast:y:0*
T0*/
_output_shapes
:���������	�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truedivRealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_1:y:0*
T0*/
_output_shapes
:���������	�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv:z:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv:z:0*
T0*/
_output_shapes
:���������	�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Round:y:0*
T0*/
_output_shapes
:���������	�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add:z:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/StopGradient:output:0*
T0*/
_output_shapes
:���������	�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_1:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast:y:0*
T0*/
_output_shapes
:���������	�
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_2RealDivjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_2/x:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast:y:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_3Subfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_2:z:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_value/MinimumMinimumcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/truediv_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/sub_3:z:0*
T0*/
_output_shapes
:���������	�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_value/Minimum:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_value/y:output:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_2Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Cast_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/clip_by_value:z:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Neg_1Neggn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/SelectV2:output:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/Neg_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_2:z:0*
T0*/
_output_shapes
:���������	�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_3/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_2:z:0*
T0*/
_output_shapes
:���������	�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/mul_3:z:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_3AddV2gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/SelectV2:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/StopGradient_1:output:0*
T0*/
_output_shapes
:���������	�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1/y:output:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOpReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv1_readvariableop_resource*&
_output_shapes
:*
dtype0�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mulMulln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp:value:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truedivRealDiv]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truediv:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truediv:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Round:y:0*
T0*&
_output_shapes
:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/StopGradient:output:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Neg_1Neg]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow:z:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Neg_1:y:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_2/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_1Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_1/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_2:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/subSub]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow:z:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/sub/y:output:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/clip_by_value/MinimumMinimum_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_1:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/sub:z:0*
T0*&
_output_shapes
:�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/clip_by_value/Minimum:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_1:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_2Mul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow_1:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/clip_by_value:z:0*
T0*&
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_2:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/truediv_1:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_1ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv1_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Neg_2Negnn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_3AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/Neg_2:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_3:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_4Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_4/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_3:z:0*
T0*&
_output_shapes
:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/mul_4:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_2ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv1_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_4AddV2nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_2:value:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/StopGradient_1:output:0*
T0*&
_output_shapes
:�
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/convolutionConv2D_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu0/add_3:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow/y:output:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/CastCast]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_1Cast_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_2Castgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/subSub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_2:y:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_2Powdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Const:output:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_1Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/LessEqual	LessEqualjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/convolution:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_1:z:0*
T0*/
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/ReluRelujn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/convolution:output:0*
T0*/
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/ones_likeOnesLikejn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/convolution:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Pow_2:z:0*
T0*
_output_shapes
: �
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mulMulcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/ones_like:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_2:z:0*
T0*/
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/SelectV2SelectV2cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/LessEqual:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Relu:activations:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_1Muljn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/convolution:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast:y:0*
T0*/
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truedivRealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_1:y:0*
T0*/
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv:z:0*
T0*/
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Round:y:0*
T0*/
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/StopGradient:output:0*
T0*/
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_1:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast:y:0*
T0*/
_output_shapes
:����������
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_2RealDivjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_2/x:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast:y:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_3Subfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_2:z:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_value/MinimumMinimumcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/truediv_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/sub_3:z:0*
T0*/
_output_shapes
:����������
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_value/Minimum:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_2Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Cast_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/clip_by_value:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Neg_1Neggn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/SelectV2:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/Neg_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_2:z:0*
T0*/
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_3/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_2:z:0*
T0*/
_output_shapes
:����������
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/mul_3:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_3AddV2gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/SelectV2:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/StopGradient_1:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1/y:output:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOpReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv2_readvariableop_resource*&
_output_shapes
:*
dtype0�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mulMulln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp:value:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truedivRealDiv]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truediv:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truediv:z:0*
T0*&
_output_shapes
:�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Round:y:0*
T0*&
_output_shapes
:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/StopGradient:output:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Neg_1Neg]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow:z:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Neg_1:y:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_2/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_1Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_1/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_2:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/subSub]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow:z:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/sub/y:output:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/clip_by_value/MinimumMinimum_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_1:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/sub:z:0*
T0*&
_output_shapes
:�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/clip_by_value/Minimum:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_1:z:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_2Mul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow_1:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/clip_by_value:z:0*
T0*&
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_2:z:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Pow:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/truediv_1:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_1ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv2_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Neg_2Negnn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_3AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/Neg_2:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_3:z:0*
T0*&
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_4Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_4/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_3:z:0*
T0*&
_output_shapes
:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/mul_4:z:0*
T0*&
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_2ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_conv2_readvariableop_resource*&
_output_shapes
:*
dtype0�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_4AddV2nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_2:value:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/StopGradient_1:output:0*
T0*&
_output_shapes
:�
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/convolutionConv2D_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu1/add_3:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow/y:output:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/CastCast]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_1Cast_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_2Castgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/subSub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_2:y:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_2Powdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Const:output:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_1Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/LessEqual	LessEqualjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/convolution:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_1:z:0*
T0*/
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/ReluRelujn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/convolution:output:0*
T0*/
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/ones_likeOnesLikejn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/convolution:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Pow_2:z:0*
T0*
_output_shapes
: �
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mulMulcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/ones_like:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_2:z:0*
T0*/
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/SelectV2SelectV2cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/LessEqual:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Relu:activations:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_1Muljn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/convolution:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast:y:0*
T0*/
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truedivRealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_1:y:0*
T0*/
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv:z:0*
T0*/
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Round:y:0*
T0*/
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/StopGradient:output:0*
T0*/
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_1:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast:y:0*
T0*/
_output_shapes
:����������
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_2RealDivjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_2/x:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast:y:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_3Subfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_2:z:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_value/MinimumMinimumcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/truediv_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/sub_3:z:0*
T0*/
_output_shapes
:����������
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_value/Minimum:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_2Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Cast_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/clip_by_value:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Neg_1Neggn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/SelectV2:output:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/Neg_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_2:z:0*
T0*/
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_3/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_2:z:0*
T0*/
_output_shapes
:����������
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/mul_3:z:0*
T0*/
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_3AddV2gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/SelectV2:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/StopGradient_1:output:0*
T0*/
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/flatten/ReshapeReshape_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu2/add_3:z:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/flatten/Const:output:0*
T0*'
_output_shapes
:���������
�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z �
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOpReadVariableOpun_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMulMatMulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/flatten/Reshape:output:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOpReadVariableOpvn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAddBiasAddgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul:product:0un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/addAddV2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp:value:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/RsqrtRsqrtn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mulMul�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/Rsqrt:y:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul_1Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd:output:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul_2Mul�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_1:value:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/subSub�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_2:value:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/add_1AddV2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul_1:z:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/AssignAddVariableOpAssignAddVariableOpsn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_assignaddvariableop_resourceen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Const:output:0*
_output_shapes
 *
dtype0	�
un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/meanMeangn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd:output:0~n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/StopGradientStopGradientln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/mean:output:0*
T0*
_output_shapes

:�
pn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/SquaredDifferenceSquaredDifferencegn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd:output:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/varianceMeantn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/SquaredDifference:z:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/SqueezeSqueezeln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/Squeeze_1Squeezepn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/addAddV2mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp:value:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add/y:output:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/RsqrtRsqrt^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add:z:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_1AddV2qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/moments/Squeeze_1:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_1/y:output:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Rsqrt_1Rsqrt`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_1:z:0*
T0*
_output_shapes
:�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mulMul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Rsqrt:y:0qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1Mulbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Rsqrt_1:y:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_1ReadVariableOpvn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/subSubon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_1:value:0qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOp:value:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_2Mul^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub:z:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_2:z:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOpReadVariableOpun_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense0_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3Mul^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul:z:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/PowPowen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow/x:output:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_4Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow:z:0*
T0*
_output_shapes

:
�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truedivRealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_4:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1:z:0*
T0*
_output_shapes

:
�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/NegNegbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/RoundRoundbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_3AddV2^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Round:y:0*
T0*
_output_shapes

:
�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradientStopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_3:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_4AddV2bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv:z:0ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient:output:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_1Neg^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_5AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_5/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_5Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_5/x:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_5:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_1Sub^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_1/y:output:0*
T0*
_output_shapes
: �
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value/MinimumMinimum`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_4:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_1:z:0*
T0*
_output_shapes

:
�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_valueMaximumpn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value/Minimum:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_5:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_6Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_1:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value:z:0*
T0*
_output_shapes

:
�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_1RealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_6:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow:z:0*
T0*
_output_shapes

:
�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_7Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_7/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_1:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_2Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_6AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_2:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_7:z:0*
T0*
_output_shapes

:
�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_8Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_8/x:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_6:z:0*
T0*
_output_shapes

:
�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_1StopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_8:z:0*
T0*
_output_shapes

:
�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_7AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_1:output:0*
T0*
_output_shapes

:
�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_9Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2:z:0*
T0*
_output_shapes
:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_2RealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_9:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_3Negdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_2:z:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Round_1Rounddn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_2:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_8AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_3:y:0bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Round_1:y:0*
T0*
_output_shapes
:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_2StopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_8:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_9AddV2dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_2:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_2:output:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_4Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_10AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_4:y:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_10/y:output:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_10Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_10/x:output:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_10:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_2/y:output:0*
T0*
_output_shapes
: �
nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value_1/MinimumMinimum`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_9:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub_2:z:0*
T0*
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value_1Maximumrn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value_1/Minimum:z:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_10:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_11Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_3:z:0jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/clip_by_value_1:z:0*
T0*
_output_shapes
:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_3RealDivan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_11:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Pow_2:z:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_12Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_12/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/truediv_3:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_5Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_11AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/Neg_5:y:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_12:z:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_13Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_13/x:output:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_11:z:0*
T0*
_output_shapes
:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_3StopGradientan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_13:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_12AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/StopGradient_3:output:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul_1MatMulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/flatten/Reshape:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_7:z:0*
T0*'
_output_shapes
:����������
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd_1BiasAddin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul_1:product:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_12:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow/y:output:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/CastCast]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_1Cast_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_2Castgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/subSub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_2:y:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_2Powdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Const:output:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_1Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/LessEqual	LessEqualin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd_1:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_1:z:0*
T0*'
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/ReluReluin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd_1:output:0*
T0*'
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/ones_likeOnesLikein_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd_1:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Pow_2:z:0*
T0*
_output_shapes
: �
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mulMulcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/ones_like:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_2:z:0*
T0*'
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/SelectV2SelectV2cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/LessEqual:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Relu:activations:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_1Mulin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd_1:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast:y:0*
T0*'
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truedivRealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_1:y:0*
T0*'
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv:z:0*
T0*'
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Round:y:0*
T0*'
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/StopGradient:output:0*
T0*'
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_1:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast:y:0*
T0*'
_output_shapes
:����������
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_2RealDivjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_2/x:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast:y:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_3Subfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_2:z:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_value/MinimumMinimumcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/truediv_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/sub_3:z:0*
T0*'
_output_shapes
:����������
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_value/Minimum:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_2Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Cast_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/clip_by_value:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Neg_1Neggn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/SelectV2:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/Neg_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_2:z:0*
T0*'
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_3/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_2:z:0*
T0*'
_output_shapes
:����������
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/mul_3:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_3AddV2gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/SelectV2:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z �
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOpReadVariableOpun_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMulMatMul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_3:z:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOpReadVariableOpvn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAddBiasAddgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul:product:0un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/addAddV2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp:value:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/RsqrtRsqrtn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mulMul�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/Rsqrt:y:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul_1Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd:output:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul_2Mul�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_1:value:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
{n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/subSub�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_2:value:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
}n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/add_1AddV2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul_1:z:0n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/AssignAddVariableOpAssignAddVariableOpsn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_assignaddvariableop_resourceen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Const:output:0*
_output_shapes
 *
dtype0	�
un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/meanMeangn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd:output:0~n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/StopGradientStopGradientln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/mean:output:0*
T0*
_output_shapes

:�
pn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/SquaredDifferenceSquaredDifferencegn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd:output:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/varianceMeantn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/SquaredDifference:z:0�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/SqueezeSqueezeln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/Squeeze_1Squeezepn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/addAddV2mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp:value:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add/y:output:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/RsqrtRsqrt^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add:z:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_1AddV2qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/moments/Squeeze_1:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_1/y:output:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Rsqrt_1Rsqrt`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_1:z:0*
T0*
_output_shapes
:�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mulMul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Rsqrt:y:0qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1Mulbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Rsqrt_1:y:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_1ReadVariableOpvn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/subSubon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_1:value:0qn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOp:value:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_2Mul^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub:z:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOpReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_2:z:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOpReadVariableOpun_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3Mul^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul:z:0sn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/PowPowen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow/x:output:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_4Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow:z:0*
T0*
_output_shapes

:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truedivRealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_4:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1:z:0*
T0*
_output_shapes

:�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/NegNegbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/RoundRoundbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_3AddV2^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Round:y:0*
T0*
_output_shapes

:�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradientStopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_3:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_4AddV2bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv:z:0ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient:output:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_1Neg^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_5AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_5/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_5Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_5/x:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_5:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_1Sub^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_1/y:output:0*
T0*
_output_shapes
: �
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value/MinimumMinimum`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_4:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_1:z:0*
T0*
_output_shapes

:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_valueMaximumpn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value/Minimum:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_5:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_6Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_1:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value:z:0*
T0*
_output_shapes

:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_1RealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_6:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow:z:0*
T0*
_output_shapes

:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_7Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_7/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_1:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_2Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_6AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_2:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_7:z:0*
T0*
_output_shapes

:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_8Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_8/x:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_6:z:0*
T0*
_output_shapes

:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_1StopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_8:z:0*
T0*
_output_shapes

:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_7AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_1:output:0*
T0*
_output_shapes

:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2/y:output:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_9Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2:z:0*
T0*
_output_shapes
:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_2RealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_9:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_3Negdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_2:z:0*
T0*
_output_shapes
:�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Round_1Rounddn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_2:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_8AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_3:y:0bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Round_1:y:0*
T0*
_output_shapes
:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_2StopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_8:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_9AddV2dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_2:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_2:output:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_4Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_10AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_4:y:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_10/y:output:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_10Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_10/x:output:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_10:z:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2:z:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_2/y:output:0*
T0*
_output_shapes
: �
nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value_1/MinimumMinimum`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_9:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub_2:z:0*
T0*
_output_shapes
:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value_1Maximumrn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value_1/Minimum:z:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_10:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_11Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_3:z:0jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/clip_by_value_1:z:0*
T0*
_output_shapes
:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_3RealDivan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_11:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Pow_2:z:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_12Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_12/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/truediv_3:z:0*
T0*
_output_shapes
:�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_5Neg`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_11AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/Neg_5:y:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_12:z:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_13Mulhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_13/x:output:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_11:z:0*
T0*
_output_shapes
:�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_3StopGradientan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_13:z:0*
T0*
_output_shapes
:�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_12AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/StopGradient_3:output:0*
T0*
_output_shapes
:�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul_1MatMul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu3/add_3:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_7:z:0*
T0*'
_output_shapes
:����������
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd_1BiasAddin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul_1:product:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_12:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/PowPowdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow/y:output:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/CastCast]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1Powfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_1Cast_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_2Castgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/subSub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_2:y:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_2Powdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Const:output:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub:z:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_1Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_2:z:0*
T0*
_output_shapes
: �
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/LessEqual	LessEqualin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd_1:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_1:z:0*
T0*'
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/ReluReluin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd_1:output:0*
T0*'
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/ones_likeOnesLikein_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd_1:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_2Sub`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Pow_2:z:0*
T0*
_output_shapes
: �
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mulMulcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/ones_like:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_2:z:0*
T0*'
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/SelectV2SelectV2cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/LessEqual:z:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Relu:activations:0]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_1Mulin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd_1:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast:y:0*
T0*'
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truedivRealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_1:y:0*
T0*'
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/NegNegan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/RoundRoundan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv:z:0*
T0*'
_output_shapes
:����������
Yn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/addAddV2]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Neg:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Round:y:0*
T0*'
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/StopGradientStopGradient]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_1AddV2an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/StopGradient:output:0*
T0*'
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_1RealDiv_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_1:z:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast:y:0*
T0*'
_output_shapes
:����������
an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_2RealDivjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_2/x:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast:y:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_3Subfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_3/x:output:0cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_2:z:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_value/MinimumMinimumcn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/truediv_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/sub_3:z:0*
T0*'
_output_shapes
:����������
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_valueMaximumon_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_value/Minimum:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_2Mul`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Cast_1:y:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/clip_by_value:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Neg_1Neggn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/SelectV2:output:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_2AddV2_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/Neg_1:y:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_2:z:0*
T0*'
_output_shapes
:����������
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_3Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_3/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_2:z:0*
T0*'
_output_shapes
:����������
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/StopGradient_1StopGradient_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/mul_3:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_3AddV2gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/SelectV2:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0A�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/PowPowkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow/x:output:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow/y:output:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1Powmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1/x:output:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1/y:output:0*
T0*
_output_shapes
: �
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOpReadVariableOptn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense_output_readvariableop_resource*
_output_shapes

:*
dtype0�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mulMulsn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp:value:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow:z:0*
T0*
_output_shapes

:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truedivRealDivdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul:z:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1:z:0*
T0*
_output_shapes

:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/NegNeghn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truediv:z:0*
T0*
_output_shapes

:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/RoundRoundhn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truediv:z:0*
T0*
_output_shapes

:�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/addAddV2dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Neg:y:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Round:y:0*
T0*
_output_shapes

:�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/StopGradientStopGradientdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add:z:0*
T0*
_output_shapes

:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_1AddV2hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truediv:z:0rn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/StopGradient:output:0*
T0*
_output_shapes

:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Neg_1Negdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow:z:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_2AddV2fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Neg_1:y:0mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_2/y:output:0*
T0*
_output_shapes
: �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_1Mulmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_1/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_2:z:0*
T0*
_output_shapes
: �
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/subSubdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow:z:0kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/sub/y:output:0*
T0*
_output_shapes
: �
rn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/clip_by_value/MinimumMinimumfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_1:z:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/sub:z:0*
T0*
_output_shapes

:�
jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/clip_by_valueMaximumvn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/clip_by_value/Minimum:z:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_1:z:0*
T0*
_output_shapes

:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_2Mulfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow_1:z:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/clip_by_value:z:0*
T0*
_output_shapes

:�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truediv_1RealDivfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_2:z:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Pow:z:0*
T0*
_output_shapes

:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_3Mulmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_3/x:output:0jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/truediv_1:z:0*
T0*
_output_shapes

:�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_1ReadVariableOptn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense_output_readvariableop_resource*
_output_shapes

:*
dtype0�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Neg_2Negun_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_1:value:0*
T0*
_output_shapes

:�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_3AddV2fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/Neg_2:y:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_3:z:0*
T0*
_output_shapes

:�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_4Mulmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_4/x:output:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_3:z:0*
T0*
_output_shapes

:�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/StopGradient_1StopGradientfn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/mul_4:z:0*
T0*
_output_shapes

:�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_2ReadVariableOptn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_false_dense_output_readvariableop_resource*
_output_shapes

:*
dtype0�
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_4AddV2un_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_2:value:0tn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/StopGradient_1:output:0*
T0*
_output_shapes

:�
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/MatMulMatMul_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/relu4/add_3:z:0fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/add_4:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/PowPowen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow/x:output:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow/y:output:0*
T0*
_output_shapes
: �
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/CastCast^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :�
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1Powgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1/x:output:0gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1/y:output:0*
T0*
_output_shapes
: �
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_1Cast`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :�
]n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_2Casthn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/subSuban_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_2:y:0en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub/y:output:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_2Powen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Const:output:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub:z:0*
T0*
_output_shapes
: �
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_1Suban_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_1:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_2:z:0*
T0*
_output_shapes
: �
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/LessEqual	LessEqualmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/MatMul:product:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_1:z:0*
T0*'
_output_shapes
:����������
[n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/ReluRelumn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/MatMul:product:0*
T0*'
_output_shapes
:����������
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/ones_likeOnesLikemn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/MatMul:product:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_2Suban_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_1:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Pow_2:z:0*
T0*
_output_shapes
: �
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mulMuldn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/ones_like:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_2:z:0*
T0*'
_output_shapes
:����������
_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/SelectV2SelectV2dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/LessEqual:z:0in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Relu:activations:0^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_1Mulmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/MatMul:product:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast:y:0*
T0*'
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truedivRealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_1:z:0an_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_1:y:0*
T0*'
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/NegNegbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/RoundRoundbn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv:z:0*
T0*'
_output_shapes
:����������
Zn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/addAddV2^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Neg:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Round:y:0*
T0*'
_output_shapes
:����������
cn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/StopGradientStopGradient^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_1AddV2bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv:z:0ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/StopGradient:output:0*
T0*'
_output_shapes
:����������
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_1RealDiv`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_1:z:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast:y:0*
T0*'
_output_shapes
:����������
bn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_2RealDivkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_2/x:output:0_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast:y:0*
T0*
_output_shapes
: �
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_3Subgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_3/x:output:0dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_2:z:0*
T0*
_output_shapes
: �
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_value/MinimumMinimumdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/truediv_1:z:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/sub_3:z:0*
T0*'
_output_shapes
:����������
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_valueMaximumpn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_value/Minimum:z:0on_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_2Mulan_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Cast_1:y:0hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/clip_by_value:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Neg_1Neghn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/SelectV2:output:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_2AddV2`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/Neg_1:y:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_2:z:0*
T0*'
_output_shapes
:����������
^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_3Mulgn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_3/x:output:0`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_2:z:0*
T0*'
_output_shapes
:����������
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/StopGradient_1StopGradient`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/mul_3:z:0*
T0*'
_output_shapes
:����������
\n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_3AddV2hn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/SelectV2:output:0nn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/StopGradient_1:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity`n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/output/add_3:z:0^NoOp*
T0*'
_output_shapes
:����������#
NoOpNoOpe^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOpg^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_1g^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_2e^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOpg^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_1g^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_2e^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOpg^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_1g^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_2k^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/AssignAddVariableOpn^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOpm^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOpf^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOph^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_1l^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOp�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_1�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_2�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOpj^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOpl^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOpl^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOpj^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOpk^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/AssignAddVariableOpn^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOpm^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOpf^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOph^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_1l^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOp�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_1�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_2�^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOpj^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOpl^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOpl^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOpj^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOpl^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOpn^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_1n^n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_1fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_12�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_2fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp_22�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOpdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv0/ReadVariableOp2�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_1fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_12�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_2fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp_22�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOpdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv1/ReadVariableOp2�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_1fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_12�
fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_2fn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp_22�
dn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOpdn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/conv2/ReadVariableOp2�
jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/AssignAddVariableOpjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/AssignAddVariableOp2�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/BiasAdd/ReadVariableOp2�
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOpln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/MatMul/ReadVariableOp2�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_1gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp_12�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOpen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/add_2/ReadVariableOp2�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_1�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_12�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp_22�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/ReadVariableOp2�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/batch_normalization_27/batchnorm/mul/ReadVariableOp2�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOpin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_1/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/mul_3/ReadVariableOp2�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOpin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense0/sub/ReadVariableOp2�
jn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/AssignAddVariableOpjn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/AssignAddVariableOp2�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOpmn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/BiasAdd/ReadVariableOp2�
ln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOpln_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/MatMul/ReadVariableOp2�
gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_1gn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp_12�
en_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOpen_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/add_2/ReadVariableOp2�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_1�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_12�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_2�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp_22�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/ReadVariableOp2�
�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOp�n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/batch_normalization_28/batchnorm/mul/ReadVariableOp2�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOpin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_1/ReadVariableOp2�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/mul_3/ReadVariableOp2�
in_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOpin_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense1/sub/ReadVariableOp2�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_1mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_12�
mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_2mn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp_22�
kn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOpkn_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False/dense_output/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
(
_output_shapes
:����������

_user_specified_nameinput
�
C
'__inference_relu0_layer_call_fn_2100489

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu0_layer_call_and_return_conditional_losses_2099143h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
.__inference_dense_output_layer_call_fn_2101472

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_output_layer_call_and_return_conditional_losses_2099796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101468:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_output_layer_call_and_return_conditional_losses_2099846

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense1_layer_call_and_return_conditional_losses_2101291

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:L
>batch_normalization_28_assignmovingavg_readvariableop_resource:N
@batch_normalization_28_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add_2/ReadVariableOp�&batch_normalization_28/AssignMovingAvg�5batch_normalization_28/AssignMovingAvg/ReadVariableOp�(batch_normalization_28/AssignMovingAvg_1�7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_28/batchnorm/ReadVariableOp�3batch_normalization_28/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_28/moments/meanMeanBiasAdd:output:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_28/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:05batch_normalization_28/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_28/AssignMovingAvgAssignSubVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_28/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:07batch_normalization_28/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_28/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource0batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/subSub7batch_normalization_28/batchnorm/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R{
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource)^batch_normalization_28/AssignMovingAvg_1*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0c
subSubReadVariableOp_1:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:E
mul_2Mul	mul_1:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_13AddV2moments/Squeeze_1:output:0add_13/y:output:0*
T0*
_output_shapes
:A
Rsqrt_2Rsqrt
add_13:z:0*
T0*
_output_shapes
:�
ReadVariableOp_2ReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource)^batch_normalization_28/AssignMovingAvg_1*
_output_shapes
:*
dtype0M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_14AddV2ReadVariableOp_2:value:0add_14/y:output:0*
T0*
_output_shapes
:=
SqrtSqrt
add_14:z:0*
T0*
_output_shapes
:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:c
add_15AddV2moments/Squeeze_1:output:0add_15/y:output:0*
T0*
_output_shapes
:A
Rsqrt_3Rsqrt
add_15:z:0*
T0*
_output_shapes
:I
mul_14MulSqrt:y:0Rsqrt_3:y:0*
T0*
_output_shapes
:_
Mul_15MulMatMul_1:product:0
mul_14:z:0*
T0*'
_output_shapes
:���������^
	BiasAdd_1BiasAdd
Mul_15:z:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^add_2/ReadVariableOp'^batch_normalization_28/AssignMovingAvg6^batch_normalization_28/AssignMovingAvg/ReadVariableOp)^batch_normalization_28/AssignMovingAvg_18^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp4^batch_normalization_28/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_28/AssignMovingAvg_1(batch_normalization_28/AssignMovingAvg_12P
&batch_normalization_28/AssignMovingAvg&batch_normalization_28/AssignMovingAvg2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu2_layer_call_and_return_conditional_losses_2099327

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������N
ReluReluinputs*
T0*/
_output_shapes
:���������W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv2_layer_call_fn_2100643

inputs!
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_2099277w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100639:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_2100741

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2099334`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_relu1_layer_call_fn_2100589

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_2099235h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�l
�
C__inference_dense1_layer_call_and_return_conditional_losses_2101413

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:H
:batch_normalization_28_batchnorm_readvariableop_1_resource:H
:batch_normalization_28_batchnorm_readvariableop_2_resource:&
assignaddvariableop_resource:	 
identity��AssignAddVariableOp�BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�add_2/ReadVariableOp�/batch_normalization_28/batchnorm/ReadVariableOp�1batch_normalization_28/batchnorm/ReadVariableOp_1�1batch_normalization_28/batchnorm/ReadVariableOp_2�3batch_normalization_28/batchnorm/mul/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_3/ReadVariableOp�sub/ReadVariableOpH
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_28/batchnorm/addAddV27batch_normalization_28/batchnorm/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/mul_1MulBiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_28/batchnorm/mul_2Mul9batch_normalization_28/batchnorm/ReadVariableOp_1:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_28/batchnorm/subSub9batch_normalization_28/batchnorm/ReadVariableOp_2:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R {
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0	h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanBiasAdd:output:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceBiasAdd:output:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:<
RsqrtRsqrtadd:z:0*
T0*
_output_shapes
:L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:a
add_1AddV2moments/Squeeze_1:output:0add_1/y:output:0*
T0*
_output_shapes
:@
Rsqrt_1Rsqrt	add_1:z:0*
T0*
_output_shapes
:�
mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0V
mulMul	Rsqrt:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
mul_1/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0\
mul_1MulRsqrt_1:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:l
ReadVariableOp_1ReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sub/ReadVariableOpReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0e
subSubReadVariableOp_1:value:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:C
mul_2Mulmul:z:0sub:z:0*
T0*
_output_shapes
:�
add_2/ReadVariableOpReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0\
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes
:s
mul_3/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0\
mul_3Mulmul:z:0mul_3/ReadVariableOp:value:0*
T0*
_output_shapes

:J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
mul_4Mul	mul_3:z:0Pow:z:0*
T0*
_output_shapes

:Q
truedivRealDiv	mul_4:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:K
add_3AddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:P
StopGradientStopGradient	add_3:z:0*
T0*
_output_shapes

:[
add_4AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_5AddV2	Neg_1:y:0add_5/y:output:0*
T0*
_output_shapes
: L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_5Mulmul_5/x:output:0	add_5:z:0*
T0*
_output_shapes
: L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
sub_1SubPow:z:0sub_1/y:output:0*
T0*
_output_shapes
: _
clip_by_value/MinimumMinimum	add_4:z:0	sub_1:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_5:z:0*
T0*
_output_shapes

:S
mul_6Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_6:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_7Mulmul_7/x:output:0truediv_1:z:0*
T0*
_output_shapes

:@
Neg_2Neg	mul_3:z:0*
T0*
_output_shapes

:M
add_6AddV2	Neg_2:y:0	mul_7:z:0*
T0*
_output_shapes

:L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_8Mulmul_8/x:output:0	add_6:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_8:z:0*
T0*
_output_shapes

:[
add_7AddV2	mul_3:z:0StopGradient_1:output:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@Q
Pow_2PowPow_2/x:output:0Pow_2/y:output:0*
T0*
_output_shapes
: L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_3PowPow_3/x:output:0Pow_3/y:output:0*
T0*
_output_shapes
: G
mul_9Mul	add_2:z:0	Pow_2:z:0*
T0*
_output_shapes
:O
	truediv_2RealDiv	mul_9:z:0	Pow_3:z:0*
T0*
_output_shapes
:@
Neg_3Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_8AddV2	Neg_3:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_8:z:0*
T0*
_output_shapes
:[
add_9AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:8
Neg_4Neg	Pow_2:z:0*
T0*
_output_shapes
: M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
add_10AddV2	Neg_4:y:0add_10/y:output:0*
T0*
_output_shapes
: M
mul_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
mul_10Mulmul_10/x:output:0
add_10:z:0*
T0*
_output_shapes
: L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
sub_2Sub	Pow_2:z:0sub_2/y:output:0*
T0*
_output_shapes
: ]
clip_by_value_1/MinimumMinimum	add_9:z:0	sub_2:z:0*
T0*
_output_shapes
:h
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0
mul_10:z:0*
T0*
_output_shapes
:R
mul_11Mul	Pow_3:z:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
	truediv_3RealDiv
mul_11:z:0	Pow_2:z:0*
T0*
_output_shapes
:M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?T
mul_12Mulmul_12/x:output:0truediv_3:z:0*
T0*
_output_shapes
:<
Neg_5Neg	add_2:z:0*
T0*
_output_shapes
:K
add_11AddV2	Neg_5:y:0
mul_12:z:0*
T0*
_output_shapes
:M
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Q
mul_13Mulmul_13/x:output:0
add_11:z:0*
T0*
_output_shapes
:O
StopGradient_3StopGradient
mul_13:z:0*
T0*
_output_shapes
:X
add_12AddV2	add_2:z:0StopGradient_3:output:0*
T0*
_output_shapes
:W
MatMul_1MatMulinputs	add_7:z:0*
T0*'
_output_shapes
:���������f
	BiasAdd_1BiasAddMatMul_1:product:0
add_12:z:0*
T0*'
_output_shapes
:���������a
IdentityIdentityBiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignAddVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^add_2/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp2^batch_normalization_28/batchnorm/ReadVariableOp_12^batch_normalization_28/batchnorm/ReadVariableOp_24^batch_normalization_28/batchnorm/mul/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_3/ReadVariableOp^sub/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2f
1batch_normalization_28/batchnorm/ReadVariableOp_11batch_normalization_28/batchnorm/ReadVariableOp_12f
1batch_normalization_28/batchnorm/ReadVariableOp_21batch_normalization_28/batchnorm/ReadVariableOp_22b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2098909

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu1_layer_call_and_return_conditional_losses_2100636

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������N
ReluReluinputs*
T0*/
_output_shapes
:���������W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_dense1_layer_call_fn_2101125

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_2099692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101121:'#
!
_user_specified_name	2101119:'#
!
_user_specified_name	2101117:'#
!
_user_specified_name	2101115:'#
!
_user_specified_name	2101113:'#
!
_user_specified_name	2101111:'#
!
_user_specified_name	2101109:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv2_layer_call_and_return_conditional_losses_2100684

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_2102170
file_prefix7
assignvariableop_conv0_kernel:9
assignvariableop_1_conv1_kernel:9
assignvariableop_2_conv2_kernel:2
 assignvariableop_3_dense0_kernel:
,
assignvariableop_4_dense0_bias:-
#assignvariableop_5_dense0_iteration:	 2
 assignvariableop_6_dense1_kernel:,
assignvariableop_7_dense1_bias:-
#assignvariableop_8_dense1_iteration:	 8
&assignvariableop_9_dense_output_kernel:E
7assignvariableop_10_dense0_batch_normalization_27_gamma:D
6assignvariableop_11_dense0_batch_normalization_27_beta:K
=assignvariableop_12_dense0_batch_normalization_27_moving_mean:O
Aassignvariableop_13_dense0_batch_normalization_27_moving_variance:E
7assignvariableop_14_dense1_batch_normalization_28_gamma:D
6assignvariableop_15_dense1_batch_normalization_28_beta:K
=assignvariableop_16_dense1_batch_normalization_28_moving_mean:O
Aassignvariableop_17_dense1_batch_normalization_28_moving_variance:'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: A
'assignvariableop_20_adam_m_conv0_kernel:A
'assignvariableop_21_adam_v_conv0_kernel:A
'assignvariableop_22_adam_m_conv1_kernel:A
'assignvariableop_23_adam_v_conv1_kernel:A
'assignvariableop_24_adam_m_conv2_kernel:A
'assignvariableop_25_adam_v_conv2_kernel::
(assignvariableop_26_adam_m_dense0_kernel:
:
(assignvariableop_27_adam_v_dense0_kernel:
4
&assignvariableop_28_adam_m_dense0_bias:4
&assignvariableop_29_adam_v_dense0_bias:L
>assignvariableop_30_adam_m_dense0_batch_normalization_27_gamma:L
>assignvariableop_31_adam_v_dense0_batch_normalization_27_gamma:K
=assignvariableop_32_adam_m_dense0_batch_normalization_27_beta:K
=assignvariableop_33_adam_v_dense0_batch_normalization_27_beta::
(assignvariableop_34_adam_m_dense1_kernel::
(assignvariableop_35_adam_v_dense1_kernel:4
&assignvariableop_36_adam_m_dense1_bias:4
&assignvariableop_37_adam_v_dense1_bias:L
>assignvariableop_38_adam_m_dense1_batch_normalization_28_gamma:L
>assignvariableop_39_adam_v_dense1_batch_normalization_28_gamma:K
=assignvariableop_40_adam_m_dense1_batch_normalization_28_beta:K
=assignvariableop_41_adam_v_dense1_batch_normalization_28_beta:@
.assignvariableop_42_adam_m_dense_output_kernel:@
.assignvariableop_43_adam_v_dense_output_kernel:#
assignvariableop_44_total: #
assignvariableop_45_count: 
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/_iteration/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/_iteration/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense0_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense0_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense0_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense1_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_dense_output_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_dense0_batch_normalization_27_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_dense0_batch_normalization_27_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp=assignvariableop_12_dense0_batch_normalization_27_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpAassignvariableop_13_dense0_batch_normalization_27_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_dense1_batch_normalization_28_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp6assignvariableop_15_dense1_batch_normalization_28_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp=assignvariableop_16_dense1_batch_normalization_28_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_dense1_batch_normalization_28_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_m_conv0_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_v_conv0_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_m_conv1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_v_conv1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_conv2_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_conv2_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_m_dense0_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_v_dense0_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_m_dense0_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_v_dense0_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_m_dense0_batch_normalization_27_gammaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_v_dense0_batch_normalization_27_gammaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_m_dense0_batch_normalization_27_betaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp=assignvariableop_33_adam_v_dense0_batch_normalization_27_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_m_dense1_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_v_dense1_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_m_dense1_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adam_v_dense1_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_m_dense1_batch_normalization_28_gammaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp>assignvariableop_39_adam_v_dense1_batch_normalization_28_gammaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_m_dense1_batch_normalization_28_betaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_v_dense1_batch_normalization_28_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_m_dense_output_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_v_dense_output_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_47Identity_47:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%.!

_user_specified_namecount:%-!

_user_specified_nametotal::,6
4
_user_specified_nameAdam/v/dense_output/kernel::+6
4
_user_specified_nameAdam/m/dense_output/kernel:I*E
C
_user_specified_name+)Adam/v/dense1/batch_normalization_28/beta:I)E
C
_user_specified_name+)Adam/m/dense1/batch_normalization_28/beta:J(F
D
_user_specified_name,*Adam/v/dense1/batch_normalization_28/gamma:J'F
D
_user_specified_name,*Adam/m/dense1/batch_normalization_28/gamma:2&.
,
_user_specified_nameAdam/v/dense1/bias:2%.
,
_user_specified_nameAdam/m/dense1/bias:4$0
.
_user_specified_nameAdam/v/dense1/kernel:4#0
.
_user_specified_nameAdam/m/dense1/kernel:I"E
C
_user_specified_name+)Adam/v/dense0/batch_normalization_27/beta:I!E
C
_user_specified_name+)Adam/m/dense0/batch_normalization_27/beta:J F
D
_user_specified_name,*Adam/v/dense0/batch_normalization_27/gamma:JF
D
_user_specified_name,*Adam/m/dense0/batch_normalization_27/gamma:2.
,
_user_specified_nameAdam/v/dense0/bias:2.
,
_user_specified_nameAdam/m/dense0/bias:40
.
_user_specified_nameAdam/v/dense0/kernel:40
.
_user_specified_nameAdam/m/dense0/kernel:3/
-
_user_specified_nameAdam/v/conv2/kernel:3/
-
_user_specified_nameAdam/m/conv2/kernel:3/
-
_user_specified_nameAdam/v/conv1/kernel:3/
-
_user_specified_nameAdam/m/conv1/kernel:3/
-
_user_specified_nameAdam/v/conv0/kernel:3/
-
_user_specified_nameAdam/m/conv0/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:MI
G
_user_specified_name/-dense1/batch_normalization_28/moving_variance:IE
C
_user_specified_name+)dense1/batch_normalization_28/moving_mean:B>
<
_user_specified_name$"dense1/batch_normalization_28/beta:C?
=
_user_specified_name%#dense1/batch_normalization_28/gamma:MI
G
_user_specified_name/-dense0/batch_normalization_27/moving_variance:IE
C
_user_specified_name+)dense0/batch_normalization_27/moving_mean:B>
<
_user_specified_name$"dense0/batch_normalization_27/beta:C?
=
_user_specified_name%#dense0/batch_normalization_27/gamma:3
/
-
_user_specified_namedense_output/kernel:0	,
*
_user_specified_namedense1/iteration:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:0,
*
_user_specified_namedense0/iteration:+'
%
_user_specified_namedense0/bias:-)
'
_user_specified_namedense0/kernel:,(
&
_user_specified_nameconv2/kernel:,(
&
_user_specified_nameconv1/kernel:,(
&
_user_specified_nameconv0/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
8__inference_batch_normalization_27_layer_call_fn_2101578

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2098909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101574:'#
!
_user_specified_name	2101572:'#
!
_user_specified_name	2101570:'#
!
_user_specified_name	2101568:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2100747

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv0_layer_call_and_return_conditional_losses_2099093

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������	Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_relu2_layer_call_fn_2100689

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_2099327h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_27_layer_call_fn_2101591

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2098929o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101587:'#
!
_user_specified_name	2101585:'#
!
_user_specified_name	2101583:'#
!
_user_specified_name	2101581:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2099849	
input'
conv0_2099094:'
conv1_2099186:'
conv2_2099278: 
dense0_2099483:

dense0_2099485:
dense0_2099487:
dense0_2099489:
dense0_2099491:
dense0_2099493:
dense0_2099495:	  
dense1_2099693:
dense1_2099695:
dense1_2099697:
dense1_2099699:
dense1_2099701:
dense1_2099703:
dense1_2099705:	 &
dense_output_2099797:
identity��conv0/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�dense0/StatefulPartitionedCall�dense1/StatefulPartitionedCall�$dense_output/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2099051�
conv0/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv0_2099094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_2099093�
relu0/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu0_layer_call_and_return_conditional_losses_2099143�
conv1/StatefulPartitionedCallStatefulPartitionedCallrelu0/PartitionedCall:output:0conv1_2099186*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_2099185�
relu1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu1_layer_call_and_return_conditional_losses_2099235�
conv2/StatefulPartitionedCallStatefulPartitionedCallrelu1/PartitionedCall:output:0conv2_2099278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_2099277�
relu2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu2_layer_call_and_return_conditional_losses_2099327�
flatten/PartitionedCallPartitionedCallrelu2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2099334�
dense0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense0_2099483dense0_2099485dense0_2099487dense0_2099489dense0_2099491dense0_2099493dense0_2099495*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense0_layer_call_and_return_conditional_losses_2099482�
relu3/PartitionedCallPartitionedCall'dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu3_layer_call_and_return_conditional_losses_2099544�
dense1/StatefulPartitionedCallStatefulPartitionedCallrelu3/PartitionedCall:output:0dense1_2099693dense1_2099695dense1_2099697dense1_2099699dense1_2099701dense1_2099703dense1_2099705*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_2099692�
relu4/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_relu4_layer_call_and_return_conditional_losses_2099754�
$dense_output/StatefulPartitionedCallStatefulPartitionedCallrelu4/PartitionedCall:output:0dense_output_2099797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_output_layer_call_and_return_conditional_losses_2099796�
output/PartitionedCallPartitionedCall-dense_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_2099846n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv0/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense0/StatefulPartitionedCall^dense1/StatefulPartitionedCall%^dense_output/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense0/StatefulPartitionedCalldense0/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2L
$dense_output/StatefulPartitionedCall$dense_output/StatefulPartitionedCall:'#
!
_user_specified_name	2099797:'#
!
_user_specified_name	2099705:'#
!
_user_specified_name	2099703:'#
!
_user_specified_name	2099701:'#
!
_user_specified_name	2099699:'#
!
_user_specified_name	2099697:'#
!
_user_specified_name	2099695:'#
!
_user_specified_name	2099693:'
#
!
_user_specified_name	2099495:'	#
!
_user_specified_name	2099493:'#
!
_user_specified_name	2099491:'#
!
_user_specified_name	2099489:'#
!
_user_specified_name	2099487:'#
!
_user_specified_name	2099485:'#
!
_user_specified_name	2099483:'#
!
_user_specified_name	2099278:'#
!
_user_specified_name	2099186:'#
!
_user_specified_name	2099094:O K
(
_output_shapes
:����������

_user_specified_nameinput
�
^
B__inference_relu3_layer_call_and_return_conditional_losses_2099544

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv1_layer_call_fn_2100543

inputs!
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_2099185w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������	: 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100539:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2098929

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_dense_output_layer_call_and_return_conditional_losses_2101513

inputs)
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0T
mulMulReadVariableOp:value:0Pow:z:0*
T0*
_output_shapes

:O
truedivRealDivmul:z:0	Pow_1:z:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: ]
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*
_output_shapes

:g
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*
_output_shapes

:S
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*
_output_shapes

:Q
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_2NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_4:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:U
MatMulMatMulinputs	add_4:z:0*
T0*'
_output_shapes
:���������_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_dense1_layer_call_fn_2101144

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_2100125o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2101140:'#
!
_user_specified_name	2101138:'#
!
_user_specified_name	2101136:'#
!
_user_specified_name	2101134:'#
!
_user_specified_name	2101132:'#
!
_user_specified_name	2101130:'#
!
_user_specified_name	2101128:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
^
B__inference_relu1_layer_call_and_return_conditional_losses_2099235

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: c
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*/
_output_shapes
:���������N
ReluReluinputs*
T0*/
_output_shapes
:���������W
	ones_likeOnesLikeinputs*
T0*/
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: ^
mulMulones_like:y:0	sub_2:z:0*
T0*/
_output_shapes
:���������z
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������X
mul_1MulinputsCast:y:0*
T0*/
_output_shapes
:���������c
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*/
_output_shapes
:���������Q
NegNegtruediv:z:0*
T0*/
_output_shapes
:���������U
RoundRoundtruediv:z:0*
T0*/
_output_shapes
:���������Z
addAddV2Neg:y:0	Round:y:0*
T0*/
_output_shapes
:���������_
StopGradientStopGradientadd:z:0*
T0*/
_output_shapes
:���������l
add_1AddV2truediv:z:0StopGradient:output:0*
T0*/
_output_shapes
:���������c
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: t
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*/
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:���������e
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*/
_output_shapes
:���������Y
Neg_1NegSelectV2:output:0*
T0*/
_output_shapes
:���������^
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*/
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*/
_output_shapes
:���������c
StopGradient_1StopGradient	mul_3:z:0*
T0*/
_output_shapes
:���������t
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:���������Y
IdentityIdentity	add_3:z:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_relu4_layer_call_and_return_conditional_losses_2099754

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: [
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*'
_output_shapes
:���������F
ReluReluinputs*
T0*'
_output_shapes
:���������O
	ones_likeOnesLikeinputs*
T0*'
_output_shapes
:���������D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: V
mulMulones_like:y:0	sub_2:z:0*
T0*'
_output_shapes
:���������r
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*'
_output_shapes
:���������P
mul_1MulinputsCast:y:0*
T0*'
_output_shapes
:���������[
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������I
NegNegtruediv:z:0*
T0*'
_output_shapes
:���������M
RoundRoundtruediv:z:0*
T0*'
_output_shapes
:���������R
addAddV2Neg:y:0	Round:y:0*
T0*'
_output_shapes
:���������W
StopGradientStopGradientadd:z:0*
T0*'
_output_shapes
:���������d
add_1AddV2truediv:z:0StopGradient:output:0*
T0*'
_output_shapes
:���������[
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*'
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: l
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������]
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*'
_output_shapes
:���������Q
Neg_1NegSelectV2:output:0*
T0*'
_output_shapes
:���������V
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*'
_output_shapes
:���������L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*'
_output_shapes
:���������[
StopGradient_1StopGradient	mul_3:z:0*
T0*'
_output_shapes
:���������l
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*'
_output_shapes
:���������Q
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_conv1_layer_call_and_return_conditional_losses_2100584

inputs1
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2J
Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  0AK
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0\
mulMulReadVariableOp:value:0Pow:z:0*
T0*&
_output_shapes
:W
truedivRealDivmul:z:0	Pow_1:z:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:6
Neg_1NegPow:z:0*
T0*
_output_shapes
: L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
add_2AddV2	Neg_1:y:0add_2/y:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?J
mul_1Mulmul_1/x:output:0	add_2:z:0*
T0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?D
subSubPow:z:0sub/y:output:0*
T0*
_output_shapes
: e
clip_by_value/MinimumMinimum	add_1:z:0sub:z:0*
T0*&
_output_shapes
:o
clip_by_valueMaximumclip_by_value/Minimum:z:0	mul_1:z:0*
T0*&
_output_shapes
:[
mul_2Mul	Pow_1:z:0clip_by_value:z:0*
T0*&
_output_shapes
:Y
	truediv_1RealDiv	mul_2:z:0Pow:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mul_3Mulmul_3/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_2NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_3AddV2	Neg_2:y:0	mul_3:z:0*
T0*&
_output_shapes
:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
mul_4Mulmul_4/x:output:0	add_3:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_4:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_4AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:�
convolutionConv2Dinputs	add_4:z:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
k
IdentityIdentityconvolution:output:0^NoOp*
T0*/
_output_shapes
:���������Y
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������	: 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
'__inference_conv0_layer_call_fn_2100443

inputs!
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_2099093w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2100439:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
8
input/
serving_default_input:0����������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$kernel_quantizer
$kernel_quantizer_internal
%
quantizers

&kernel
 '_jit_compiled_convolution_op"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.	quantizer"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5kernel_quantizer
5kernel_quantizer_internal
6
quantizers

7kernel
 8_jit_compiled_convolution_op"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?	quantizer"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Fkernel_quantizer
Fkernel_quantizer_internal
G
quantizers

Hkernel
 I_jit_compiled_convolution_op"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P	quantizer"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]kernel_quantizer
^bias_quantizer
]kernel_quantizer_internal
^bias_quantizer_internal
_
quantizers
`	batchnorm

akernel
bbias
c
_iteration"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j	quantizer"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qkernel_quantizer
rbias_quantizer
qkernel_quantizer_internal
rbias_quantizer_internal
s
quantizers
t	batchnorm

ukernel
vbias
w
_iteration"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~	quantizer"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel_quantizer
�kernel_quantizer_internal
�
quantizers
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�	quantizer"
_tf_keras_layer
�
&0
71
H2
a3
b4
�5
�6
c7
�8
�9
u10
v11
�12
�13
w14
�15
�16
�17"
trackable_list_wrapper
{
&0
71
H2
a3
b4
�5
�6
u7
v8
�9
�10
�11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100188
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100229�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2099849
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2100147�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
"__inference__wrapped_model_2098875input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_reshape_layer_call_fn_2100422�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_reshape_layer_call_and_return_conditional_losses_2100436�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv0_layer_call_fn_2100443�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv0_layer_call_and_return_conditional_losses_2100484�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
'
$0"
trackable_list_wrapper
&:$2conv0/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu0_layer_call_fn_2100489�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu0_layer_call_and_return_conditional_losses_2100536�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv1_layer_call_fn_2100543�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv1_layer_call_and_return_conditional_losses_2100584�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
'
50"
trackable_list_wrapper
&:$2conv1/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu1_layer_call_fn_2100589�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu1_layer_call_and_return_conditional_losses_2100636�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2_layer_call_fn_2100643�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2_layer_call_and_return_conditional_losses_2100684�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
'
F0"
trackable_list_wrapper
&:$2conv2/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu2_layer_call_fn_2100689�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu2_layer_call_and_return_conditional_losses_2100736�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_layer_call_fn_2100741�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_layer_call_and_return_conditional_losses_2100747�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
U
a0
b1
�2
�3
c4
�5
�6"
trackable_list_wrapper
>
a0
b1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dense0_layer_call_fn_2100766
(__inference_dense0_layer_call_fn_2100785�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dense0_layer_call_and_return_conditional_losses_2100932
C__inference_dense0_layer_call_and_return_conditional_losses_2101054�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
.
]0
^1"
trackable_list_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
:
2dense0/kernel
:2dense0/bias
:	 2dense0/iteration
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu3_layer_call_fn_2101059�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu3_layer_call_and_return_conditional_losses_2101106�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
U
u0
v1
�2
�3
w4
�5
�6"
trackable_list_wrapper
>
u0
v1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dense1_layer_call_fn_2101125
(__inference_dense1_layer_call_fn_2101144�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dense1_layer_call_and_return_conditional_losses_2101291
C__inference_dense1_layer_call_and_return_conditional_losses_2101413�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
:2dense1/kernel
:2dense1/bias
:	 2dense1/iteration
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_relu4_layer_call_fn_2101418�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_relu4_layer_call_and_return_conditional_losses_2101465�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_output_layer_call_fn_2101472�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_output_layer_call_and_return_conditional_losses_2101513�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
(
�0"
trackable_list_wrapper
%:#2dense_output/kernel
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_output_layer_call_fn_2101518�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_output_layer_call_and_return_conditional_losses_2101565�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
1:/2#dense0/batch_normalization_27/gamma
0:.2"dense0/batch_normalization_27/beta
9:7 (2)dense0/batch_normalization_27/moving_mean
=:; (2-dense0/batch_normalization_27/moving_variance
1:/2#dense1/batch_normalization_28/gamma
0:.2"dense1/batch_normalization_28/beta
9:7 (2)dense1/batch_normalization_28/moving_mean
=:; (2-dense1/batch_normalization_28/moving_variance
N
c0
�1
�2
w3
�4
�5"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100188input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100229input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2099849input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2100147input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_2100417input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�	
jinput
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_reshape_layer_call_fn_2100422inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_reshape_layer_call_and_return_conditional_losses_2100436inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv0_layer_call_fn_2100443inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv0_layer_call_and_return_conditional_losses_2100484inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu0_layer_call_fn_2100489inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu0_layer_call_and_return_conditional_losses_2100536inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv1_layer_call_fn_2100543inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv1_layer_call_and_return_conditional_losses_2100584inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu1_layer_call_fn_2100589inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu1_layer_call_and_return_conditional_losses_2100636inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2_layer_call_fn_2100643inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2_layer_call_and_return_conditional_losses_2100684inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu2_layer_call_fn_2100689inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu2_layer_call_and_return_conditional_losses_2100736inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_flatten_layer_call_fn_2100741inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_layer_call_and_return_conditional_losses_2100747inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7
c0
�1
�2"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense0_layer_call_fn_2100766inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dense0_layer_call_fn_2100785inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense0_layer_call_and_return_conditional_losses_2100932inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense0_layer_call_and_return_conditional_losses_2101054inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_27_layer_call_fn_2101578
8__inference_batch_normalization_27_layer_call_fn_2101591�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101625
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101645�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu3_layer_call_fn_2101059inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu3_layer_call_and_return_conditional_losses_2101106inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7
w0
�1
�2"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense1_layer_call_fn_2101125inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dense1_layer_call_fn_2101144inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense1_layer_call_and_return_conditional_losses_2101291inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense1_layer_call_and_return_conditional_losses_2101413inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_28_layer_call_fn_2101658
8__inference_batch_normalization_28_layer_call_fn_2101671�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101705
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101725�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_relu4_layer_call_fn_2101418inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_relu4_layer_call_and_return_conditional_losses_2101465inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_output_layer_call_fn_2101472inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_output_layer_call_and_return_conditional_losses_2101513inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_output_layer_call_fn_2101518inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_output_layer_call_and_return_conditional_losses_2101565inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
+:)2Adam/m/conv0/kernel
+:)2Adam/v/conv0/kernel
+:)2Adam/m/conv1/kernel
+:)2Adam/v/conv1/kernel
+:)2Adam/m/conv2/kernel
+:)2Adam/v/conv2/kernel
$:"
2Adam/m/dense0/kernel
$:"
2Adam/v/dense0/kernel
:2Adam/m/dense0/bias
:2Adam/v/dense0/bias
6:42*Adam/m/dense0/batch_normalization_27/gamma
6:42*Adam/v/dense0/batch_normalization_27/gamma
5:32)Adam/m/dense0/batch_normalization_27/beta
5:32)Adam/v/dense0/batch_normalization_27/beta
$:"2Adam/m/dense1/kernel
$:"2Adam/v/dense1/kernel
:2Adam/m/dense1/bias
:2Adam/v/dense1/bias
6:42*Adam/m/dense1/batch_normalization_28/gamma
6:42*Adam/v/dense1/batch_normalization_28/gamma
5:32)Adam/m/dense1/batch_normalization_28/beta
5:32)Adam/v/dense1/batch_normalization_28/beta
*:(2Adam/m/dense_output/kernel
*:(2Adam/v/dense_output/kernel
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_27_layer_call_fn_2101578inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_27_layer_call_fn_2101591inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101625inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101645inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_28_layer_call_fn_2101658inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_28_layer_call_fn_2101671inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101705inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101725inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_2098875&7Hab����cuv����w�/�,
%�"
 �
input����������
� "/�,
*
output �
output����������
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101625q����7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
S__inference_batch_normalization_27_layer_call_and_return_conditional_losses_2101645q����7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
8__inference_batch_normalization_27_layer_call_fn_2101578f����7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
8__inference_batch_normalization_27_layer_call_fn_2101591f����7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101705q����7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_2101725q����7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
8__inference_batch_normalization_28_layer_call_fn_2101658f����7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
8__inference_batch_normalization_28_layer_call_fn_2101671f����7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
B__inference_conv0_layer_call_and_return_conditional_losses_2100484r&7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������	
� �
'__inference_conv0_layer_call_fn_2100443g&7�4
-�*
(�%
inputs���������
� ")�&
unknown���������	�
B__inference_conv1_layer_call_and_return_conditional_losses_2100584r77�4
-�*
(�%
inputs���������	
� "4�1
*�'
tensor_0���������
� �
'__inference_conv1_layer_call_fn_2100543g77�4
-�*
(�%
inputs���������	
� ")�&
unknown����������
B__inference_conv2_layer_call_and_return_conditional_losses_2100684rH7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
'__inference_conv2_layer_call_fn_2100643gH7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
C__inference_dense0_layer_call_and_return_conditional_losses_2100932pab����c3�0
)�&
 �
inputs���������

p
� ",�)
"�
tensor_0���������
� �
C__inference_dense0_layer_call_and_return_conditional_losses_2101054pab����c3�0
)�&
 �
inputs���������

p 
� ",�)
"�
tensor_0���������
� �
(__inference_dense0_layer_call_fn_2100766eab����c3�0
)�&
 �
inputs���������

p
� "!�
unknown����������
(__inference_dense0_layer_call_fn_2100785eab����c3�0
)�&
 �
inputs���������

p 
� "!�
unknown����������
C__inference_dense1_layer_call_and_return_conditional_losses_2101291puv����w3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
C__inference_dense1_layer_call_and_return_conditional_losses_2101413puv����w3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
(__inference_dense1_layer_call_fn_2101125euv����w3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
(__inference_dense1_layer_call_fn_2101144euv����w3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
I__inference_dense_output_layer_call_and_return_conditional_losses_2101513c�/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_output_layer_call_fn_2101472X�/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_flatten_layer_call_and_return_conditional_losses_2100747g7�4
-�*
(�%
inputs���������
� ",�)
"�
tensor_0���������

� �
)__inference_flatten_layer_call_fn_2100741\7�4
-�*
(�%
inputs���������
� "!�
unknown���������
�
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2099849�&7Hab����cuv����w�7�4
-�*
 �
input����������
p

 
� ",�)
"�
tensor_0���������
� �
�__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_and_return_conditional_losses_2100147�&7Hab����cuv����w�7�4
-�*
 �
input����������
p 

 
� ",�)
"�
tensor_0���������
� �
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100188y&7Hab����cuv����w�7�4
-�*
 �
input����������
p

 
� "!�
unknown����������
q__inference_n_filters_5_n_conv_layers_3_n_dense_units_22_n_dense_layers_2_use_dropout_False_layer_call_fn_2100229y&7Hab����cuv����w�7�4
-�*
 �
input����������
p 

 
� "!�
unknown����������
C__inference_output_layer_call_and_return_conditional_losses_2101565_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_output_layer_call_fn_2101518T/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_relu0_layer_call_and_return_conditional_losses_2100536o7�4
-�*
(�%
inputs���������	
� "4�1
*�'
tensor_0���������	
� �
'__inference_relu0_layer_call_fn_2100489d7�4
-�*
(�%
inputs���������	
� ")�&
unknown���������	�
B__inference_relu1_layer_call_and_return_conditional_losses_2100636o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
'__inference_relu1_layer_call_fn_2100589d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
B__inference_relu2_layer_call_and_return_conditional_losses_2100736o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
'__inference_relu2_layer_call_fn_2100689d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
B__inference_relu3_layer_call_and_return_conditional_losses_2101106_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
'__inference_relu3_layer_call_fn_2101059T/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_relu4_layer_call_and_return_conditional_losses_2101465_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
'__inference_relu4_layer_call_fn_2101418T/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_reshape_layer_call_and_return_conditional_losses_2100436h0�-
&�#
!�
inputs����������
� "4�1
*�'
tensor_0���������
� �
)__inference_reshape_layer_call_fn_2100422]0�-
&�#
!�
inputs����������
� ")�&
unknown����������
%__inference_signature_wrapper_2100417�&7Hab����cuv����w�8�5
� 
.�+
)
input �
input����������"/�,
*
output �
output���������