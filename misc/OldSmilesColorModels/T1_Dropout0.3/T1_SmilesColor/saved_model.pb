??)
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??#
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:0*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:0@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? * 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
:? *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
: *
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:0*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:0*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: (*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
: (*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:(*
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_12/kernel
~
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(0*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:(0*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:0*
dtype0
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P?*!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
:P?*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:?*
dtype0
?
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_21/kernel
~
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
: *
dtype0
?
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_19/kernel
~
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
: *
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: $*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
: $*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:$*
dtype0
?
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:0*
dtype0
?
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: $*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
: $*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:$*
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$(*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:$(*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:(*
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_25/kernel
~
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
: *
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: %*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
: %*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:%*
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *!
shared_nameconv2d_24/kernel
~
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*'
_output_shapes
:? *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:%**!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:%**
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:**
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:J?*!
shared_nameconv2d_28/kernel
~
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*'
_output_shapes
:J?*
dtype0
u
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_28/bias
n
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes	
:?*
dtype0
?
predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namepredictions/kernel
z
&predictions/kernel/Read/ReadVariableOpReadVariableOppredictions/kernel*
_output_shapes
:	?*
dtype0
x
predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namepredictions/bias
q
$predictions/bias/Read/ReadVariableOpReadVariableOppredictions/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
RMSprop/conv2d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d/kernel/rms
?
-RMSprop/conv2d/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d/kernel/rms*&
_output_shapes
: *
dtype0
?
RMSprop/conv2d/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameRMSprop/conv2d/bias/rms

+RMSprop/conv2d/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv2d_4/kernel/rms
?
/RMSprop/conv2d_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_4/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_4/bias/rms
?
-RMSprop/conv2d_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_4/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv2d_2/kernel/rms
?
/RMSprop/conv2d_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_2/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_2/bias/rms
?
-RMSprop/conv2d_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_2/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*,
shared_nameRMSprop/conv2d_5/kernel/rms
?
/RMSprop/conv2d_5/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_5/kernel/rms*&
_output_shapes
: 0*
dtype0
?
RMSprop/conv2d_5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0**
shared_nameRMSprop/conv2d_5/bias/rms
?
-RMSprop/conv2d_5/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_5/bias/rms*
_output_shapes
:0*
dtype0
?
RMSprop/conv2d_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv2d_1/kernel/rms
?
/RMSprop/conv2d_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_1/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_1/bias/rms
?
-RMSprop/conv2d_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_1/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv2d_3/kernel/rms
?
/RMSprop/conv2d_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_3/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_3/bias/rms
?
-RMSprop/conv2d_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_3/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_6/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*,
shared_nameRMSprop/conv2d_6/kernel/rms
?
/RMSprop/conv2d_6/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_6/kernel/rms*&
_output_shapes
:0@*
dtype0
?
RMSprop/conv2d_6/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameRMSprop/conv2d_6/bias/rms
?
-RMSprop/conv2d_6/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_6/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2d_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *,
shared_nameRMSprop/conv2d_7/kernel/rms
?
/RMSprop/conv2d_7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_7/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_7/bias/rms
?
-RMSprop/conv2d_7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_7/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_9/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv2d_9/kernel/rms
?
/RMSprop/conv2d_9/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_9/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_9/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv2d_9/bias/rms
?
-RMSprop/conv2d_9/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_9/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_10/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameRMSprop/conv2d_10/kernel/rms
?
0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_10/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_10/bias/rms
?
.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_8/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*,
shared_nameRMSprop/conv2d_8/kernel/rms
?
/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/kernel/rms*&
_output_shapes
: 0*
dtype0
?
RMSprop/conv2d_8/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0**
shared_nameRMSprop/conv2d_8/bias/rms
?
-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/bias/rms*
_output_shapes
:0*
dtype0
?
RMSprop/conv2d_11/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*-
shared_nameRMSprop/conv2d_11/kernel/rms
?
0RMSprop/conv2d_11/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_11/kernel/rms*&
_output_shapes
: 0*
dtype0
?
RMSprop/conv2d_11/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameRMSprop/conv2d_11/bias/rms
?
.RMSprop/conv2d_11/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_11/bias/rms*
_output_shapes
:0*
dtype0
?
RMSprop/conv2d_13/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_13/kernel/rms
?
0RMSprop/conv2d_13/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_13/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_13/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_13/bias/rms
?
.RMSprop/conv2d_13/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_13/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_14/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: (*-
shared_nameRMSprop/conv2d_14/kernel/rms
?
0RMSprop/conv2d_14/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_14/kernel/rms*&
_output_shapes
: (*
dtype0
?
RMSprop/conv2d_14/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_nameRMSprop/conv2d_14/bias/rms
?
.RMSprop/conv2d_14/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_14/bias/rms*
_output_shapes
:(*
dtype0
?
RMSprop/conv2d_12/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_12/kernel/rms
?
0RMSprop/conv2d_12/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_12/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_12/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_12/bias/rms
?
.RMSprop/conv2d_12/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_12/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_15/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:(0*-
shared_nameRMSprop/conv2d_15/kernel/rms
?
0RMSprop/conv2d_15/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_15/kernel/rms*&
_output_shapes
:(0*
dtype0
?
RMSprop/conv2d_15/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameRMSprop/conv2d_15/bias/rms
?
.RMSprop/conv2d_15/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_15/bias/rms*
_output_shapes
:0*
dtype0
?
RMSprop/conv2d_16/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:P?*-
shared_nameRMSprop/conv2d_16/kernel/rms
?
0RMSprop/conv2d_16/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_16/kernel/rms*'
_output_shapes
:P?*
dtype0
?
RMSprop/conv2d_16/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameRMSprop/conv2d_16/bias/rms
?
.RMSprop/conv2d_16/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_16/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv2d_21/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_21/kernel/rms
?
0RMSprop/conv2d_21/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_21/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_21/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_21/bias/rms
?
.RMSprop/conv2d_21/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_21/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_17/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_17/kernel/rms
?
0RMSprop/conv2d_17/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_17/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_17/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_17/bias/rms
?
.RMSprop/conv2d_17/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_17/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_19/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_19/kernel/rms
?
0RMSprop/conv2d_19/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_19/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_19/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_19/bias/rms
?
.RMSprop/conv2d_19/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_19/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_22/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: $*-
shared_nameRMSprop/conv2d_22/kernel/rms
?
0RMSprop/conv2d_22/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_22/kernel/rms*&
_output_shapes
: $*
dtype0
?
RMSprop/conv2d_22/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*+
shared_nameRMSprop/conv2d_22/bias/rms
?
.RMSprop/conv2d_22/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_22/bias/rms*
_output_shapes
:$*
dtype0
?
RMSprop/conv2d_18/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*-
shared_nameRMSprop/conv2d_18/kernel/rms
?
0RMSprop/conv2d_18/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_18/kernel/rms*&
_output_shapes
: 0*
dtype0
?
RMSprop/conv2d_18/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameRMSprop/conv2d_18/bias/rms
?
.RMSprop/conv2d_18/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_18/bias/rms*
_output_shapes
:0*
dtype0
?
RMSprop/conv2d_20/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: $*-
shared_nameRMSprop/conv2d_20/kernel/rms
?
0RMSprop/conv2d_20/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_20/kernel/rms*&
_output_shapes
: $*
dtype0
?
RMSprop/conv2d_20/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*+
shared_nameRMSprop/conv2d_20/bias/rms
?
.RMSprop/conv2d_20/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_20/bias/rms*
_output_shapes
:$*
dtype0
?
RMSprop/conv2d_23/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:$(*-
shared_nameRMSprop/conv2d_23/kernel/rms
?
0RMSprop/conv2d_23/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_23/kernel/rms*&
_output_shapes
:$(*
dtype0
?
RMSprop/conv2d_23/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*+
shared_nameRMSprop/conv2d_23/bias/rms
?
.RMSprop/conv2d_23/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_23/bias/rms*
_output_shapes
:(*
dtype0
?
RMSprop/conv2d_25/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_25/kernel/rms
?
0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_25/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_25/bias/rms
?
.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_26/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: %*-
shared_nameRMSprop/conv2d_26/kernel/rms
?
0RMSprop/conv2d_26/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_26/kernel/rms*&
_output_shapes
: %*
dtype0
?
RMSprop/conv2d_26/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*+
shared_nameRMSprop/conv2d_26/bias/rms
?
.RMSprop/conv2d_26/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_26/bias/rms*
_output_shapes
:%*
dtype0
?
RMSprop/conv2d_24/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *-
shared_nameRMSprop/conv2d_24/kernel/rms
?
0RMSprop/conv2d_24/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_24/kernel/rms*'
_output_shapes
:? *
dtype0
?
RMSprop/conv2d_24/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_24/bias/rms
?
.RMSprop/conv2d_24/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_24/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_27/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:%**-
shared_nameRMSprop/conv2d_27/kernel/rms
?
0RMSprop/conv2d_27/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_27/kernel/rms*&
_output_shapes
:%**
dtype0
?
RMSprop/conv2d_27/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**+
shared_nameRMSprop/conv2d_27/bias/rms
?
.RMSprop/conv2d_27/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_27/bias/rms*
_output_shapes
:**
dtype0
?
RMSprop/conv2d_28/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:J?*-
shared_nameRMSprop/conv2d_28/kernel/rms
?
0RMSprop/conv2d_28/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_28/kernel/rms*'
_output_shapes
:J?*
dtype0
?
RMSprop/conv2d_28/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameRMSprop/conv2d_28/bias/rms
?
.RMSprop/conv2d_28/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_28/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/predictions/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name RMSprop/predictions/kernel/rms
?
2RMSprop/predictions/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/predictions/kernel/rms*
_output_shapes
:	?*
dtype0
?
RMSprop/predictions/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/predictions/bias/rms
?
0RMSprop/predictions/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/predictions/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer-27
layer_with_weights-17
layer-28
layer_with_weights-18
layer-29
layer_with_weights-19
layer-30
 layer_with_weights-20
 layer-31
!layer-32
"layer_with_weights-21
"layer-33
#layer_with_weights-22
#layer-34
$layer_with_weights-23
$layer-35
%layer-36
&layer-37
'layer_with_weights-24
'layer-38
(layer_with_weights-25
(layer-39
)layer_with_weights-26
)layer-40
*layer_with_weights-27
*layer-41
+layer-42
,layer_with_weights-28
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer_with_weights-29
1layer-48
2	optimizer
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7
signatures
 
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
h

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
h

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api
h

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
l

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?iter

?decay
?learning_rate
?momentum
?rho
8rms?
9rms?
Brms?
Crms?
Hrms?
Irms?
Nrms?
Orms?
Trms?
Urms?
Zrms?
[rms?
`rms?
arms?
jrms?
krms?
xrms?
yrms?
~rms?
rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms?
?
80
91
B2
C3
H4
I5
N6
O7
T8
U9
Z10
[11
`12
a13
j14
k15
x16
y17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?
80
91
B2
C3
H4
I5
N6
O7
T8
U9
Z10
[11
`12
a13
j14
k15
x16
y17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

Z0
[1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1

~0
1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
\Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_11/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_11/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_13/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_13/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_14/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_14/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_12/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_12/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_16/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_16/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_21/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_21/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_17/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_17/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_19/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_19/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_22/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_22/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_18/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_18/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_20/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_20/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_23/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_23/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_25/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_25/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_26/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_26/bias5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_24/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_24/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_27/kernel7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_27/bias5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEconv2d_28/kernel7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_28/bias5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
_]
VARIABLE_VALUEpredictions/kernel7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEpredictions/bias5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
?
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
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUERMSprop/conv2d/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/conv2d/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_4/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_4/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_5/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_5/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_1/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_1/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_3/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_3/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_6/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_6/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_7/kernel/rmsTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_7/bias/rmsRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_9/kernel/rmsTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv2d_9/bias/rmsRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_10/kernel/rmsTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_10/bias/rmsRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_8/kernel/rmsUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_8/bias/rmsSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_11/kernel/rmsUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_11/bias/rmsSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_13/kernel/rmsUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_13/bias/rmsSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_14/kernel/rmsUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_14/bias/rmsSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_12/kernel/rmsUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_12/bias/rmsSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_15/kernel/rmsUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_15/bias/rmsSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_16/kernel/rmsUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_16/bias/rmsSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_21/kernel/rmsUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_21/bias/rmsSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_17/kernel/rmsUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_17/bias/rmsSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_19/kernel/rmsUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_19/bias/rmsSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_22/kernel/rmsUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_22/bias/rmsSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_18/kernel/rmsUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_18/bias/rmsSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_20/kernel/rmsUlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_20/bias/rmsSlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_23/kernel/rmsUlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_23/bias/rmsSlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_25/kernel/rmsUlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_25/bias/rmsSlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_26/kernel/rmsUlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_26/bias/rmsSlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_24/kernel/rmsUlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_24/bias/rmsSlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_27/kernel/rmsUlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_27/bias/rmsSlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_28/kernel/rmsUlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_28/bias/rmsSlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/predictions/kernel/rmsUlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/predictions/bias/rmsSlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*0
_output_shapes
:?????????&?*
dtype0*%
shape:?????????&?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_2/kernelconv2d_2/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_8/kernelconv2d_8/biasconv2d_11/kernelconv2d_11/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_12/kernelconv2d_12/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_19/kernelconv2d_19/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_20/kernelconv2d_20/biasconv2d_23/kernelconv2d_23/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_24/kernelconv2d_24/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biaspredictions/kernelpredictions/bias*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_268186
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp&predictions/kernel/Read/ReadVariableOp$predictions/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-RMSprop/conv2d/kernel/rms/Read/ReadVariableOp+RMSprop/conv2d/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_4/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_4/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_2/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_2/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_5/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_5/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_1/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_1/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_3/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_3/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_6/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_6/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_7/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_7/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_9/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_9/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOp/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_11/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_11/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_13/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_13/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_14/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_14/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_12/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_12/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_15/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_15/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_16/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_16/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_21/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_21/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_17/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_17/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_19/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_19/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_22/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_22/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_18/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_18/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_20/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_20/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_23/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_23/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_26/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_26/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_24/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_24/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_27/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_27/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_28/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_28/bias/rms/Read/ReadVariableOp2RMSprop/predictions/kernel/rms/Read/ReadVariableOp0RMSprop/predictions/bias/rms/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_270175
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_4/kernelconv2d_4/biasconv2d_2/kernelconv2d_2/biasconv2d_5/kernelconv2d_5/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_8/kernelconv2d_8/biasconv2d_11/kernelconv2d_11/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_12/kernelconv2d_12/biasconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_21/kernelconv2d_21/biasconv2d_17/kernelconv2d_17/biasconv2d_19/kernelconv2d_19/biasconv2d_22/kernelconv2d_22/biasconv2d_18/kernelconv2d_18/biasconv2d_20/kernelconv2d_20/biasconv2d_23/kernelconv2d_23/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_24/kernelconv2d_24/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biaspredictions/kernelpredictions/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d/kernel/rmsRMSprop/conv2d/bias/rmsRMSprop/conv2d_4/kernel/rmsRMSprop/conv2d_4/bias/rmsRMSprop/conv2d_2/kernel/rmsRMSprop/conv2d_2/bias/rmsRMSprop/conv2d_5/kernel/rmsRMSprop/conv2d_5/bias/rmsRMSprop/conv2d_1/kernel/rmsRMSprop/conv2d_1/bias/rmsRMSprop/conv2d_3/kernel/rmsRMSprop/conv2d_3/bias/rmsRMSprop/conv2d_6/kernel/rmsRMSprop/conv2d_6/bias/rmsRMSprop/conv2d_7/kernel/rmsRMSprop/conv2d_7/bias/rmsRMSprop/conv2d_9/kernel/rmsRMSprop/conv2d_9/bias/rmsRMSprop/conv2d_10/kernel/rmsRMSprop/conv2d_10/bias/rmsRMSprop/conv2d_8/kernel/rmsRMSprop/conv2d_8/bias/rmsRMSprop/conv2d_11/kernel/rmsRMSprop/conv2d_11/bias/rmsRMSprop/conv2d_13/kernel/rmsRMSprop/conv2d_13/bias/rmsRMSprop/conv2d_14/kernel/rmsRMSprop/conv2d_14/bias/rmsRMSprop/conv2d_12/kernel/rmsRMSprop/conv2d_12/bias/rmsRMSprop/conv2d_15/kernel/rmsRMSprop/conv2d_15/bias/rmsRMSprop/conv2d_16/kernel/rmsRMSprop/conv2d_16/bias/rmsRMSprop/conv2d_21/kernel/rmsRMSprop/conv2d_21/bias/rmsRMSprop/conv2d_17/kernel/rmsRMSprop/conv2d_17/bias/rmsRMSprop/conv2d_19/kernel/rmsRMSprop/conv2d_19/bias/rmsRMSprop/conv2d_22/kernel/rmsRMSprop/conv2d_22/bias/rmsRMSprop/conv2d_18/kernel/rmsRMSprop/conv2d_18/bias/rmsRMSprop/conv2d_20/kernel/rmsRMSprop/conv2d_20/bias/rmsRMSprop/conv2d_23/kernel/rmsRMSprop/conv2d_23/bias/rmsRMSprop/conv2d_25/kernel/rmsRMSprop/conv2d_25/bias/rmsRMSprop/conv2d_26/kernel/rmsRMSprop/conv2d_26/bias/rmsRMSprop/conv2d_24/kernel/rmsRMSprop/conv2d_24/bias/rmsRMSprop/conv2d_27/kernel/rmsRMSprop/conv2d_27/bias/rmsRMSprop/conv2d_28/kernel/rmsRMSprop/conv2d_28/bias/rmsRMSprop/predictions/kernel/rmsRMSprop/predictions/bias/rms*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_270572??
?

?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_269094

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242

inputs8
conv2d_readvariableop_resource: (-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: (*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?(j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_269136

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_19_layer_call_fn_269424

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_265922

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_28_layer_call_and_return_conditional_losses_269674

inputs9
conv2d_readvariableop_resource:J?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:J?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????J?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????JJ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????JJ
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269176

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????	? *
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????	? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_27_layer_call_fn_269631

inputs!
unknown:%*
	unknown_0:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J%: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????J%
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_2_layer_call_fn_269327
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????	?P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????	? :?????????	?0:Z V
0
_output_shapes
:?????????	? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/1
?
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_5_layer_call_and_return_conditional_losses_269696

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????J?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
?
*__inference_conv2d_17_layer_call_fn_269404

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_20_layer_call_and_return_conditional_losses_269515

inputs8
conv2d_readvariableop_resource: $-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J$i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
m
A__inference_add_1_layer_call_and_return_conditional_losses_269365
inputs_0
inputs_1
identity\
addAddV2inputs_0inputs_1*
T0*1
_output_shapes
:?????????	??Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????	??:?????????	??:[ W
1
_output_shapes
:?????????	??
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????	??
"
_user_specified_name
inputs/1
?
G
+__inference_activation_layer_call_fn_268935

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_265978i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_4_layer_call_fn_269557

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_266463i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
?
)__inference_conv2d_2_layer_call_fn_268969

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
G__inference_Chemception_layer_call_and_return_conditional_losses_266605

inputs'
conv2d_265968: 
conv2d_265970: )
conv2d_4_265992:  
conv2d_4_265994: )
conv2d_5_266009: 0
conv2d_5_266011:0)
conv2d_2_266026:  
conv2d_2_266028: )
conv2d_1_266043:  
conv2d_1_266045: )
conv2d_3_266060:  
conv2d_3_266062: )
conv2d_6_266077:0@
conv2d_6_266079:@*
conv2d_7_266103:? 
conv2d_7_266105: )
conv2d_9_266135:  
conv2d_9_266137: *
conv2d_10_266152:  
conv2d_10_266154: )
conv2d_8_266175: 0
conv2d_8_266177:0*
conv2d_11_266192: 0
conv2d_11_266194:0+
conv2d_13_266226:? 
conv2d_13_266228: *
conv2d_14_266243: (
conv2d_14_266245:(+
conv2d_12_266260:? 
conv2d_12_266262: *
conv2d_15_266277:(0
conv2d_15_266279:0+
conv2d_16_266302:P?
conv2d_16_266304:	?+
conv2d_21_266334:? 
conv2d_21_266336: *
conv2d_22_266351: $
conv2d_22_266353:$+
conv2d_19_266368:? 
conv2d_19_266370: +
conv2d_17_266385:? 
conv2d_17_266387: *
conv2d_18_266408: 0
conv2d_18_266410:0*
conv2d_20_266425: $
conv2d_20_266427:$*
conv2d_23_266442:$(
conv2d_23_266444:(+
conv2d_25_266477:? 
conv2d_25_266479: *
conv2d_26_266494: %
conv2d_26_266496:%+
conv2d_24_266511:? 
conv2d_24_266513: *
conv2d_27_266528:%*
conv2d_27_266530:*+
conv2d_28_266553:J?
conv2d_28_266555:	?%
predictions_266599:	? 
predictions_266601:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_265968conv2d_265970*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_265967?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_265978?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_4_265992conv2d_4_265994*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_266009conv2d_5_266011*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_2_266026conv2d_2_266028*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_266043conv2d_1_266045*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_266060conv2d_3_266062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_266077conv2d_6_266079*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076?
concatenate/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_266090?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_7_266103conv2d_7_266105*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102?
add/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_266114?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_266121?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_9_266135conv2d_9_266137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_266152conv2d_10_266154*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151?
max_pooling2d/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_8_266175conv2d_8_266177*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_266192conv2d_11_266194*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191?
concatenate_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_8/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205?
activation_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_266212?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_13_266226conv2d_13_266228*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_266243conv2d_14_266245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_12_266260conv2d_12_266262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_266277conv2d_15_266279*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276?
concatenate_2/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_16_266302conv2d_16_266304*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301?
add_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_266313?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_266320?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_21_266334conv2d_21_266336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_266351conv2d_22_266353*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_19_266368conv2d_19_266370*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_17_266385conv2d_17_266387*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_266408conv2d_18_266410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_266425conv2d_20_266427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_266442conv2d_23_266444*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441?
concatenate_3/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456?
activation_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_266463?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_25_266477conv2d_25_266479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_266494conv2d_26_266496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_24_266511conv2d_24_266513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_266528conv2d_27_266530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527?
concatenate_4/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????JJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_28_266553conv2d_28_266555*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552?
add_2/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_266564?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_266571?
final_pool/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_266578?
dropout_end/PartitionedCallPartitionedCall#final_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266585?
#predictions/StatefulPartitionedCallStatefulPartitionedCall$dropout_end/PartitionedCall:output:0predictions_266599predictions_266601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_predictions_layer_call_and_return_conditional_losses_266598{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
??
?6
__inference__traced_save_270175
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop1
-savev2_predictions_kernel_read_readvariableop/
+savev2_predictions_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_rmsprop_conv2d_kernel_rms_read_readvariableop6
2savev2_rmsprop_conv2d_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_4_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_4_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_2_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_2_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_5_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_5_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_1_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_1_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_3_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_3_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_6_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_6_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_7_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_7_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_9_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_9_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2d_8_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_8_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_11_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_11_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_13_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_13_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_14_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_14_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_12_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_12_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_15_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_15_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_16_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_16_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_21_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_21_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_17_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_17_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_19_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_19_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_22_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_22_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_18_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_18_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_20_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_20_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_23_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_23_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_26_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_26_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_24_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_24_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_27_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_27_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_28_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_28_bias_rms_read_readvariableop=
9savev2_rmsprop_predictions_kernel_rms_read_readvariableop;
7savev2_rmsprop_predictions_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?F
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?F
value?FB?F?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop-savev2_predictions_kernel_read_readvariableop+savev2_predictions_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_rmsprop_conv2d_kernel_rms_read_readvariableop2savev2_rmsprop_conv2d_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_4_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_4_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_2_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_2_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_5_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_5_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_1_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_1_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_3_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_3_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_6_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_6_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_7_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_7_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_9_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_9_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableop6savev2_rmsprop_conv2d_8_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_8_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_11_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_11_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_13_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_13_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_14_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_14_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_12_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_12_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_15_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_15_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_16_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_16_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_21_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_21_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_17_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_17_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_19_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_19_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_22_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_22_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_18_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_18_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_20_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_20_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_23_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_23_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_26_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_26_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_24_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_24_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_27_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_27_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_28_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_28_bias_rms_read_readvariableop9savev2_rmsprop_predictions_kernel_rms_read_readvariableop7savev2_rmsprop_predictions_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : :  : : 0:0:  : :  : :0@:@:? : :  : :  : : 0:0: 0:0:? : : (:(:? : :(0:0:P?:?:? : :? : :? : : $:$: 0:0: $:$:$(:(:? : : %:%:? : :%*:*:J?:?:	?:: : : : : : : : : : : :  : :  : : 0:0:  : :  : :0@:@:? : :  : :  : : 0:0: 0:0:? : : (:(:? : :(0:0:P?:?:? : :? : :? : : $:$: 0:0: $:$:$(:(:? : : %:%:? : :%*:*:J?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:,	(
&
_output_shapes
:  : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:0@: 

_output_shapes
:@:-)
'
_output_shapes
:? : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:,(
&
_output_shapes
: 0: 

_output_shapes
:0:-)
'
_output_shapes
:? : 

_output_shapes
: :,(
&
_output_shapes
: (: 

_output_shapes
:(:-)
'
_output_shapes
:? : 

_output_shapes
: :,(
&
_output_shapes
:(0:  

_output_shapes
:0:-!)
'
_output_shapes
:P?:!"

_output_shapes	
:?:-#)
'
_output_shapes
:? : $

_output_shapes
: :-%)
'
_output_shapes
:? : &

_output_shapes
: :-')
'
_output_shapes
:? : (

_output_shapes
: :,)(
&
_output_shapes
: $: *

_output_shapes
:$:,+(
&
_output_shapes
: 0: ,

_output_shapes
:0:,-(
&
_output_shapes
: $: .

_output_shapes
:$:,/(
&
_output_shapes
:$(: 0

_output_shapes
:(:-1)
'
_output_shapes
:? : 2

_output_shapes
: :,3(
&
_output_shapes
: %: 4

_output_shapes
:%:-5)
'
_output_shapes
:? : 6

_output_shapes
: :,7(
&
_output_shapes
:%*: 8

_output_shapes
:*:-9)
'
_output_shapes
:J?:!:

_output_shapes	
:?:%;!

_output_shapes
:	?: <

_output_shapes
::=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :,F(
&
_output_shapes
: : G

_output_shapes
: :,H(
&
_output_shapes
:  : I

_output_shapes
: :,J(
&
_output_shapes
:  : K

_output_shapes
: :,L(
&
_output_shapes
: 0: M

_output_shapes
:0:,N(
&
_output_shapes
:  : O

_output_shapes
: :,P(
&
_output_shapes
:  : Q

_output_shapes
: :,R(
&
_output_shapes
:0@: S

_output_shapes
:@:-T)
'
_output_shapes
:? : U

_output_shapes
: :,V(
&
_output_shapes
:  : W

_output_shapes
: :,X(
&
_output_shapes
:  : Y

_output_shapes
: :,Z(
&
_output_shapes
: 0: [

_output_shapes
:0:,\(
&
_output_shapes
: 0: ]

_output_shapes
:0:-^)
'
_output_shapes
:? : _

_output_shapes
: :,`(
&
_output_shapes
: (: a

_output_shapes
:(:-b)
'
_output_shapes
:? : c

_output_shapes
: :,d(
&
_output_shapes
:(0: e

_output_shapes
:0:-f)
'
_output_shapes
:P?:!g

_output_shapes	
:?:-h)
'
_output_shapes
:? : i

_output_shapes
: :-j)
'
_output_shapes
:? : k

_output_shapes
: :-l)
'
_output_shapes
:? : m

_output_shapes
: :,n(
&
_output_shapes
: $: o

_output_shapes
:$:,p(
&
_output_shapes
: 0: q

_output_shapes
:0:,r(
&
_output_shapes
: $: s

_output_shapes
:$:,t(
&
_output_shapes
:$(: u

_output_shapes
:(:-v)
'
_output_shapes
:? : w

_output_shapes
: :,x(
&
_output_shapes
: %: y

_output_shapes
:%:-z)
'
_output_shapes
:? : {

_output_shapes
: :,|(
&
_output_shapes
:%*: }

_output_shapes
:*:-~)
'
_output_shapes
:J?:!

_output_shapes	
:?:&?!

_output_shapes
:	?:!?

_output_shapes
::?

_output_shapes
: 
?
J
.__inference_max_pooling2d_layer_call_fn_269161

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_265922?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_23_layer_call_fn_269524

inputs!
unknown:$(
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?$: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	?$
 
_user_specified_nameinputs
?
I
-__inference_activation_5_layer_call_fn_269691

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_266571i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_265934

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_269455

inputs8
conv2d_readvariableop_resource: $-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?$j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_269281

inputs8
conv2d_readvariableop_resource: (-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: (*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?(j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_269241

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:?????????	??d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
*__inference_conv2d_15_layer_call_fn_269310

inputs!
unknown:(0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?(: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	?(
 
_user_specified_nameinputs
?
?
,__inference_Chemception_layer_call_fn_267709
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11:0@

unknown_12:@%

unknown_13:? 

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: $

unknown_19: 0

unknown_20:0$

unknown_21: 0

unknown_22:0%

unknown_23:? 

unknown_24: $

unknown_25: (

unknown_26:(%

unknown_27:? 

unknown_28: $

unknown_29:(0

unknown_30:0%

unknown_31:P?

unknown_32:	?%

unknown_33:? 

unknown_34: $

unknown_35: $

unknown_36:$%

unknown_37:? 

unknown_38: %

unknown_39:? 

unknown_40: $

unknown_41: 0

unknown_42:0$

unknown_43: $

unknown_44:$$

unknown_45:$(

unknown_46:(%

unknown_47:? 

unknown_48: $

unknown_49: %

unknown_50:%%

unknown_51:? 

unknown_52: $

unknown_53:%*

unknown_54:*%

unknown_55:J?

unknown_56:	?

unknown_57:	?

unknown_58:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Chemception_layer_call_and_return_conditional_losses_267461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
?
?
*__inference_conv2d_16_layer_call_fn_269343

inputs"
unknown:P?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:?????????	??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?P: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	?P
 
_user_specified_nameinputs
?

?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_269353

inputs9
conv2d_readvariableop_resource:P?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:P?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:?????????	??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?P
 
_user_specified_nameinputs
?
I
-__inference_activation_3_layer_call_fn_269370

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_266320j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
H
,__inference_dropout_end_layer_call_fn_269723

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266585a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350

inputs8
conv2d_readvariableop_resource: $-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?$j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493

inputs8
conv2d_readvariableop_resource: %-
biasadd_readvariableop_resource:%
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: %*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J%i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J%w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????J 
 
_user_specified_nameinputs
?
?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????	??a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????	? :?????????	?0:?????????	?0:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????	?0
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????	?0
 
_user_specified_nameinputs
?
?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_269495

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
??
?-
G__inference_Chemception_layer_call_and_return_conditional_losses_268911

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: 06
(conv2d_5_biasadd_readvariableop_resource:0A
'conv2d_2_conv2d_readvariableop_resource:  6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource:0@6
(conv2d_6_biasadd_readvariableop_resource:@B
'conv2d_7_conv2d_readvariableop_resource:? 6
(conv2d_7_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource:  6
(conv2d_9_biasadd_readvariableop_resource: B
(conv2d_10_conv2d_readvariableop_resource:  7
)conv2d_10_biasadd_readvariableop_resource: A
'conv2d_8_conv2d_readvariableop_resource: 06
(conv2d_8_biasadd_readvariableop_resource:0B
(conv2d_11_conv2d_readvariableop_resource: 07
)conv2d_11_biasadd_readvariableop_resource:0C
(conv2d_13_conv2d_readvariableop_resource:? 7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: (7
)conv2d_14_biasadd_readvariableop_resource:(C
(conv2d_12_conv2d_readvariableop_resource:? 7
)conv2d_12_biasadd_readvariableop_resource: B
(conv2d_15_conv2d_readvariableop_resource:(07
)conv2d_15_biasadd_readvariableop_resource:0C
(conv2d_16_conv2d_readvariableop_resource:P?8
)conv2d_16_biasadd_readvariableop_resource:	?C
(conv2d_21_conv2d_readvariableop_resource:? 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource: $7
)conv2d_22_biasadd_readvariableop_resource:$C
(conv2d_19_conv2d_readvariableop_resource:? 7
)conv2d_19_biasadd_readvariableop_resource: C
(conv2d_17_conv2d_readvariableop_resource:? 7
)conv2d_17_biasadd_readvariableop_resource: B
(conv2d_18_conv2d_readvariableop_resource: 07
)conv2d_18_biasadd_readvariableop_resource:0B
(conv2d_20_conv2d_readvariableop_resource: $7
)conv2d_20_biasadd_readvariableop_resource:$B
(conv2d_23_conv2d_readvariableop_resource:$(7
)conv2d_23_biasadd_readvariableop_resource:(C
(conv2d_25_conv2d_readvariableop_resource:? 7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: %7
)conv2d_26_biasadd_readvariableop_resource:%C
(conv2d_24_conv2d_readvariableop_resource:? 7
)conv2d_24_biasadd_readvariableop_resource: B
(conv2d_27_conv2d_readvariableop_resource:%*7
)conv2d_27_biasadd_readvariableop_resource:*C
(conv2d_28_conv2d_readvariableop_resource:J?8
)conv2d_28_biasadd_readvariableop_resource:	?=
*predictions_matmul_readvariableop_resource:	?9
+predictions_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?"predictions/BiasAdd/ReadVariableOp?!predictions/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2Dactivation/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingSAME*
strides
?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0k
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????0?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_2/Conv2DConv2Dactivation/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@k
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????@Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2conv2d_1/Relu:activations:0conv2d_3/Relu:activations:0conv2d_6/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:????????????
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_7/Conv2DConv2Dconcatenate/concat:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
add/addAddV2activation/Relu:activations:0conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? a
activation_1/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????? ?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_9/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? m
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
max_pooling2d/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:?????????	? *
ksize
*
paddingVALID*
strides
?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_8/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0k
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0m
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2max_pooling2d/MaxPool:output:0conv2d_8/Relu:activations:0conv2d_11/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????	??t
activation_2/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:?????????	???
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_13/Conv2DConv2Dactivation_2/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: (*
dtype0?
conv2d_14/Conv2DConv2Dconv2d_13/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(*
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(m
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?(?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_12/Conv2DConv2Dactivation_2/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:(0*
dtype0?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0m
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV2conv2d_12/Relu:activations:0conv2d_15/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????	?P?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:P?*
dtype0?
conv2d_16/Conv2DConv2Dconcatenate_2/concat:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	???
	add_1/addAddV2activation_2/Relu:activations:0conv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:?????????	??d
activation_3/ReluReluadd_1/add:z:0*
T0*1
_output_shapes
:?????????	???
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_21/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$m
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?$?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_19/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_17/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
max_pooling2d_1/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????J?*
ksize
*
paddingVALID*
strides
?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0*
paddingVALID*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0l
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J0?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$*
paddingVALID*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J$?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:$(*
dtype0?
conv2d_23/Conv2DConv2Dconv2d_22/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(*
paddingVALID*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J([
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_3/concatConcatV2 max_pooling2d_1/MaxPool:output:0conv2d_18/Relu:activations:0conv2d_20/Relu:activations:0conv2d_23/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????J?s
activation_4/ReluReluconcatenate_3/concat:output:0*
T0*0
_output_shapes
:?????????J??
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_25/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: %*
dtype0?
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%l
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J%?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_24/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:%**
dtype0?
conv2d_27/Conv2DConv2Dconv2d_26/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J**
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J*l
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J*[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_4/concatConcatV2conv2d_24/Relu:activations:0conv2d_27/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????JJ?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:J?*
dtype0?
conv2d_28/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?*
paddingSAME*
strides
?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J??
	add_2/addAddV2activation_4/Relu:activations:0conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:?????????J?c
activation_5/ReluReluadd_2/add:z:0*
T0*0
_output_shapes
:?????????J?r
!final_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
final_pool/MeanMeanactivation_5/Relu:activations:0*final_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????^
dropout_end/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_end/dropout/MulMulfinal_pool/Mean:output:0"dropout_end/dropout/Const:output:0*
T0*(
_output_shapes
:??????????a
dropout_end/dropout/ShapeShapefinal_pool/Mean:output:0*
T0*
_output_shapes
:?
0dropout_end/dropout/random_uniform/RandomUniformRandomUniform"dropout_end/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed}g
"dropout_end/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
 dropout_end/dropout/GreaterEqualGreaterEqual9dropout_end/dropout/random_uniform/RandomUniform:output:0+dropout_end/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_end/dropout/CastCast$dropout_end/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_end/dropout/Mul_1Muldropout_end/dropout/Mul:z:0dropout_end/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
predictions/MatMulMatMuldropout_end/dropout/Mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitypredictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
R
&__inference_add_1_layer_call_fn_269359
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_266313j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????	??:?????????	??:[ W
1
_output_shapes
:?????????	??
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????	??
"
_user_specified_name
inputs/1
?
?
*__inference_conv2d_20_layer_call_fn_269504

inputs!
unknown: $
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_269321

inputs8
conv2d_readvariableop_resource:(0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?(
 
_user_specified_nameinputs
?
?
*__inference_conv2d_25_layer_call_fn_269571

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
??
?8
!__inference__wrapped_model_265913
input_1K
1chemception_conv2d_conv2d_readvariableop_resource: @
2chemception_conv2d_biasadd_readvariableop_resource: M
3chemception_conv2d_4_conv2d_readvariableop_resource:  B
4chemception_conv2d_4_biasadd_readvariableop_resource: M
3chemception_conv2d_5_conv2d_readvariableop_resource: 0B
4chemception_conv2d_5_biasadd_readvariableop_resource:0M
3chemception_conv2d_2_conv2d_readvariableop_resource:  B
4chemception_conv2d_2_biasadd_readvariableop_resource: M
3chemception_conv2d_1_conv2d_readvariableop_resource:  B
4chemception_conv2d_1_biasadd_readvariableop_resource: M
3chemception_conv2d_3_conv2d_readvariableop_resource:  B
4chemception_conv2d_3_biasadd_readvariableop_resource: M
3chemception_conv2d_6_conv2d_readvariableop_resource:0@B
4chemception_conv2d_6_biasadd_readvariableop_resource:@N
3chemception_conv2d_7_conv2d_readvariableop_resource:? B
4chemception_conv2d_7_biasadd_readvariableop_resource: M
3chemception_conv2d_9_conv2d_readvariableop_resource:  B
4chemception_conv2d_9_biasadd_readvariableop_resource: N
4chemception_conv2d_10_conv2d_readvariableop_resource:  C
5chemception_conv2d_10_biasadd_readvariableop_resource: M
3chemception_conv2d_8_conv2d_readvariableop_resource: 0B
4chemception_conv2d_8_biasadd_readvariableop_resource:0N
4chemception_conv2d_11_conv2d_readvariableop_resource: 0C
5chemception_conv2d_11_biasadd_readvariableop_resource:0O
4chemception_conv2d_13_conv2d_readvariableop_resource:? C
5chemception_conv2d_13_biasadd_readvariableop_resource: N
4chemception_conv2d_14_conv2d_readvariableop_resource: (C
5chemception_conv2d_14_biasadd_readvariableop_resource:(O
4chemception_conv2d_12_conv2d_readvariableop_resource:? C
5chemception_conv2d_12_biasadd_readvariableop_resource: N
4chemception_conv2d_15_conv2d_readvariableop_resource:(0C
5chemception_conv2d_15_biasadd_readvariableop_resource:0O
4chemception_conv2d_16_conv2d_readvariableop_resource:P?D
5chemception_conv2d_16_biasadd_readvariableop_resource:	?O
4chemception_conv2d_21_conv2d_readvariableop_resource:? C
5chemception_conv2d_21_biasadd_readvariableop_resource: N
4chemception_conv2d_22_conv2d_readvariableop_resource: $C
5chemception_conv2d_22_biasadd_readvariableop_resource:$O
4chemception_conv2d_19_conv2d_readvariableop_resource:? C
5chemception_conv2d_19_biasadd_readvariableop_resource: O
4chemception_conv2d_17_conv2d_readvariableop_resource:? C
5chemception_conv2d_17_biasadd_readvariableop_resource: N
4chemception_conv2d_18_conv2d_readvariableop_resource: 0C
5chemception_conv2d_18_biasadd_readvariableop_resource:0N
4chemception_conv2d_20_conv2d_readvariableop_resource: $C
5chemception_conv2d_20_biasadd_readvariableop_resource:$N
4chemception_conv2d_23_conv2d_readvariableop_resource:$(C
5chemception_conv2d_23_biasadd_readvariableop_resource:(O
4chemception_conv2d_25_conv2d_readvariableop_resource:? C
5chemception_conv2d_25_biasadd_readvariableop_resource: N
4chemception_conv2d_26_conv2d_readvariableop_resource: %C
5chemception_conv2d_26_biasadd_readvariableop_resource:%O
4chemception_conv2d_24_conv2d_readvariableop_resource:? C
5chemception_conv2d_24_biasadd_readvariableop_resource: N
4chemception_conv2d_27_conv2d_readvariableop_resource:%*C
5chemception_conv2d_27_biasadd_readvariableop_resource:*O
4chemception_conv2d_28_conv2d_readvariableop_resource:J?D
5chemception_conv2d_28_biasadd_readvariableop_resource:	?I
6chemception_predictions_matmul_readvariableop_resource:	?E
7chemception_predictions_biasadd_readvariableop_resource:
identity??)Chemception/conv2d/BiasAdd/ReadVariableOp?(Chemception/conv2d/Conv2D/ReadVariableOp?+Chemception/conv2d_1/BiasAdd/ReadVariableOp?*Chemception/conv2d_1/Conv2D/ReadVariableOp?,Chemception/conv2d_10/BiasAdd/ReadVariableOp?+Chemception/conv2d_10/Conv2D/ReadVariableOp?,Chemception/conv2d_11/BiasAdd/ReadVariableOp?+Chemception/conv2d_11/Conv2D/ReadVariableOp?,Chemception/conv2d_12/BiasAdd/ReadVariableOp?+Chemception/conv2d_12/Conv2D/ReadVariableOp?,Chemception/conv2d_13/BiasAdd/ReadVariableOp?+Chemception/conv2d_13/Conv2D/ReadVariableOp?,Chemception/conv2d_14/BiasAdd/ReadVariableOp?+Chemception/conv2d_14/Conv2D/ReadVariableOp?,Chemception/conv2d_15/BiasAdd/ReadVariableOp?+Chemception/conv2d_15/Conv2D/ReadVariableOp?,Chemception/conv2d_16/BiasAdd/ReadVariableOp?+Chemception/conv2d_16/Conv2D/ReadVariableOp?,Chemception/conv2d_17/BiasAdd/ReadVariableOp?+Chemception/conv2d_17/Conv2D/ReadVariableOp?,Chemception/conv2d_18/BiasAdd/ReadVariableOp?+Chemception/conv2d_18/Conv2D/ReadVariableOp?,Chemception/conv2d_19/BiasAdd/ReadVariableOp?+Chemception/conv2d_19/Conv2D/ReadVariableOp?+Chemception/conv2d_2/BiasAdd/ReadVariableOp?*Chemception/conv2d_2/Conv2D/ReadVariableOp?,Chemception/conv2d_20/BiasAdd/ReadVariableOp?+Chemception/conv2d_20/Conv2D/ReadVariableOp?,Chemception/conv2d_21/BiasAdd/ReadVariableOp?+Chemception/conv2d_21/Conv2D/ReadVariableOp?,Chemception/conv2d_22/BiasAdd/ReadVariableOp?+Chemception/conv2d_22/Conv2D/ReadVariableOp?,Chemception/conv2d_23/BiasAdd/ReadVariableOp?+Chemception/conv2d_23/Conv2D/ReadVariableOp?,Chemception/conv2d_24/BiasAdd/ReadVariableOp?+Chemception/conv2d_24/Conv2D/ReadVariableOp?,Chemception/conv2d_25/BiasAdd/ReadVariableOp?+Chemception/conv2d_25/Conv2D/ReadVariableOp?,Chemception/conv2d_26/BiasAdd/ReadVariableOp?+Chemception/conv2d_26/Conv2D/ReadVariableOp?,Chemception/conv2d_27/BiasAdd/ReadVariableOp?+Chemception/conv2d_27/Conv2D/ReadVariableOp?,Chemception/conv2d_28/BiasAdd/ReadVariableOp?+Chemception/conv2d_28/Conv2D/ReadVariableOp?+Chemception/conv2d_3/BiasAdd/ReadVariableOp?*Chemception/conv2d_3/Conv2D/ReadVariableOp?+Chemception/conv2d_4/BiasAdd/ReadVariableOp?*Chemception/conv2d_4/Conv2D/ReadVariableOp?+Chemception/conv2d_5/BiasAdd/ReadVariableOp?*Chemception/conv2d_5/Conv2D/ReadVariableOp?+Chemception/conv2d_6/BiasAdd/ReadVariableOp?*Chemception/conv2d_6/Conv2D/ReadVariableOp?+Chemception/conv2d_7/BiasAdd/ReadVariableOp?*Chemception/conv2d_7/Conv2D/ReadVariableOp?+Chemception/conv2d_8/BiasAdd/ReadVariableOp?*Chemception/conv2d_8/Conv2D/ReadVariableOp?+Chemception/conv2d_9/BiasAdd/ReadVariableOp?*Chemception/conv2d_9/Conv2D/ReadVariableOp?.Chemception/predictions/BiasAdd/ReadVariableOp?-Chemception/predictions/MatMul/ReadVariableOp?
(Chemception/conv2d/Conv2D/ReadVariableOpReadVariableOp1chemception_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Chemception/conv2d/Conv2DConv2Dinput_10Chemception/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
)Chemception/conv2d/BiasAdd/ReadVariableOpReadVariableOp2chemception_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d/BiasAddBiasAdd"Chemception/conv2d/Conv2D:output:01Chemception/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/activation/ReluRelu#Chemception/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_4/Conv2DConv2D)Chemception/activation/Relu:activations:02Chemception/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_4/BiasAddBiasAdd$Chemception/conv2d_4/Conv2D:output:03Chemception/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_4/ReluRelu%Chemception/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Chemception/conv2d_5/Conv2DConv2D'Chemception/conv2d_4/Relu:activations:02Chemception/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingSAME*
strides
?
+Chemception/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
Chemception/conv2d_5/BiasAddBiasAdd$Chemception/conv2d_5/Conv2D:output:03Chemception/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0?
Chemception/conv2d_5/ReluRelu%Chemception/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????0?
*Chemception/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_2/Conv2DConv2D)Chemception/activation/Relu:activations:02Chemception/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_2/BiasAddBiasAdd$Chemception/conv2d_2/Conv2D:output:03Chemception/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_2/ReluRelu%Chemception/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_1/Conv2DConv2D)Chemception/activation/Relu:activations:02Chemception/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_1/BiasAddBiasAdd$Chemception/conv2d_1/Conv2D:output:03Chemception/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_1/ReluRelu%Chemception/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_3/Conv2DConv2D'Chemception/conv2d_2/Relu:activations:02Chemception/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_3/BiasAddBiasAdd$Chemception/conv2d_3/Conv2D:output:03Chemception/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_3/ReluRelu%Chemception/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Chemception/conv2d_6/Conv2DConv2D'Chemception/conv2d_5/Relu:activations:02Chemception/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
?
+Chemception/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Chemception/conv2d_6/BiasAddBiasAdd$Chemception/conv2d_6/Conv2D:output:03Chemception/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@?
Chemception/conv2d_6/ReluRelu%Chemception/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????@e
#Chemception/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
Chemception/concatenate/concatConcatV2'Chemception/conv2d_1/Relu:activations:0'Chemception/conv2d_3/Relu:activations:0'Chemception/conv2d_6/Relu:activations:0,Chemception/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:????????????
*Chemception/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_7/Conv2DConv2D'Chemception/concatenate/concat:output:02Chemception/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_7/BiasAddBiasAdd$Chemception/conv2d_7/Conv2D:output:03Chemception/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/add/addAddV2)Chemception/activation/Relu:activations:0%Chemception/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? y
Chemception/activation_1/ReluReluChemception/add/add:z:0*
T0*0
_output_shapes
:?????????? ?
*Chemception/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_9/Conv2DConv2D+Chemception/activation_1/Relu:activations:02Chemception/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
+Chemception/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_9/BiasAddBiasAdd$Chemception/conv2d_9/Conv2D:output:03Chemception/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_9/ReluRelu%Chemception/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
+Chemception/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Chemception/conv2d_10/Conv2DConv2D'Chemception/conv2d_9/Relu:activations:03Chemception/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
,Chemception/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_10/BiasAddBiasAdd%Chemception/conv2d_10/Conv2D:output:04Chemception/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
Chemception/conv2d_10/ReluRelu&Chemception/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
!Chemception/max_pooling2d/MaxPoolMaxPool+Chemception/activation_1/Relu:activations:0*0
_output_shapes
:?????????	? *
ksize
*
paddingVALID*
strides
?
*Chemception/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3chemception_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Chemception/conv2d_8/Conv2DConv2D+Chemception/activation_1/Relu:activations:02Chemception/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
+Chemception/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4chemception_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
Chemception/conv2d_8/BiasAddBiasAdd$Chemception/conv2d_8/Conv2D:output:03Chemception/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0?
Chemception/conv2d_8/ReluRelu%Chemception/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0?
+Chemception/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Chemception/conv2d_11/Conv2DConv2D(Chemception/conv2d_10/Relu:activations:03Chemception/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
,Chemception/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
Chemception/conv2d_11/BiasAddBiasAdd%Chemception/conv2d_11/Conv2D:output:04Chemception/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0?
Chemception/conv2d_11/ReluRelu&Chemception/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0g
%Chemception/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 Chemception/concatenate_1/concatConcatV2*Chemception/max_pooling2d/MaxPool:output:0'Chemception/conv2d_8/Relu:activations:0(Chemception/conv2d_11/Relu:activations:0.Chemception/concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????	???
Chemception/activation_2/ReluRelu)Chemception/concatenate_1/concat:output:0*
T0*1
_output_shapes
:?????????	???
+Chemception/conv2d_13/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_13/Conv2DConv2D+Chemception/activation_2/Relu:activations:03Chemception/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
,Chemception/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_13/BiasAddBiasAdd%Chemception/conv2d_13/Conv2D:output:04Chemception/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? ?
Chemception/conv2d_13/ReluRelu&Chemception/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
+Chemception/conv2d_14/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: (*
dtype0?
Chemception/conv2d_14/Conv2DConv2D(Chemception/conv2d_13/Relu:activations:03Chemception/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(*
paddingSAME*
strides
?
,Chemception/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
Chemception/conv2d_14/BiasAddBiasAdd%Chemception/conv2d_14/Conv2D:output:04Chemception/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(?
Chemception/conv2d_14/ReluRelu&Chemception/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?(?
+Chemception/conv2d_12/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_12/Conv2DConv2D+Chemception/activation_2/Relu:activations:03Chemception/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
,Chemception/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_12/BiasAddBiasAdd%Chemception/conv2d_12/Conv2D:output:04Chemception/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? ?
Chemception/conv2d_12/ReluRelu&Chemception/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
+Chemception/conv2d_15/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:(0*
dtype0?
Chemception/conv2d_15/Conv2DConv2D(Chemception/conv2d_14/Relu:activations:03Chemception/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingSAME*
strides
?
,Chemception/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
Chemception/conv2d_15/BiasAddBiasAdd%Chemception/conv2d_15/Conv2D:output:04Chemception/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0?
Chemception/conv2d_15/ReluRelu&Chemception/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0g
%Chemception/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 Chemception/concatenate_2/concatConcatV2(Chemception/conv2d_12/Relu:activations:0(Chemception/conv2d_15/Relu:activations:0.Chemception/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????	?P?
+Chemception/conv2d_16/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:P?*
dtype0?
Chemception/conv2d_16/Conv2DConv2D)Chemception/concatenate_2/concat:output:03Chemception/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??*
paddingSAME*
strides
?
,Chemception/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Chemception/conv2d_16/BiasAddBiasAdd%Chemception/conv2d_16/Conv2D:output:04Chemception/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	???
Chemception/add_1/addAddV2+Chemception/activation_2/Relu:activations:0&Chemception/conv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:?????????	??|
Chemception/activation_3/ReluReluChemception/add_1/add:z:0*
T0*1
_output_shapes
:?????????	???
+Chemception/conv2d_21/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_21/Conv2DConv2D+Chemception/activation_3/Relu:activations:03Chemception/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
,Chemception/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_21/BiasAddBiasAdd%Chemception/conv2d_21/Conv2D:output:04Chemception/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? ?
Chemception/conv2d_21/ReluRelu&Chemception/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
+Chemception/conv2d_22/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Chemception/conv2d_22/Conv2DConv2D(Chemception/conv2d_21/Relu:activations:03Chemception/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$*
paddingSAME*
strides
?
,Chemception/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
Chemception/conv2d_22/BiasAddBiasAdd%Chemception/conv2d_22/Conv2D:output:04Chemception/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$?
Chemception/conv2d_22/ReluRelu&Chemception/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?$?
+Chemception/conv2d_19/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_19/Conv2DConv2D+Chemception/activation_3/Relu:activations:03Chemception/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
,Chemception/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_19/BiasAddBiasAdd%Chemception/conv2d_19/Conv2D:output:04Chemception/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? ?
Chemception/conv2d_19/ReluRelu&Chemception/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
+Chemception/conv2d_17/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_17/Conv2DConv2D+Chemception/activation_3/Relu:activations:03Chemception/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
,Chemception/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_17/BiasAddBiasAdd%Chemception/conv2d_17/Conv2D:output:04Chemception/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? ?
Chemception/conv2d_17/ReluRelu&Chemception/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
#Chemception/max_pooling2d_1/MaxPoolMaxPool+Chemception/activation_3/Relu:activations:0*0
_output_shapes
:?????????J?*
ksize
*
paddingVALID*
strides
?
+Chemception/conv2d_18/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Chemception/conv2d_18/Conv2DConv2D(Chemception/conv2d_17/Relu:activations:03Chemception/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0*
paddingVALID*
strides
?
,Chemception/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
Chemception/conv2d_18/BiasAddBiasAdd%Chemception/conv2d_18/Conv2D:output:04Chemception/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0?
Chemception/conv2d_18/ReluRelu&Chemception/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J0?
+Chemception/conv2d_20/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Chemception/conv2d_20/Conv2DConv2D(Chemception/conv2d_19/Relu:activations:03Chemception/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$*
paddingVALID*
strides
?
,Chemception/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
Chemception/conv2d_20/BiasAddBiasAdd%Chemception/conv2d_20/Conv2D:output:04Chemception/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$?
Chemception/conv2d_20/ReluRelu&Chemception/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J$?
+Chemception/conv2d_23/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:$(*
dtype0?
Chemception/conv2d_23/Conv2DConv2D(Chemception/conv2d_22/Relu:activations:03Chemception/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(*
paddingVALID*
strides
?
,Chemception/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
Chemception/conv2d_23/BiasAddBiasAdd%Chemception/conv2d_23/Conv2D:output:04Chemception/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(?
Chemception/conv2d_23/ReluRelu&Chemception/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J(g
%Chemception/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 Chemception/concatenate_3/concatConcatV2,Chemception/max_pooling2d_1/MaxPool:output:0(Chemception/conv2d_18/Relu:activations:0(Chemception/conv2d_20/Relu:activations:0(Chemception/conv2d_23/Relu:activations:0.Chemception/concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????J??
Chemception/activation_4/ReluRelu)Chemception/concatenate_3/concat:output:0*
T0*0
_output_shapes
:?????????J??
+Chemception/conv2d_25/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_25/Conv2DConv2D+Chemception/activation_4/Relu:activations:03Chemception/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
,Chemception/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_25/BiasAddBiasAdd%Chemception/conv2d_25/Conv2D:output:04Chemception/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J ?
Chemception/conv2d_25/ReluRelu&Chemception/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
+Chemception/conv2d_26/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: %*
dtype0?
Chemception/conv2d_26/Conv2DConv2D(Chemception/conv2d_25/Relu:activations:03Chemception/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%*
paddingSAME*
strides
?
,Chemception/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0?
Chemception/conv2d_26/BiasAddBiasAdd%Chemception/conv2d_26/Conv2D:output:04Chemception/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%?
Chemception/conv2d_26/ReluRelu&Chemception/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J%?
+Chemception/conv2d_24/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Chemception/conv2d_24/Conv2DConv2D+Chemception/activation_4/Relu:activations:03Chemception/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
,Chemception/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Chemception/conv2d_24/BiasAddBiasAdd%Chemception/conv2d_24/Conv2D:output:04Chemception/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J ?
Chemception/conv2d_24/ReluRelu&Chemception/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
+Chemception/conv2d_27/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:%**
dtype0?
Chemception/conv2d_27/Conv2DConv2D(Chemception/conv2d_26/Relu:activations:03Chemception/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J**
paddingSAME*
strides
?
,Chemception/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0?
Chemception/conv2d_27/BiasAddBiasAdd%Chemception/conv2d_27/Conv2D:output:04Chemception/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J*?
Chemception/conv2d_27/ReluRelu&Chemception/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J*g
%Chemception/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 Chemception/concatenate_4/concatConcatV2(Chemception/conv2d_24/Relu:activations:0(Chemception/conv2d_27/Relu:activations:0.Chemception/concatenate_4/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????JJ?
+Chemception/conv2d_28/Conv2D/ReadVariableOpReadVariableOp4chemception_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:J?*
dtype0?
Chemception/conv2d_28/Conv2DConv2D)Chemception/concatenate_4/concat:output:03Chemception/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?*
paddingSAME*
strides
?
,Chemception/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp5chemception_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Chemception/conv2d_28/BiasAddBiasAdd%Chemception/conv2d_28/Conv2D:output:04Chemception/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J??
Chemception/add_2/addAddV2+Chemception/activation_4/Relu:activations:0&Chemception/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:?????????J?{
Chemception/activation_5/ReluReluChemception/add_2/add:z:0*
T0*0
_output_shapes
:?????????J?~
-Chemception/final_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
Chemception/final_pool/MeanMean+Chemception/activation_5/Relu:activations:06Chemception/final_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
 Chemception/dropout_end/IdentityIdentity$Chemception/final_pool/Mean:output:0*
T0*(
_output_shapes
:???????????
-Chemception/predictions/MatMul/ReadVariableOpReadVariableOp6chemception_predictions_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Chemception/predictions/MatMulMatMul)Chemception/dropout_end/Identity:output:05Chemception/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.Chemception/predictions/BiasAdd/ReadVariableOpReadVariableOp7chemception_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Chemception/predictions/BiasAddBiasAdd(Chemception/predictions/MatMul:product:06Chemception/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Chemception/predictions/SoftmaxSoftmax(Chemception/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
IdentityIdentity)Chemception/predictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp*^Chemception/conv2d/BiasAdd/ReadVariableOp)^Chemception/conv2d/Conv2D/ReadVariableOp,^Chemception/conv2d_1/BiasAdd/ReadVariableOp+^Chemception/conv2d_1/Conv2D/ReadVariableOp-^Chemception/conv2d_10/BiasAdd/ReadVariableOp,^Chemception/conv2d_10/Conv2D/ReadVariableOp-^Chemception/conv2d_11/BiasAdd/ReadVariableOp,^Chemception/conv2d_11/Conv2D/ReadVariableOp-^Chemception/conv2d_12/BiasAdd/ReadVariableOp,^Chemception/conv2d_12/Conv2D/ReadVariableOp-^Chemception/conv2d_13/BiasAdd/ReadVariableOp,^Chemception/conv2d_13/Conv2D/ReadVariableOp-^Chemception/conv2d_14/BiasAdd/ReadVariableOp,^Chemception/conv2d_14/Conv2D/ReadVariableOp-^Chemception/conv2d_15/BiasAdd/ReadVariableOp,^Chemception/conv2d_15/Conv2D/ReadVariableOp-^Chemception/conv2d_16/BiasAdd/ReadVariableOp,^Chemception/conv2d_16/Conv2D/ReadVariableOp-^Chemception/conv2d_17/BiasAdd/ReadVariableOp,^Chemception/conv2d_17/Conv2D/ReadVariableOp-^Chemception/conv2d_18/BiasAdd/ReadVariableOp,^Chemception/conv2d_18/Conv2D/ReadVariableOp-^Chemception/conv2d_19/BiasAdd/ReadVariableOp,^Chemception/conv2d_19/Conv2D/ReadVariableOp,^Chemception/conv2d_2/BiasAdd/ReadVariableOp+^Chemception/conv2d_2/Conv2D/ReadVariableOp-^Chemception/conv2d_20/BiasAdd/ReadVariableOp,^Chemception/conv2d_20/Conv2D/ReadVariableOp-^Chemception/conv2d_21/BiasAdd/ReadVariableOp,^Chemception/conv2d_21/Conv2D/ReadVariableOp-^Chemception/conv2d_22/BiasAdd/ReadVariableOp,^Chemception/conv2d_22/Conv2D/ReadVariableOp-^Chemception/conv2d_23/BiasAdd/ReadVariableOp,^Chemception/conv2d_23/Conv2D/ReadVariableOp-^Chemception/conv2d_24/BiasAdd/ReadVariableOp,^Chemception/conv2d_24/Conv2D/ReadVariableOp-^Chemception/conv2d_25/BiasAdd/ReadVariableOp,^Chemception/conv2d_25/Conv2D/ReadVariableOp-^Chemception/conv2d_26/BiasAdd/ReadVariableOp,^Chemception/conv2d_26/Conv2D/ReadVariableOp-^Chemception/conv2d_27/BiasAdd/ReadVariableOp,^Chemception/conv2d_27/Conv2D/ReadVariableOp-^Chemception/conv2d_28/BiasAdd/ReadVariableOp,^Chemception/conv2d_28/Conv2D/ReadVariableOp,^Chemception/conv2d_3/BiasAdd/ReadVariableOp+^Chemception/conv2d_3/Conv2D/ReadVariableOp,^Chemception/conv2d_4/BiasAdd/ReadVariableOp+^Chemception/conv2d_4/Conv2D/ReadVariableOp,^Chemception/conv2d_5/BiasAdd/ReadVariableOp+^Chemception/conv2d_5/Conv2D/ReadVariableOp,^Chemception/conv2d_6/BiasAdd/ReadVariableOp+^Chemception/conv2d_6/Conv2D/ReadVariableOp,^Chemception/conv2d_7/BiasAdd/ReadVariableOp+^Chemception/conv2d_7/Conv2D/ReadVariableOp,^Chemception/conv2d_8/BiasAdd/ReadVariableOp+^Chemception/conv2d_8/Conv2D/ReadVariableOp,^Chemception/conv2d_9/BiasAdd/ReadVariableOp+^Chemception/conv2d_9/Conv2D/ReadVariableOp/^Chemception/predictions/BiasAdd/ReadVariableOp.^Chemception/predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)Chemception/conv2d/BiasAdd/ReadVariableOp)Chemception/conv2d/BiasAdd/ReadVariableOp2T
(Chemception/conv2d/Conv2D/ReadVariableOp(Chemception/conv2d/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_1/BiasAdd/ReadVariableOp+Chemception/conv2d_1/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_1/Conv2D/ReadVariableOp*Chemception/conv2d_1/Conv2D/ReadVariableOp2\
,Chemception/conv2d_10/BiasAdd/ReadVariableOp,Chemception/conv2d_10/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_10/Conv2D/ReadVariableOp+Chemception/conv2d_10/Conv2D/ReadVariableOp2\
,Chemception/conv2d_11/BiasAdd/ReadVariableOp,Chemception/conv2d_11/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_11/Conv2D/ReadVariableOp+Chemception/conv2d_11/Conv2D/ReadVariableOp2\
,Chemception/conv2d_12/BiasAdd/ReadVariableOp,Chemception/conv2d_12/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_12/Conv2D/ReadVariableOp+Chemception/conv2d_12/Conv2D/ReadVariableOp2\
,Chemception/conv2d_13/BiasAdd/ReadVariableOp,Chemception/conv2d_13/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_13/Conv2D/ReadVariableOp+Chemception/conv2d_13/Conv2D/ReadVariableOp2\
,Chemception/conv2d_14/BiasAdd/ReadVariableOp,Chemception/conv2d_14/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_14/Conv2D/ReadVariableOp+Chemception/conv2d_14/Conv2D/ReadVariableOp2\
,Chemception/conv2d_15/BiasAdd/ReadVariableOp,Chemception/conv2d_15/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_15/Conv2D/ReadVariableOp+Chemception/conv2d_15/Conv2D/ReadVariableOp2\
,Chemception/conv2d_16/BiasAdd/ReadVariableOp,Chemception/conv2d_16/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_16/Conv2D/ReadVariableOp+Chemception/conv2d_16/Conv2D/ReadVariableOp2\
,Chemception/conv2d_17/BiasAdd/ReadVariableOp,Chemception/conv2d_17/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_17/Conv2D/ReadVariableOp+Chemception/conv2d_17/Conv2D/ReadVariableOp2\
,Chemception/conv2d_18/BiasAdd/ReadVariableOp,Chemception/conv2d_18/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_18/Conv2D/ReadVariableOp+Chemception/conv2d_18/Conv2D/ReadVariableOp2\
,Chemception/conv2d_19/BiasAdd/ReadVariableOp,Chemception/conv2d_19/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_19/Conv2D/ReadVariableOp+Chemception/conv2d_19/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_2/BiasAdd/ReadVariableOp+Chemception/conv2d_2/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_2/Conv2D/ReadVariableOp*Chemception/conv2d_2/Conv2D/ReadVariableOp2\
,Chemception/conv2d_20/BiasAdd/ReadVariableOp,Chemception/conv2d_20/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_20/Conv2D/ReadVariableOp+Chemception/conv2d_20/Conv2D/ReadVariableOp2\
,Chemception/conv2d_21/BiasAdd/ReadVariableOp,Chemception/conv2d_21/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_21/Conv2D/ReadVariableOp+Chemception/conv2d_21/Conv2D/ReadVariableOp2\
,Chemception/conv2d_22/BiasAdd/ReadVariableOp,Chemception/conv2d_22/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_22/Conv2D/ReadVariableOp+Chemception/conv2d_22/Conv2D/ReadVariableOp2\
,Chemception/conv2d_23/BiasAdd/ReadVariableOp,Chemception/conv2d_23/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_23/Conv2D/ReadVariableOp+Chemception/conv2d_23/Conv2D/ReadVariableOp2\
,Chemception/conv2d_24/BiasAdd/ReadVariableOp,Chemception/conv2d_24/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_24/Conv2D/ReadVariableOp+Chemception/conv2d_24/Conv2D/ReadVariableOp2\
,Chemception/conv2d_25/BiasAdd/ReadVariableOp,Chemception/conv2d_25/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_25/Conv2D/ReadVariableOp+Chemception/conv2d_25/Conv2D/ReadVariableOp2\
,Chemception/conv2d_26/BiasAdd/ReadVariableOp,Chemception/conv2d_26/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_26/Conv2D/ReadVariableOp+Chemception/conv2d_26/Conv2D/ReadVariableOp2\
,Chemception/conv2d_27/BiasAdd/ReadVariableOp,Chemception/conv2d_27/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_27/Conv2D/ReadVariableOp+Chemception/conv2d_27/Conv2D/ReadVariableOp2\
,Chemception/conv2d_28/BiasAdd/ReadVariableOp,Chemception/conv2d_28/BiasAdd/ReadVariableOp2Z
+Chemception/conv2d_28/Conv2D/ReadVariableOp+Chemception/conv2d_28/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_3/BiasAdd/ReadVariableOp+Chemception/conv2d_3/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_3/Conv2D/ReadVariableOp*Chemception/conv2d_3/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_4/BiasAdd/ReadVariableOp+Chemception/conv2d_4/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_4/Conv2D/ReadVariableOp*Chemception/conv2d_4/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_5/BiasAdd/ReadVariableOp+Chemception/conv2d_5/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_5/Conv2D/ReadVariableOp*Chemception/conv2d_5/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_6/BiasAdd/ReadVariableOp+Chemception/conv2d_6/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_6/Conv2D/ReadVariableOp*Chemception/conv2d_6/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_7/BiasAdd/ReadVariableOp+Chemception/conv2d_7/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_7/Conv2D/ReadVariableOp*Chemception/conv2d_7/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_8/BiasAdd/ReadVariableOp+Chemception/conv2d_8/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_8/Conv2D/ReadVariableOp*Chemception/conv2d_8/Conv2D/ReadVariableOp2Z
+Chemception/conv2d_9/BiasAdd/ReadVariableOp+Chemception/conv2d_9/BiasAdd/ReadVariableOp2X
*Chemception/conv2d_9/Conv2D/ReadVariableOp*Chemception/conv2d_9/Conv2D/ReadVariableOp2`
.Chemception/predictions/BiasAdd/ReadVariableOp.Chemception/predictions/BiasAdd/ReadVariableOp2^
-Chemception/predictions/MatMul/ReadVariableOp-Chemception/predictions/MatMul/ReadVariableOp:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
??
?-
G__inference_Chemception_layer_call_and_return_conditional_losses_268670

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource: 06
(conv2d_5_biasadd_readvariableop_resource:0A
'conv2d_2_conv2d_readvariableop_resource:  6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource:0@6
(conv2d_6_biasadd_readvariableop_resource:@B
'conv2d_7_conv2d_readvariableop_resource:? 6
(conv2d_7_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource:  6
(conv2d_9_biasadd_readvariableop_resource: B
(conv2d_10_conv2d_readvariableop_resource:  7
)conv2d_10_biasadd_readvariableop_resource: A
'conv2d_8_conv2d_readvariableop_resource: 06
(conv2d_8_biasadd_readvariableop_resource:0B
(conv2d_11_conv2d_readvariableop_resource: 07
)conv2d_11_biasadd_readvariableop_resource:0C
(conv2d_13_conv2d_readvariableop_resource:? 7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: (7
)conv2d_14_biasadd_readvariableop_resource:(C
(conv2d_12_conv2d_readvariableop_resource:? 7
)conv2d_12_biasadd_readvariableop_resource: B
(conv2d_15_conv2d_readvariableop_resource:(07
)conv2d_15_biasadd_readvariableop_resource:0C
(conv2d_16_conv2d_readvariableop_resource:P?8
)conv2d_16_biasadd_readvariableop_resource:	?C
(conv2d_21_conv2d_readvariableop_resource:? 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource: $7
)conv2d_22_biasadd_readvariableop_resource:$C
(conv2d_19_conv2d_readvariableop_resource:? 7
)conv2d_19_biasadd_readvariableop_resource: C
(conv2d_17_conv2d_readvariableop_resource:? 7
)conv2d_17_biasadd_readvariableop_resource: B
(conv2d_18_conv2d_readvariableop_resource: 07
)conv2d_18_biasadd_readvariableop_resource:0B
(conv2d_20_conv2d_readvariableop_resource: $7
)conv2d_20_biasadd_readvariableop_resource:$B
(conv2d_23_conv2d_readvariableop_resource:$(7
)conv2d_23_biasadd_readvariableop_resource:(C
(conv2d_25_conv2d_readvariableop_resource:? 7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: %7
)conv2d_26_biasadd_readvariableop_resource:%C
(conv2d_24_conv2d_readvariableop_resource:? 7
)conv2d_24_biasadd_readvariableop_resource: B
(conv2d_27_conv2d_readvariableop_resource:%*7
)conv2d_27_biasadd_readvariableop_resource:*C
(conv2d_28_conv2d_readvariableop_resource:J?8
)conv2d_28_biasadd_readvariableop_resource:	?=
*predictions_matmul_readvariableop_resource:	?9
+predictions_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?"predictions/BiasAdd/ReadVariableOp?!predictions/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2Dactivation/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingSAME*
strides
?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0k
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????0?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_2/Conv2DConv2Dactivation/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@k
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:??????????@Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2conv2d_1/Relu:activations:0conv2d_3/Relu:activations:0conv2d_6/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:????????????
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_7/Conv2DConv2Dconcatenate/concat:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? ?
add/addAddV2activation/Relu:activations:0conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? a
activation_1/ReluReluadd/add:z:0*
T0*0
_output_shapes
:?????????? ?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_9/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? k
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? m
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? ?
max_pooling2d/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:?????????	? *
ksize
*
paddingVALID*
strides
?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_8/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0k
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0m
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV2max_pooling2d/MaxPool:output:0conv2d_8/Relu:activations:0conv2d_11/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????	??t
activation_2/ReluReluconcatenate_1/concat:output:0*
T0*1
_output_shapes
:?????????	???
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_13/Conv2DConv2Dactivation_2/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: (*
dtype0?
conv2d_14/Conv2DConv2Dconv2d_13/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(*
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?(m
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?(?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_12/Conv2DConv2Dactivation_2/Relu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:(0*
dtype0?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0m
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV2conv2d_12/Relu:activations:0conv2d_15/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????	?P?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:P?*
dtype0?
conv2d_16/Conv2DConv2Dconcatenate_2/concat:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	???
	add_1/addAddV2activation_2/Relu:activations:0conv2d_16/BiasAdd:output:0*
T0*1
_output_shapes
:?????????	??d
activation_3/ReluReluadd_1/add:z:0*
T0*1
_output_shapes
:?????????	???
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_21/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$*
paddingSAME*
strides
?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?$m
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	?$?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_19/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_17/Conv2DConv2Dactivation_3/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? m
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	? ?
max_pooling2d_1/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????J?*
ksize
*
paddingVALID*
strides
?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0*
paddingVALID*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0l
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J0?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$*
paddingVALID*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$l
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J$?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:$(*
dtype0?
conv2d_23/Conv2DConv2Dconv2d_22/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(*
paddingVALID*
strides
?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J([
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_3/concatConcatV2 max_pooling2d_1/MaxPool:output:0conv2d_18/Relu:activations:0conv2d_20/Relu:activations:0conv2d_23/Relu:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????J?s
activation_4/ReluReluconcatenate_3/concat:output:0*
T0*0
_output_shapes
:?????????J??
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_25/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: %*
dtype0?
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%*
paddingSAME*
strides
?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype0?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%l
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J%?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
conv2d_24/Conv2DConv2Dactivation_4/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J ?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:%**
dtype0?
conv2d_27/Conv2DConv2Dconv2d_26/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J**
paddingSAME*
strides
?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:**
dtype0?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J*l
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????J*[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_4/concatConcatV2conv2d_24/Relu:activations:0conv2d_27/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????JJ?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
:J?*
dtype0?
conv2d_28/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?*
paddingSAME*
strides
?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J??
	add_2/addAddV2activation_4/Relu:activations:0conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:?????????J?c
activation_5/ReluReluadd_2/add:z:0*
T0*0
_output_shapes
:?????????J?r
!final_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
final_pool/MeanMeanactivation_5/Relu:activations:0*final_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????m
dropout_end/IdentityIdentityfinal_pool/Mean:output:0*
T0*(
_output_shapes
:???????????
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
predictions/MatMulMatMuldropout_end/Identity:output:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitypredictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_269261

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_269216

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_266212

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:?????????	??d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_269375

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:?????????	??d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276

inputs8
conv2d_readvariableop_resource:(0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?(
 
_user_specified_nameinputs
?
?
,__inference_Chemception_layer_call_fn_268311

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11:0@

unknown_12:@%

unknown_13:? 

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: $

unknown_19: 0

unknown_20:0$

unknown_21: 0

unknown_22:0%

unknown_23:? 

unknown_24: $

unknown_25: (

unknown_26:(%

unknown_27:? 

unknown_28: $

unknown_29:(0

unknown_30:0%

unknown_31:P?

unknown_32:	?%

unknown_33:? 

unknown_34: $

unknown_35: $

unknown_36:$%

unknown_37:? 

unknown_38: %

unknown_39:? 

unknown_40: $

unknown_41: 0

unknown_42:0$

unknown_43: $

unknown_44:$$

unknown_45:$(

unknown_46:(%

unknown_47:? 

unknown_48: $

unknown_49: %

unknown_50:%%

unknown_51:? 

unknown_52: $

unknown_53:%*

unknown_54:*%

unknown_55:J?

unknown_56:	?

unknown_57:	?

unknown_58:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Chemception_layer_call_and_return_conditional_losses_266605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?

?
G__inference_predictions_layer_call_and_return_conditional_losses_269765

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
.__inference_concatenate_1_layer_call_fn_269223
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????	? :?????????	?0:?????????	?0:Z V
0
_output_shapes
:?????????	? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/2
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_265967

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????&?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
??
?
G__inference_Chemception_layer_call_and_return_conditional_losses_267461

inputs'
conv2d_267292: 
conv2d_267294: )
conv2d_4_267298:  
conv2d_4_267300: )
conv2d_5_267303: 0
conv2d_5_267305:0)
conv2d_2_267308:  
conv2d_2_267310: )
conv2d_1_267313:  
conv2d_1_267315: )
conv2d_3_267318:  
conv2d_3_267320: )
conv2d_6_267323:0@
conv2d_6_267325:@*
conv2d_7_267329:? 
conv2d_7_267331: )
conv2d_9_267336:  
conv2d_9_267338: *
conv2d_10_267341:  
conv2d_10_267343: )
conv2d_8_267347: 0
conv2d_8_267349:0*
conv2d_11_267352: 0
conv2d_11_267354:0+
conv2d_13_267359:? 
conv2d_13_267361: *
conv2d_14_267364: (
conv2d_14_267366:(+
conv2d_12_267369:? 
conv2d_12_267371: *
conv2d_15_267374:(0
conv2d_15_267376:0+
conv2d_16_267380:P?
conv2d_16_267382:	?+
conv2d_21_267387:? 
conv2d_21_267389: *
conv2d_22_267392: $
conv2d_22_267394:$+
conv2d_19_267397:? 
conv2d_19_267399: +
conv2d_17_267402:? 
conv2d_17_267404: *
conv2d_18_267408: 0
conv2d_18_267410:0*
conv2d_20_267413: $
conv2d_20_267415:$*
conv2d_23_267418:$(
conv2d_23_267420:(+
conv2d_25_267425:? 
conv2d_25_267427: *
conv2d_26_267430: %
conv2d_26_267432:%+
conv2d_24_267435:? 
conv2d_24_267437: *
conv2d_27_267440:%*
conv2d_27_267442:*+
conv2d_28_267446:J?
conv2d_28_267448:	?%
predictions_267455:	? 
predictions_267457:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?#dropout_end/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_267292conv2d_267294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_265967?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_265978?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_4_267298conv2d_4_267300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_267303conv2d_5_267305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_2_267308conv2d_2_267310*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_267313conv2d_1_267315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_267318conv2d_3_267320*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_267323conv2d_6_267325*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076?
concatenate/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_266090?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_7_267329conv2d_7_267331*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102?
add/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_266114?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_266121?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_9_267336conv2d_9_267338*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_267341conv2d_10_267343*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151?
max_pooling2d/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_8_267347conv2d_8_267349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_267352conv2d_11_267354*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191?
concatenate_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_8/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205?
activation_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_266212?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_13_267359conv2d_13_267361*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_267364conv2d_14_267366*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_12_267369conv2d_12_267371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_267374conv2d_15_267376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276?
concatenate_2/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_16_267380conv2d_16_267382*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301?
add_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_266313?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_266320?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_21_267387conv2d_21_267389*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_267392conv2d_22_267394*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_19_267397conv2d_19_267399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_17_267402conv2d_17_267404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_267408conv2d_18_267410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_267413conv2d_20_267415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_267418conv2d_23_267420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441?
concatenate_3/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456?
activation_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_266463?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_25_267425conv2d_25_267427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_267430conv2d_26_267432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_24_267435conv2d_24_267437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_267440conv2d_27_267442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527?
concatenate_4/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????JJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_28_267446conv2d_28_267448*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552?
add_2/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_266564?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_266571?
final_pool/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_266578?
#dropout_end/StatefulPartitionedCallStatefulPartitionedCall#final_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266758?
#predictions/StatefulPartitionedCallStatefulPartitionedCall,dropout_end/StatefulPartitionedCall:output:0predictions_267455predictions_267457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_predictions_layer_call_and_return_conditional_losses_266598{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall$^dropout_end/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2J
#dropout_end/StatefulPartitionedCall#dropout_end/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
m
A__inference_add_2_layer_call_and_return_conditional_losses_269686
inputs_0
inputs_1
identity[
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????J?X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????J?:?????????J?:Z V
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/1
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_269655
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????JJ_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????JJ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????J :?????????J*:Y U
/
_output_shapes
:?????????J 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????J*
"
_user_specified_name
inputs/1
??
?T
"__inference__traced_restore_270572
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: <
"assignvariableop_2_conv2d_4_kernel:  .
 assignvariableop_3_conv2d_4_bias: <
"assignvariableop_4_conv2d_2_kernel:  .
 assignvariableop_5_conv2d_2_bias: <
"assignvariableop_6_conv2d_5_kernel: 0.
 assignvariableop_7_conv2d_5_bias:0<
"assignvariableop_8_conv2d_1_kernel:  .
 assignvariableop_9_conv2d_1_bias: =
#assignvariableop_10_conv2d_3_kernel:  /
!assignvariableop_11_conv2d_3_bias: =
#assignvariableop_12_conv2d_6_kernel:0@/
!assignvariableop_13_conv2d_6_bias:@>
#assignvariableop_14_conv2d_7_kernel:? /
!assignvariableop_15_conv2d_7_bias: =
#assignvariableop_16_conv2d_9_kernel:  /
!assignvariableop_17_conv2d_9_bias: >
$assignvariableop_18_conv2d_10_kernel:  0
"assignvariableop_19_conv2d_10_bias: =
#assignvariableop_20_conv2d_8_kernel: 0/
!assignvariableop_21_conv2d_8_bias:0>
$assignvariableop_22_conv2d_11_kernel: 00
"assignvariableop_23_conv2d_11_bias:0?
$assignvariableop_24_conv2d_13_kernel:? 0
"assignvariableop_25_conv2d_13_bias: >
$assignvariableop_26_conv2d_14_kernel: (0
"assignvariableop_27_conv2d_14_bias:(?
$assignvariableop_28_conv2d_12_kernel:? 0
"assignvariableop_29_conv2d_12_bias: >
$assignvariableop_30_conv2d_15_kernel:(00
"assignvariableop_31_conv2d_15_bias:0?
$assignvariableop_32_conv2d_16_kernel:P?1
"assignvariableop_33_conv2d_16_bias:	??
$assignvariableop_34_conv2d_21_kernel:? 0
"assignvariableop_35_conv2d_21_bias: ?
$assignvariableop_36_conv2d_17_kernel:? 0
"assignvariableop_37_conv2d_17_bias: ?
$assignvariableop_38_conv2d_19_kernel:? 0
"assignvariableop_39_conv2d_19_bias: >
$assignvariableop_40_conv2d_22_kernel: $0
"assignvariableop_41_conv2d_22_bias:$>
$assignvariableop_42_conv2d_18_kernel: 00
"assignvariableop_43_conv2d_18_bias:0>
$assignvariableop_44_conv2d_20_kernel: $0
"assignvariableop_45_conv2d_20_bias:$>
$assignvariableop_46_conv2d_23_kernel:$(0
"assignvariableop_47_conv2d_23_bias:(?
$assignvariableop_48_conv2d_25_kernel:? 0
"assignvariableop_49_conv2d_25_bias: >
$assignvariableop_50_conv2d_26_kernel: %0
"assignvariableop_51_conv2d_26_bias:%?
$assignvariableop_52_conv2d_24_kernel:? 0
"assignvariableop_53_conv2d_24_bias: >
$assignvariableop_54_conv2d_27_kernel:%*0
"assignvariableop_55_conv2d_27_bias:*?
$assignvariableop_56_conv2d_28_kernel:J?1
"assignvariableop_57_conv2d_28_bias:	?9
&assignvariableop_58_predictions_kernel:	?2
$assignvariableop_59_predictions_bias:*
 assignvariableop_60_rmsprop_iter:	 +
!assignvariableop_61_rmsprop_decay: 3
)assignvariableop_62_rmsprop_learning_rate: .
$assignvariableop_63_rmsprop_momentum: )
assignvariableop_64_rmsprop_rho: #
assignvariableop_65_total: #
assignvariableop_66_count: %
assignvariableop_67_total_1: %
assignvariableop_68_count_1: G
-assignvariableop_69_rmsprop_conv2d_kernel_rms: 9
+assignvariableop_70_rmsprop_conv2d_bias_rms: I
/assignvariableop_71_rmsprop_conv2d_4_kernel_rms:  ;
-assignvariableop_72_rmsprop_conv2d_4_bias_rms: I
/assignvariableop_73_rmsprop_conv2d_2_kernel_rms:  ;
-assignvariableop_74_rmsprop_conv2d_2_bias_rms: I
/assignvariableop_75_rmsprop_conv2d_5_kernel_rms: 0;
-assignvariableop_76_rmsprop_conv2d_5_bias_rms:0I
/assignvariableop_77_rmsprop_conv2d_1_kernel_rms:  ;
-assignvariableop_78_rmsprop_conv2d_1_bias_rms: I
/assignvariableop_79_rmsprop_conv2d_3_kernel_rms:  ;
-assignvariableop_80_rmsprop_conv2d_3_bias_rms: I
/assignvariableop_81_rmsprop_conv2d_6_kernel_rms:0@;
-assignvariableop_82_rmsprop_conv2d_6_bias_rms:@J
/assignvariableop_83_rmsprop_conv2d_7_kernel_rms:? ;
-assignvariableop_84_rmsprop_conv2d_7_bias_rms: I
/assignvariableop_85_rmsprop_conv2d_9_kernel_rms:  ;
-assignvariableop_86_rmsprop_conv2d_9_bias_rms: J
0assignvariableop_87_rmsprop_conv2d_10_kernel_rms:  <
.assignvariableop_88_rmsprop_conv2d_10_bias_rms: I
/assignvariableop_89_rmsprop_conv2d_8_kernel_rms: 0;
-assignvariableop_90_rmsprop_conv2d_8_bias_rms:0J
0assignvariableop_91_rmsprop_conv2d_11_kernel_rms: 0<
.assignvariableop_92_rmsprop_conv2d_11_bias_rms:0K
0assignvariableop_93_rmsprop_conv2d_13_kernel_rms:? <
.assignvariableop_94_rmsprop_conv2d_13_bias_rms: J
0assignvariableop_95_rmsprop_conv2d_14_kernel_rms: (<
.assignvariableop_96_rmsprop_conv2d_14_bias_rms:(K
0assignvariableop_97_rmsprop_conv2d_12_kernel_rms:? <
.assignvariableop_98_rmsprop_conv2d_12_bias_rms: J
0assignvariableop_99_rmsprop_conv2d_15_kernel_rms:(0=
/assignvariableop_100_rmsprop_conv2d_15_bias_rms:0L
1assignvariableop_101_rmsprop_conv2d_16_kernel_rms:P?>
/assignvariableop_102_rmsprop_conv2d_16_bias_rms:	?L
1assignvariableop_103_rmsprop_conv2d_21_kernel_rms:? =
/assignvariableop_104_rmsprop_conv2d_21_bias_rms: L
1assignvariableop_105_rmsprop_conv2d_17_kernel_rms:? =
/assignvariableop_106_rmsprop_conv2d_17_bias_rms: L
1assignvariableop_107_rmsprop_conv2d_19_kernel_rms:? =
/assignvariableop_108_rmsprop_conv2d_19_bias_rms: K
1assignvariableop_109_rmsprop_conv2d_22_kernel_rms: $=
/assignvariableop_110_rmsprop_conv2d_22_bias_rms:$K
1assignvariableop_111_rmsprop_conv2d_18_kernel_rms: 0=
/assignvariableop_112_rmsprop_conv2d_18_bias_rms:0K
1assignvariableop_113_rmsprop_conv2d_20_kernel_rms: $=
/assignvariableop_114_rmsprop_conv2d_20_bias_rms:$K
1assignvariableop_115_rmsprop_conv2d_23_kernel_rms:$(=
/assignvariableop_116_rmsprop_conv2d_23_bias_rms:(L
1assignvariableop_117_rmsprop_conv2d_25_kernel_rms:? =
/assignvariableop_118_rmsprop_conv2d_25_bias_rms: K
1assignvariableop_119_rmsprop_conv2d_26_kernel_rms: %=
/assignvariableop_120_rmsprop_conv2d_26_bias_rms:%L
1assignvariableop_121_rmsprop_conv2d_24_kernel_rms:? =
/assignvariableop_122_rmsprop_conv2d_24_bias_rms: K
1assignvariableop_123_rmsprop_conv2d_27_kernel_rms:%*=
/assignvariableop_124_rmsprop_conv2d_27_bias_rms:*L
1assignvariableop_125_rmsprop_conv2d_28_kernel_rms:J?>
/assignvariableop_126_rmsprop_conv2d_28_bias_rms:	?F
3assignvariableop_127_rmsprop_predictions_kernel_rms:	??
1assignvariableop_128_rmsprop_predictions_bias_rms:
identity_130??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?F
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?F
value?FB?F?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-27/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-27/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-28/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-29/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-29/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-23/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-23/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-24/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-24/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-25/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-25/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-26/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-26/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-27/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-27/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-28/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-28/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-29/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-29/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_9_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_10_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_10_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_8_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_8_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_13_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_14_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_14_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_12_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_12_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_21_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_21_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_17_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_17_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_19_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_19_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv2d_22_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv2d_22_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp$assignvariableop_42_conv2d_18_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv2d_18_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp$assignvariableop_44_conv2d_20_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv2d_20_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp$assignvariableop_46_conv2d_23_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp"assignvariableop_47_conv2d_23_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_25_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv2d_25_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp$assignvariableop_50_conv2d_26_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp"assignvariableop_51_conv2d_26_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp$assignvariableop_52_conv2d_24_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp"assignvariableop_53_conv2d_24_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp$assignvariableop_54_conv2d_27_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp"assignvariableop_55_conv2d_27_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_conv2d_28_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp"assignvariableop_57_conv2d_28_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp&assignvariableop_58_predictions_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp$assignvariableop_59_predictions_biasIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp assignvariableop_60_rmsprop_iterIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp!assignvariableop_61_rmsprop_decayIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_rmsprop_learning_rateIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp$assignvariableop_63_rmsprop_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpassignvariableop_64_rmsprop_rhoIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpassignvariableop_65_totalIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpassignvariableop_66_countIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpassignvariableop_67_total_1Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpassignvariableop_68_count_1Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp-assignvariableop_69_rmsprop_conv2d_kernel_rmsIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp+assignvariableop_70_rmsprop_conv2d_bias_rmsIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp/assignvariableop_71_rmsprop_conv2d_4_kernel_rmsIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp-assignvariableop_72_rmsprop_conv2d_4_bias_rmsIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp/assignvariableop_73_rmsprop_conv2d_2_kernel_rmsIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp-assignvariableop_74_rmsprop_conv2d_2_bias_rmsIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp/assignvariableop_75_rmsprop_conv2d_5_kernel_rmsIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp-assignvariableop_76_rmsprop_conv2d_5_bias_rmsIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp/assignvariableop_77_rmsprop_conv2d_1_kernel_rmsIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp-assignvariableop_78_rmsprop_conv2d_1_bias_rmsIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp/assignvariableop_79_rmsprop_conv2d_3_kernel_rmsIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp-assignvariableop_80_rmsprop_conv2d_3_bias_rmsIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp/assignvariableop_81_rmsprop_conv2d_6_kernel_rmsIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp-assignvariableop_82_rmsprop_conv2d_6_bias_rmsIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp/assignvariableop_83_rmsprop_conv2d_7_kernel_rmsIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp-assignvariableop_84_rmsprop_conv2d_7_bias_rmsIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp/assignvariableop_85_rmsprop_conv2d_9_kernel_rmsIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp-assignvariableop_86_rmsprop_conv2d_9_bias_rmsIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp0assignvariableop_87_rmsprop_conv2d_10_kernel_rmsIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp.assignvariableop_88_rmsprop_conv2d_10_bias_rmsIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp/assignvariableop_89_rmsprop_conv2d_8_kernel_rmsIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp-assignvariableop_90_rmsprop_conv2d_8_bias_rmsIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp0assignvariableop_91_rmsprop_conv2d_11_kernel_rmsIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp.assignvariableop_92_rmsprop_conv2d_11_bias_rmsIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp0assignvariableop_93_rmsprop_conv2d_13_kernel_rmsIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp.assignvariableop_94_rmsprop_conv2d_13_bias_rmsIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp0assignvariableop_95_rmsprop_conv2d_14_kernel_rmsIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp.assignvariableop_96_rmsprop_conv2d_14_bias_rmsIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp0assignvariableop_97_rmsprop_conv2d_12_kernel_rmsIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp.assignvariableop_98_rmsprop_conv2d_12_bias_rmsIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp0assignvariableop_99_rmsprop_conv2d_15_kernel_rmsIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp/assignvariableop_100_rmsprop_conv2d_15_bias_rmsIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp1assignvariableop_101_rmsprop_conv2d_16_kernel_rmsIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp/assignvariableop_102_rmsprop_conv2d_16_bias_rmsIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp1assignvariableop_103_rmsprop_conv2d_21_kernel_rmsIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp/assignvariableop_104_rmsprop_conv2d_21_bias_rmsIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp1assignvariableop_105_rmsprop_conv2d_17_kernel_rmsIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp/assignvariableop_106_rmsprop_conv2d_17_bias_rmsIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp1assignvariableop_107_rmsprop_conv2d_19_kernel_rmsIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp/assignvariableop_108_rmsprop_conv2d_19_bias_rmsIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp1assignvariableop_109_rmsprop_conv2d_22_kernel_rmsIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp/assignvariableop_110_rmsprop_conv2d_22_bias_rmsIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp1assignvariableop_111_rmsprop_conv2d_18_kernel_rmsIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp/assignvariableop_112_rmsprop_conv2d_18_bias_rmsIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp1assignvariableop_113_rmsprop_conv2d_20_kernel_rmsIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp/assignvariableop_114_rmsprop_conv2d_20_bias_rmsIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp1assignvariableop_115_rmsprop_conv2d_23_kernel_rmsIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp/assignvariableop_116_rmsprop_conv2d_23_bias_rmsIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp1assignvariableop_117_rmsprop_conv2d_25_kernel_rmsIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp/assignvariableop_118_rmsprop_conv2d_25_bias_rmsIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp1assignvariableop_119_rmsprop_conv2d_26_kernel_rmsIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp/assignvariableop_120_rmsprop_conv2d_26_bias_rmsIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp1assignvariableop_121_rmsprop_conv2d_24_kernel_rmsIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp/assignvariableop_122_rmsprop_conv2d_24_bias_rmsIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp1assignvariableop_123_rmsprop_conv2d_27_kernel_rmsIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp/assignvariableop_124_rmsprop_conv2d_27_bias_rmsIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp1assignvariableop_125_rmsprop_conv2d_28_kernel_rmsIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp/assignvariableop_126_rmsprop_conv2d_28_bias_rmsIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOp3assignvariableop_127_rmsprop_predictions_kernel_rmsIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp1assignvariableop_128_rmsprop_predictions_bias_rmsIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_129Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_130IdentityIdentity_129:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_130Identity_130:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_269562

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????J?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_21_layer_call_fn_269384

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_268186
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11:0@

unknown_12:@%

unknown_13:? 

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: $

unknown_19: 0

unknown_20:0$

unknown_21: 0

unknown_22:0%

unknown_23:? 

unknown_24: $

unknown_25: (

unknown_26:(%

unknown_27:? 

unknown_28: $

unknown_29:(0

unknown_30:0%

unknown_31:P?

unknown_32:	?%

unknown_33:? 

unknown_34: $

unknown_35: $

unknown_36:$%

unknown_37:? 

unknown_38: %

unknown_39:? 

unknown_40: $

unknown_41: 0

unknown_42:0$

unknown_43: $

unknown_44:$$

unknown_45:$(

unknown_46:(%

unknown_47:? 

unknown_48: $

unknown_49: %

unknown_50:%%

unknown_51:? 

unknown_52: $

unknown_53:%*

unknown_54:*%

unknown_55:J?

unknown_56:	?

unknown_57:	?

unknown_58:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_265913o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
?
?
)__inference_conv2d_8_layer_call_fn_269185

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_269156

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????J?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapeso
m:?????????J?:?????????J0:?????????J$:?????????J(:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????J0
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????J$
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????J(
 
_user_specified_nameinputs
?
k
A__inference_add_2_layer_call_and_return_conditional_losses_266564

inputs
inputs_1
identityY
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????J?X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????J?:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
?
*__inference_conv2d_22_layer_call_fn_269444

inputs!
unknown: $
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
G
+__inference_final_pool_layer_call_fn_269706

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_266578a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?	
v
.__inference_concatenate_3_layer_call_fn_269543
inputs_0
inputs_1
inputs_2
inputs_3
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapeso
m:?????????J?:?????????J0:?????????J$:?????????J(:Z V
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????J0
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????J$
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:?????????J(
"
_user_specified_name
inputs/3
?
i
?__inference_add_layer_call_and_return_conditional_losses_266114

inputs
inputs_1
identityY
addAddV2inputsinputs_1*
T0*0
_output_shapes
:?????????? X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????? :?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
G__inference_predictions_layer_call_and_return_conditional_losses_266598

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_268930

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????&?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
?
*__inference_conv2d_11_layer_call_fn_269205

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_3_layer_call_fn_269029

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
b
F__inference_final_pool_layer_call_and_return_conditional_losses_269712

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_26_layer_call_fn_269591

inputs!
unknown: %
	unknown_0:%
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J%`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????J 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_269060

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_269535

inputs8
conv2d_readvariableop_resource:$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$(*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?$
 
_user_specified_nameinputs
?
?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_269231
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:?????????	??a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????	? :?????????	?0:?????????	?0:Z V
0
_output_shapes
:?????????	? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/2
?
?
*__inference_conv2d_12_layer_call_fn_269290

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
e
,__inference_dropout_end_layer_call_fn_269728

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266758p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_Chemception_layer_call_fn_268436

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11:0@

unknown_12:@%

unknown_13:? 

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: $

unknown_19: 0

unknown_20:0$

unknown_21: 0

unknown_22:0%

unknown_23:? 

unknown_24: $

unknown_25: (

unknown_26:(%

unknown_27:? 

unknown_28: $

unknown_29:(0

unknown_30:0%

unknown_31:P?

unknown_32:	?%

unknown_33:? 

unknown_34: $

unknown_35: $

unknown_36:$%

unknown_37:? 

unknown_38: %

unknown_39:? 

unknown_40: $

unknown_41: 0

unknown_42:0$

unknown_43: $

unknown_44:$$

unknown_45:$(

unknown_46:(%

unknown_47:? 

unknown_48: $

unknown_49: %

unknown_50:%%

unknown_51:? 

unknown_52: $

unknown_53:%*

unknown_54:*%

unknown_55:J?

unknown_56:	?

unknown_57:	?

unknown_58:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Chemception_layer_call_and_return_conditional_losses_267461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
?
*__inference_conv2d_14_layer_call_fn_269270

inputs!
unknown: (
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
I__inference_concatenate_3_layer_call_and_return_conditional_losses_269552
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????J?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapeso
m:?????????J?:?????????J0:?????????J$:?????????J(:Z V
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????J0
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:?????????J$
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:?????????J(
"
_user_specified_name
inputs/3
?
?
,__inference_predictions_layer_call_fn_269754

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_predictions_layer_call_and_return_conditional_losses_266598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552

inputs9
conv2d_readvariableop_resource:J?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:J?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????J?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????J?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????JJ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????JJ
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_268940

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????? c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_4_layer_call_fn_268949

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_13_layer_call_fn_269250

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
,__inference_Chemception_layer_call_fn_266728
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11:0@

unknown_12:@%

unknown_13:? 

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: $

unknown_19: 0

unknown_20:0$

unknown_21: 0

unknown_22:0%

unknown_23:? 

unknown_24: $

unknown_25: (

unknown_26:(%

unknown_27:? 

unknown_28: $

unknown_29:(0

unknown_30:0%

unknown_31:P?

unknown_32:	?%

unknown_33:? 

unknown_34: $

unknown_35: $

unknown_36:$%

unknown_37:? 

unknown_38: %

unknown_39:? 

unknown_40: $

unknown_41: 0

unknown_42:0$

unknown_43: $

unknown_44:$$

unknown_45:$(

unknown_46:(%

unknown_47:? 

unknown_48: $

unknown_49: %

unknown_50:%%

unknown_51:? 

unknown_52: $

unknown_53:%*

unknown_54:*%

unknown_55:J?

unknown_56:	?

unknown_57:	?

unknown_58:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Chemception_layer_call_and_return_conditional_losses_266605o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
?
?
'__inference_conv2d_layer_call_fn_268920

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_265967x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????&?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????&?
 
_user_specified_nameinputs
?
b
F__inference_final_pool_layer_call_and_return_conditional_losses_269718

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
k
?__inference_add_layer_call_and_return_conditional_losses_269106
inputs_0
inputs_1
identity[
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:?????????? X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1
?
?
G__inference_concatenate_layer_call_and_return_conditional_losses_269075
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????? :?????????? :??????????@:Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????@
"
_user_specified_name
inputs/2
?
b
F__inference_final_pool_layer_call_and_return_conditional_losses_265947

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
P
$__inference_add_layer_call_fn_269100
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_266114i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1
?
?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_269395

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_269435

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_269465

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_266121

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????? c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_4_layer_call_fn_269648
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????JJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????JJ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????J :?????????J*:Y U
/
_output_shapes
:?????????J 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????J*
"
_user_specified_name
inputs/1
?
?
)__inference_conv2d_5_layer_call_fn_268989

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????J?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
e
G__inference_dropout_end_layer_call_and_return_conditional_losses_269733

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_266463

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????J?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
??
?
G__inference_Chemception_layer_call_and_return_conditional_losses_268053
input_1'
conv2d_267884: 
conv2d_267886: )
conv2d_4_267890:  
conv2d_4_267892: )
conv2d_5_267895: 0
conv2d_5_267897:0)
conv2d_2_267900:  
conv2d_2_267902: )
conv2d_1_267905:  
conv2d_1_267907: )
conv2d_3_267910:  
conv2d_3_267912: )
conv2d_6_267915:0@
conv2d_6_267917:@*
conv2d_7_267921:? 
conv2d_7_267923: )
conv2d_9_267928:  
conv2d_9_267930: *
conv2d_10_267933:  
conv2d_10_267935: )
conv2d_8_267939: 0
conv2d_8_267941:0*
conv2d_11_267944: 0
conv2d_11_267946:0+
conv2d_13_267951:? 
conv2d_13_267953: *
conv2d_14_267956: (
conv2d_14_267958:(+
conv2d_12_267961:? 
conv2d_12_267963: *
conv2d_15_267966:(0
conv2d_15_267968:0+
conv2d_16_267972:P?
conv2d_16_267974:	?+
conv2d_21_267979:? 
conv2d_21_267981: *
conv2d_22_267984: $
conv2d_22_267986:$+
conv2d_19_267989:? 
conv2d_19_267991: +
conv2d_17_267994:? 
conv2d_17_267996: *
conv2d_18_268000: 0
conv2d_18_268002:0*
conv2d_20_268005: $
conv2d_20_268007:$*
conv2d_23_268010:$(
conv2d_23_268012:(+
conv2d_25_268017:? 
conv2d_25_268019: *
conv2d_26_268022: %
conv2d_26_268024:%+
conv2d_24_268027:? 
conv2d_24_268029: *
conv2d_27_268032:%*
conv2d_27_268034:*+
conv2d_28_268038:J?
conv2d_28_268040:	?%
predictions_268047:	? 
predictions_268049:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?#dropout_end/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_267884conv2d_267886*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_265967?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_265978?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_4_267890conv2d_4_267892*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_267895conv2d_5_267897*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_2_267900conv2d_2_267902*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_267905conv2d_1_267907*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_267910conv2d_3_267912*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_267915conv2d_6_267917*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076?
concatenate/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_266090?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_7_267921conv2d_7_267923*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102?
add/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_266114?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_266121?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_9_267928conv2d_9_267930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_267933conv2d_10_267935*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151?
max_pooling2d/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_8_267939conv2d_8_267941*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_267944conv2d_11_267946*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191?
concatenate_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_8/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205?
activation_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_266212?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_13_267951conv2d_13_267953*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_267956conv2d_14_267958*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_12_267961conv2d_12_267963*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_267966conv2d_15_267968*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276?
concatenate_2/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_16_267972conv2d_16_267974*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301?
add_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_266313?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_266320?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_21_267979conv2d_21_267981*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_267984conv2d_22_267986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_19_267989conv2d_19_267991*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_17_267994conv2d_17_267996*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_268000conv2d_18_268002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_268005conv2d_20_268007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_268010conv2d_23_268012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441?
concatenate_3/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456?
activation_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_266463?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_25_268017conv2d_25_268019*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_268022conv2d_26_268024*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_24_268027conv2d_24_268029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_268032conv2d_27_268034*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527?
concatenate_4/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????JJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_28_268038conv2d_28_268040*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552?
add_2/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_266564?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_266571?
final_pool/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_266578?
#dropout_end/StatefulPartitionedCallStatefulPartitionedCall#final_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266758?
#predictions/StatefulPartitionedCallStatefulPartitionedCall,dropout_end/StatefulPartitionedCall:output:0predictions_268047predictions_268049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_predictions_layer_call_and_return_conditional_losses_266598{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall$^dropout_end/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2J
#dropout_end/StatefulPartitionedCall#dropout_end/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_269196

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_28_layer_call_fn_269664

inputs"
unknown:J?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????J?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????JJ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????JJ
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_269020

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_266320

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:?????????	??d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_269116

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????? c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_269642

inputs8
conv2d_readvariableop_resource:%*-
biasadd_readvariableop_resource:*
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:%**
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J**
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J*X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J*i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J*w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????J%
 
_user_specified_nameinputs
?
I
-__inference_activation_1_layer_call_fn_269111

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_266121i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527

inputs8
conv2d_readvariableop_resource:%*-
biasadd_readvariableop_resource:*
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:%**
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J**
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J*X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J*i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J*w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????J%
 
_user_specified_nameinputs
?
G
+__inference_final_pool_layer_call_fn_269701

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_265947i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
R
&__inference_add_2_layer_call_fn_269680
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_266564i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????J?:?????????J?:Z V
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????J?
"
_user_specified_name
inputs/1
?
?
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424

inputs8
conv2d_readvariableop_resource: $-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: $*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J$X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J$i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269171

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????	? *
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????	? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_24_layer_call_fn_269611

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?

f
G__inference_dropout_end_layer_call_and_return_conditional_losses_269745

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed}[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????	?P`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????	?P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????	? :?????????	?0:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????	?0
 
_user_specified_nameinputs
?

?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301

inputs9
conv2d_readvariableop_resource:P?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:P?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????	??i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:?????????	??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?P
 
_user_specified_nameinputs
?
?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_269582

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269470

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_269334
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????	?P`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????	?P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????	? :?????????	?0:Z V
0
_output_shapes
:?????????	? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????	?0
"
_user_specified_name
inputs/1
?
b
F__inference_final_pool_layer_call_and_return_conditional_losses_266578

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_269040

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
e
G__inference_dropout_end_layer_call_and_return_conditional_losses_266585

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_269301

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
I
-__inference_activation_2_layer_call_fn_269236

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_266212j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_269460

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_265934?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

G__inference_concatenate_layer_call_and_return_conditional_losses_266090

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????? :?????????? :??????????@:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_269009

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
k
A__inference_add_1_layer_call_and_return_conditional_losses_266313

inputs
inputs_1
identityZ
addAddV2inputsinputs_1*
T0*1
_output_shapes
:?????????	??Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:?????????	??"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::?????????	??:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs:YU
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
)__inference_conv2d_9_layer_call_fn_269125

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_269602

inputs8
conv2d_readvariableop_resource: %-
biasadd_readvariableop_resource:%
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: %*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J%X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J%i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J%w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????J : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????J 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_18_layer_call_fn_269484

inputs!
unknown: 0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????J0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_6_layer_call_fn_269049

inputs!
unknown:0@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
f
,__inference_concatenate_layer_call_fn_269067
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_266090j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:?????????? :?????????? :??????????@:Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????@
"
_user_specified_name
inputs/2
?
?
)__inference_conv2d_7_layer_call_fn_269084

inputs"
unknown:? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????JJ_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????JJ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????J :?????????J*:W S
/
_output_shapes
:?????????J 
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????J*
 
_user_specified_nameinputs
?
?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441

inputs8
conv2d_readvariableop_resource:$(-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$(*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J(X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J(i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	?$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	?$
 
_user_specified_nameinputs
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_269000

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????0j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
*__inference_conv2d_10_layer_call_fn_269145

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_268980

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
d
H__inference_activation_5_layer_call_and_return_conditional_losses_266571

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????J?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????J?:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_265978

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????? c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_268960

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_269166

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????	? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
G__inference_Chemception_layer_call_and_return_conditional_losses_267881
input_1'
conv2d_267712: 
conv2d_267714: )
conv2d_4_267718:  
conv2d_4_267720: )
conv2d_5_267723: 0
conv2d_5_267725:0)
conv2d_2_267728:  
conv2d_2_267730: )
conv2d_1_267733:  
conv2d_1_267735: )
conv2d_3_267738:  
conv2d_3_267740: )
conv2d_6_267743:0@
conv2d_6_267745:@*
conv2d_7_267749:? 
conv2d_7_267751: )
conv2d_9_267756:  
conv2d_9_267758: *
conv2d_10_267761:  
conv2d_10_267763: )
conv2d_8_267767: 0
conv2d_8_267769:0*
conv2d_11_267772: 0
conv2d_11_267774:0+
conv2d_13_267779:? 
conv2d_13_267781: *
conv2d_14_267784: (
conv2d_14_267786:(+
conv2d_12_267789:? 
conv2d_12_267791: *
conv2d_15_267794:(0
conv2d_15_267796:0+
conv2d_16_267800:P?
conv2d_16_267802:	?+
conv2d_21_267807:? 
conv2d_21_267809: *
conv2d_22_267812: $
conv2d_22_267814:$+
conv2d_19_267817:? 
conv2d_19_267819: +
conv2d_17_267822:? 
conv2d_17_267824: *
conv2d_18_267828: 0
conv2d_18_267830:0*
conv2d_20_267833: $
conv2d_20_267835:$*
conv2d_23_267838:$(
conv2d_23_267840:(+
conv2d_25_267845:? 
conv2d_25_267847: *
conv2d_26_267850: %
conv2d_26_267852:%+
conv2d_24_267855:? 
conv2d_24_267857: *
conv2d_27_267860:%*
conv2d_27_267862:*+
conv2d_28_267866:J?
conv2d_28_267868:	?%
predictions_267875:	? 
predictions_267877:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_267712conv2d_267714*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_265967?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_265978?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_4_267718conv2d_4_267720*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_265991?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_267723conv2d_5_267725*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_266008?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_2_267728conv2d_2_267730*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_266025?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_267733conv2d_1_267735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_266042?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_267738conv2d_3_267740*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_266059?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_267743conv2d_6_267745*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_266076?
concatenate/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0)conv2d_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_266090?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0conv2d_7_267749conv2d_7_267751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_266102?
add/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_266114?
activation_1/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_266121?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_9_267756conv2d_9_267758*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_266134?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_267761conv2d_10_267763*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_266151?
max_pooling2d/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_266161?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_8_267767conv2d_8_267769*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_266174?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_267772conv2d_11_267774*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_266191?
concatenate_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0)conv2d_8/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_266205?
activation_2/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_266212?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_13_267779conv2d_13_267781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_266225?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_267784conv2d_14_267786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_266242?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_12_267789conv2d_12_267791*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_266259?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_267794conv2d_15_267796*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_266276?
concatenate_2/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_266289?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_16_267800conv2d_16_267802*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_266301?
add_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_266313?
activation_3/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????	??* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_266320?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_21_267807conv2d_21_267809*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_21_layer_call_and_return_conditional_losses_266333?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_267812conv2d_22_267814*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_22_layer_call_and_return_conditional_losses_266350?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_19_267817conv2d_19_267819*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_19_layer_call_and_return_conditional_losses_266367?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_17_267822conv2d_17_267824*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_266384?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_266394?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_267828conv2d_18_267830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_18_layer_call_and_return_conditional_losses_266407?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_267833conv2d_20_267835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_20_layer_call_and_return_conditional_losses_266424?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_267838conv2d_23_267840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_23_layer_call_and_return_conditional_losses_266441?
concatenate_3/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*conv2d_18/StatefulPartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_266456?
activation_4/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_266463?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_25_267845conv2d_25_267847*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_25_layer_call_and_return_conditional_losses_266476?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_267850conv2d_26_267852*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266493?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_24_267855conv2d_24_267857*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_24_layer_call_and_return_conditional_losses_266510?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0conv2d_27_267860conv2d_27_267862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????J**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266527?
concatenate_4/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????JJ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_266540?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_28_267866conv2d_28_267868*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_266552?
add_2/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_266564?
activation_5/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????J?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_266571?
final_pool/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_final_pool_layer_call_and_return_conditional_losses_266578?
dropout_end/PartitionedCallPartitionedCall#final_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_end_layer_call_and_return_conditional_losses_266585?
#predictions/StatefulPartitionedCallStatefulPartitionedCall$dropout_end/PartitionedCall:output:0predictions_267875predictions_267877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_predictions_layer_call_and_return_conditional_losses_266598{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????&?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????&?
!
_user_specified_name	input_1
?
?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_269622

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????J X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????J i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????J w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????J?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????J?
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269475

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????J?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????J?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	??:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?
?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_269415

inputs9
conv2d_readvariableop_resource:? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:? *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	? Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	? j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????	??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????	??
 
_user_specified_nameinputs
?

f
G__inference_dropout_end_layer_call_and_return_conditional_losses_266758

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed}[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
input_19
serving_default_input_1:0?????????&??
predictions0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer-27
layer_with_weights-17
layer-28
layer_with_weights-18
layer-29
layer_with_weights-19
layer-30
 layer_with_weights-20
 layer-31
!layer-32
"layer_with_weights-21
"layer-33
#layer_with_weights-22
#layer-34
$layer_with_weights-23
$layer-35
%layer-36
&layer-37
'layer_with_weights-24
'layer-38
(layer_with_weights-25
(layer-39
)layer_with_weights-26
)layer-40
*layer_with_weights-27
*layer-41
+layer-42
,layer_with_weights-28
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer_with_weights-29
1layer-48
2	optimizer
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
t	variables
utrainable_variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter

?decay
?learning_rate
?momentum
?rho
8rms?
9rms?
Brms?
Crms?
Hrms?
Irms?
Nrms?
Orms?
Trms?
Urms?
Zrms?
[rms?
`rms?
arms?
jrms?
krms?
xrms?
yrms?
~rms?
rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms?"
	optimizer
?
80
91
B2
C3
H4
I5
N6
O7
T8
U9
Z10
[11
`12
a13
j14
k15
x16
y17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59"
trackable_list_wrapper
?
80
91
B2
C3
H4
I5
N6
O7
T8
U9
Z10
[11
`12
a13
j14
k15
x16
y17
~18
19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_4/kernel
: 2conv2d_4/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_2/kernel
: 2conv2d_2/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_5/kernel
:02conv2d_5/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_3/kernel
: 2conv2d_3/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'0@2conv2d_6/kernel
:@2conv2d_6/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(? 2conv2d_7/kernel
: 2conv2d_7/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_9/kernel
: 2conv2d_9/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_10/kernel
: 2conv2d_10/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_8/kernel
:02conv2d_8/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_11/kernel
:02conv2d_11/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_13/kernel
: 2conv2d_13/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( (2conv2d_14/kernel
:(2conv2d_14/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_12/kernel
: 2conv2d_12/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:((02conv2d_15/kernel
:02conv2d_15/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)P?2conv2d_16/kernel
:?2conv2d_16/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_21/kernel
: 2conv2d_21/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_17/kernel
: 2conv2d_17/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_19/kernel
: 2conv2d_19/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( $2conv2d_22/kernel
:$2conv2d_22/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_18/kernel
:02conv2d_18/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( $2conv2d_20/kernel
:$2conv2d_20/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:($(2conv2d_23/kernel
:(2conv2d_23/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_25/kernel
: 2conv2d_25/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( %2conv2d_26/kernel
:%2conv2d_26/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)? 2conv2d_24/kernel
: 2conv2d_24/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(%*2conv2d_27/kernel
:*2conv2d_27/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)J?2conv2d_28/kernel
:?2conv2d_28/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	?2predictions/kernel
:2predictions/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
?
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
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
1:/ 2RMSprop/conv2d/kernel/rms
#:! 2RMSprop/conv2d/bias/rms
3:1  2RMSprop/conv2d_4/kernel/rms
%:# 2RMSprop/conv2d_4/bias/rms
3:1  2RMSprop/conv2d_2/kernel/rms
%:# 2RMSprop/conv2d_2/bias/rms
3:1 02RMSprop/conv2d_5/kernel/rms
%:#02RMSprop/conv2d_5/bias/rms
3:1  2RMSprop/conv2d_1/kernel/rms
%:# 2RMSprop/conv2d_1/bias/rms
3:1  2RMSprop/conv2d_3/kernel/rms
%:# 2RMSprop/conv2d_3/bias/rms
3:10@2RMSprop/conv2d_6/kernel/rms
%:#@2RMSprop/conv2d_6/bias/rms
4:2? 2RMSprop/conv2d_7/kernel/rms
%:# 2RMSprop/conv2d_7/bias/rms
3:1  2RMSprop/conv2d_9/kernel/rms
%:# 2RMSprop/conv2d_9/bias/rms
4:2  2RMSprop/conv2d_10/kernel/rms
&:$ 2RMSprop/conv2d_10/bias/rms
3:1 02RMSprop/conv2d_8/kernel/rms
%:#02RMSprop/conv2d_8/bias/rms
4:2 02RMSprop/conv2d_11/kernel/rms
&:$02RMSprop/conv2d_11/bias/rms
5:3? 2RMSprop/conv2d_13/kernel/rms
&:$ 2RMSprop/conv2d_13/bias/rms
4:2 (2RMSprop/conv2d_14/kernel/rms
&:$(2RMSprop/conv2d_14/bias/rms
5:3? 2RMSprop/conv2d_12/kernel/rms
&:$ 2RMSprop/conv2d_12/bias/rms
4:2(02RMSprop/conv2d_15/kernel/rms
&:$02RMSprop/conv2d_15/bias/rms
5:3P?2RMSprop/conv2d_16/kernel/rms
':%?2RMSprop/conv2d_16/bias/rms
5:3? 2RMSprop/conv2d_21/kernel/rms
&:$ 2RMSprop/conv2d_21/bias/rms
5:3? 2RMSprop/conv2d_17/kernel/rms
&:$ 2RMSprop/conv2d_17/bias/rms
5:3? 2RMSprop/conv2d_19/kernel/rms
&:$ 2RMSprop/conv2d_19/bias/rms
4:2 $2RMSprop/conv2d_22/kernel/rms
&:$$2RMSprop/conv2d_22/bias/rms
4:2 02RMSprop/conv2d_18/kernel/rms
&:$02RMSprop/conv2d_18/bias/rms
4:2 $2RMSprop/conv2d_20/kernel/rms
&:$$2RMSprop/conv2d_20/bias/rms
4:2$(2RMSprop/conv2d_23/kernel/rms
&:$(2RMSprop/conv2d_23/bias/rms
5:3? 2RMSprop/conv2d_25/kernel/rms
&:$ 2RMSprop/conv2d_25/bias/rms
4:2 %2RMSprop/conv2d_26/kernel/rms
&:$%2RMSprop/conv2d_26/bias/rms
5:3? 2RMSprop/conv2d_24/kernel/rms
&:$ 2RMSprop/conv2d_24/bias/rms
4:2%*2RMSprop/conv2d_27/kernel/rms
&:$*2RMSprop/conv2d_27/bias/rms
5:3J?2RMSprop/conv2d_28/kernel/rms
':%?2RMSprop/conv2d_28/bias/rms
/:-	?2RMSprop/predictions/kernel/rms
(:&2RMSprop/predictions/bias/rms
?2?
,__inference_Chemception_layer_call_fn_266728
,__inference_Chemception_layer_call_fn_268311
,__inference_Chemception_layer_call_fn_268436
,__inference_Chemception_layer_call_fn_267709?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_Chemception_layer_call_and_return_conditional_losses_268670
G__inference_Chemception_layer_call_and_return_conditional_losses_268911
G__inference_Chemception_layer_call_and_return_conditional_losses_267881
G__inference_Chemception_layer_call_and_return_conditional_losses_268053?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_265913input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_layer_call_fn_268920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_268930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_layer_call_fn_268935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_layer_call_and_return_conditional_losses_268940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_4_layer_call_fn_268949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_268960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_2_layer_call_fn_268969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_268980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_5_layer_call_fn_268989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_269000?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_1_layer_call_fn_269009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_269020?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_3_layer_call_fn_269029?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_269040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_6_layer_call_fn_269049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_269060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_layer_call_fn_269067?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_269075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_7_layer_call_fn_269084?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_269094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_add_layer_call_fn_269100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_add_layer_call_and_return_conditional_losses_269106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_1_layer_call_fn_269111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_1_layer_call_and_return_conditional_losses_269116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_9_layer_call_fn_269125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_269136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_10_layer_call_fn_269145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_269156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_layer_call_fn_269161
.__inference_max_pooling2d_layer_call_fn_269166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269171
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_8_layer_call_fn_269185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_269196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_11_layer_call_fn_269205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_269216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_1_layer_call_fn_269223?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_269231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_2_layer_call_fn_269236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_2_layer_call_and_return_conditional_losses_269241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_13_layer_call_fn_269250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_269261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_14_layer_call_fn_269270?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_269281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_12_layer_call_fn_269290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_269301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_15_layer_call_fn_269310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_269321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_2_layer_call_fn_269327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_2_layer_call_and_return_conditional_losses_269334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_16_layer_call_fn_269343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_269353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_add_1_layer_call_fn_269359?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_add_1_layer_call_and_return_conditional_losses_269365?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_3_layer_call_fn_269370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_3_layer_call_and_return_conditional_losses_269375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_21_layer_call_fn_269384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_269395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_17_layer_call_fn_269404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_269415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_19_layer_call_fn_269424?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_269435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_22_layer_call_fn_269444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_269455?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_1_layer_call_fn_269460
0__inference_max_pooling2d_1_layer_call_fn_269465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269470
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_18_layer_call_fn_269484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_269495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_20_layer_call_fn_269504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_20_layer_call_and_return_conditional_losses_269515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_23_layer_call_fn_269524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_269535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_3_layer_call_fn_269543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_3_layer_call_and_return_conditional_losses_269552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_4_layer_call_fn_269557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_4_layer_call_and_return_conditional_losses_269562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_25_layer_call_fn_269571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_269582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_26_layer_call_fn_269591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_269602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_24_layer_call_fn_269611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_269622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_27_layer_call_fn_269631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_269642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_4_layer_call_fn_269648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_4_layer_call_and_return_conditional_losses_269655?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_28_layer_call_fn_269664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_28_layer_call_and_return_conditional_losses_269674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_add_2_layer_call_fn_269680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_add_2_layer_call_and_return_conditional_losses_269686?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_5_layer_call_fn_269691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_5_layer_call_and_return_conditional_losses_269696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_final_pool_layer_call_fn_269701
+__inference_final_pool_layer_call_fn_269706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_final_pool_layer_call_and_return_conditional_losses_269712
F__inference_final_pool_layer_call_and_return_conditional_losses_269718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dropout_end_layer_call_fn_269723
,__inference_dropout_end_layer_call_fn_269728?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_end_layer_call_and_return_conditional_losses_269733
G__inference_dropout_end_layer_call_and_return_conditional_losses_269745?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_predictions_layer_call_fn_269754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_predictions_layer_call_and_return_conditional_losses_269765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_268186input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
G__inference_Chemception_layer_call_and_return_conditional_losses_267881?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????A?>
7?4
*?'
input_1?????????&?
p 

 
? "%?"
?
0?????????
? ?
G__inference_Chemception_layer_call_and_return_conditional_losses_268053?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????A?>
7?4
*?'
input_1?????????&?
p

 
? "%?"
?
0?????????
? ?
G__inference_Chemception_layer_call_and_return_conditional_losses_268670?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????@?=
6?3
)?&
inputs?????????&?
p 

 
? "%?"
?
0?????????
? ?
G__inference_Chemception_layer_call_and_return_conditional_losses_268911?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????@?=
6?3
)?&
inputs?????????&?
p

 
? "%?"
?
0?????????
? ?
,__inference_Chemception_layer_call_fn_266728?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????A?>
7?4
*?'
input_1?????????&?
p 

 
? "???????????
,__inference_Chemception_layer_call_fn_267709?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????A?>
7?4
*?'
input_1?????????&?
p

 
? "???????????
,__inference_Chemception_layer_call_fn_268311?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????@?=
6?3
)?&
inputs?????????&?
p 

 
? "???????????
,__inference_Chemception_layer_call_fn_268436?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????@?=
6?3
)?&
inputs?????????&?
p

 
? "???????????
!__inference__wrapped_model_265913?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????9?6
/?,
*?'
input_1?????????&?
? "9?6
4
predictions%?"
predictions??????????
H__inference_activation_1_layer_call_and_return_conditional_losses_269116j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
-__inference_activation_1_layer_call_fn_269111]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
H__inference_activation_2_layer_call_and_return_conditional_losses_269241l9?6
/?,
*?'
inputs?????????	??
? "/?,
%?"
0?????????	??
? ?
-__inference_activation_2_layer_call_fn_269236_9?6
/?,
*?'
inputs?????????	??
? ""??????????	???
H__inference_activation_3_layer_call_and_return_conditional_losses_269375l9?6
/?,
*?'
inputs?????????	??
? "/?,
%?"
0?????????	??
? ?
-__inference_activation_3_layer_call_fn_269370_9?6
/?,
*?'
inputs?????????	??
? ""??????????	???
H__inference_activation_4_layer_call_and_return_conditional_losses_269562j8?5
.?+
)?&
inputs?????????J?
? ".?+
$?!
0?????????J?
? ?
-__inference_activation_4_layer_call_fn_269557]8?5
.?+
)?&
inputs?????????J?
? "!??????????J??
H__inference_activation_5_layer_call_and_return_conditional_losses_269696j8?5
.?+
)?&
inputs?????????J?
? ".?+
$?!
0?????????J?
? ?
-__inference_activation_5_layer_call_fn_269691]8?5
.?+
)?&
inputs?????????J?
? "!??????????J??
F__inference_activation_layer_call_and_return_conditional_losses_268940j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
+__inference_activation_layer_call_fn_268935]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
A__inference_add_1_layer_call_and_return_conditional_losses_269365?n?k
d?a
_?\
,?)
inputs/0?????????	??
,?)
inputs/1?????????	??
? "/?,
%?"
0?????????	??
? ?
&__inference_add_1_layer_call_fn_269359?n?k
d?a
_?\
,?)
inputs/0?????????	??
,?)
inputs/1?????????	??
? ""??????????	???
A__inference_add_2_layer_call_and_return_conditional_losses_269686?l?i
b?_
]?Z
+?(
inputs/0?????????J?
+?(
inputs/1?????????J?
? ".?+
$?!
0?????????J?
? ?
&__inference_add_2_layer_call_fn_269680?l?i
b?_
]?Z
+?(
inputs/0?????????J?
+?(
inputs/1?????????J?
? "!??????????J??
?__inference_add_layer_call_and_return_conditional_losses_269106?l?i
b?_
]?Z
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
? ".?+
$?!
0?????????? 
? ?
$__inference_add_layer_call_fn_269100?l?i
b?_
]?Z
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
? "!??????????? ?
I__inference_concatenate_1_layer_call_and_return_conditional_losses_269231????
???
???
+?(
inputs/0?????????	? 
+?(
inputs/1?????????	?0
+?(
inputs/2?????????	?0
? "/?,
%?"
0?????????	??
? ?
.__inference_concatenate_1_layer_call_fn_269223????
???
???
+?(
inputs/0?????????	? 
+?(
inputs/1?????????	?0
+?(
inputs/2?????????	?0
? ""??????????	???
I__inference_concatenate_2_layer_call_and_return_conditional_losses_269334?l?i
b?_
]?Z
+?(
inputs/0?????????	? 
+?(
inputs/1?????????	?0
? ".?+
$?!
0?????????	?P
? ?
.__inference_concatenate_2_layer_call_fn_269327?l?i
b?_
]?Z
+?(
inputs/0?????????	? 
+?(
inputs/1?????????	?0
? "!??????????	?P?
I__inference_concatenate_3_layer_call_and_return_conditional_losses_269552????
???
???
+?(
inputs/0?????????J?
*?'
inputs/1?????????J0
*?'
inputs/2?????????J$
*?'
inputs/3?????????J(
? ".?+
$?!
0?????????J?
? ?
.__inference_concatenate_3_layer_call_fn_269543????
???
???
+?(
inputs/0?????????J?
*?'
inputs/1?????????J0
*?'
inputs/2?????????J$
*?'
inputs/3?????????J(
? "!??????????J??
I__inference_concatenate_4_layer_call_and_return_conditional_losses_269655?j?g
`?]
[?X
*?'
inputs/0?????????J 
*?'
inputs/1?????????J*
? "-?*
#? 
0?????????JJ
? ?
.__inference_concatenate_4_layer_call_fn_269648?j?g
`?]
[?X
*?'
inputs/0?????????J 
*?'
inputs/1?????????J*
? " ??????????JJ?
G__inference_concatenate_layer_call_and_return_conditional_losses_269075????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2??????????@
? "/?,
%?"
0???????????
? ?
,__inference_concatenate_layer_call_fn_269067????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2??????????@
? ""?????????????
E__inference_conv2d_10_layer_call_and_return_conditional_losses_269156n~8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
*__inference_conv2d_10_layer_call_fn_269145a~8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_269216p??8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????	?0
? ?
*__inference_conv2d_11_layer_call_fn_269205c??8?5
.?+
)?&
inputs?????????? 
? "!??????????	?0?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_269301q??9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????	? 
? ?
*__inference_conv2d_12_layer_call_fn_269290d??9?6
/?,
*?'
inputs?????????	??
? "!??????????	? ?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_269261q??9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????	? 
? ?
*__inference_conv2d_13_layer_call_fn_269250d??9?6
/?,
*?'
inputs?????????	??
? "!??????????	? ?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_269281p??8?5
.?+
)?&
inputs?????????	? 
? ".?+
$?!
0?????????	?(
? ?
*__inference_conv2d_14_layer_call_fn_269270c??8?5
.?+
)?&
inputs?????????	? 
? "!??????????	?(?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_269321p??8?5
.?+
)?&
inputs?????????	?(
? ".?+
$?!
0?????????	?0
? ?
*__inference_conv2d_15_layer_call_fn_269310c??8?5
.?+
)?&
inputs?????????	?(
? "!??????????	?0?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_269353q??8?5
.?+
)?&
inputs?????????	?P
? "/?,
%?"
0?????????	??
? ?
*__inference_conv2d_16_layer_call_fn_269343d??8?5
.?+
)?&
inputs?????????	?P
? ""??????????	???
E__inference_conv2d_17_layer_call_and_return_conditional_losses_269415q??9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????	? 
? ?
*__inference_conv2d_17_layer_call_fn_269404d??9?6
/?,
*?'
inputs?????????	??
? "!??????????	? ?
E__inference_conv2d_18_layer_call_and_return_conditional_losses_269495o??8?5
.?+
)?&
inputs?????????	? 
? "-?*
#? 
0?????????J0
? ?
*__inference_conv2d_18_layer_call_fn_269484b??8?5
.?+
)?&
inputs?????????	? 
? " ??????????J0?
E__inference_conv2d_19_layer_call_and_return_conditional_losses_269435q??9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????	? 
? ?
*__inference_conv2d_19_layer_call_fn_269424d??9?6
/?,
*?'
inputs?????????	??
? "!??????????	? ?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_269020nTU8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_1_layer_call_fn_269009aTU8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
E__inference_conv2d_20_layer_call_and_return_conditional_losses_269515o??8?5
.?+
)?&
inputs?????????	? 
? "-?*
#? 
0?????????J$
? ?
*__inference_conv2d_20_layer_call_fn_269504b??8?5
.?+
)?&
inputs?????????	? 
? " ??????????J$?
E__inference_conv2d_21_layer_call_and_return_conditional_losses_269395q??9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????	? 
? ?
*__inference_conv2d_21_layer_call_fn_269384d??9?6
/?,
*?'
inputs?????????	??
? "!??????????	? ?
E__inference_conv2d_22_layer_call_and_return_conditional_losses_269455p??8?5
.?+
)?&
inputs?????????	? 
? ".?+
$?!
0?????????	?$
? ?
*__inference_conv2d_22_layer_call_fn_269444c??8?5
.?+
)?&
inputs?????????	? 
? "!??????????	?$?
E__inference_conv2d_23_layer_call_and_return_conditional_losses_269535o??8?5
.?+
)?&
inputs?????????	?$
? "-?*
#? 
0?????????J(
? ?
*__inference_conv2d_23_layer_call_fn_269524b??8?5
.?+
)?&
inputs?????????	?$
? " ??????????J(?
E__inference_conv2d_24_layer_call_and_return_conditional_losses_269622o??8?5
.?+
)?&
inputs?????????J?
? "-?*
#? 
0?????????J 
? ?
*__inference_conv2d_24_layer_call_fn_269611b??8?5
.?+
)?&
inputs?????????J?
? " ??????????J ?
E__inference_conv2d_25_layer_call_and_return_conditional_losses_269582o??8?5
.?+
)?&
inputs?????????J?
? "-?*
#? 
0?????????J 
? ?
*__inference_conv2d_25_layer_call_fn_269571b??8?5
.?+
)?&
inputs?????????J?
? " ??????????J ?
E__inference_conv2d_26_layer_call_and_return_conditional_losses_269602n??7?4
-?*
(?%
inputs?????????J 
? "-?*
#? 
0?????????J%
? ?
*__inference_conv2d_26_layer_call_fn_269591a??7?4
-?*
(?%
inputs?????????J 
? " ??????????J%?
E__inference_conv2d_27_layer_call_and_return_conditional_losses_269642n??7?4
-?*
(?%
inputs?????????J%
? "-?*
#? 
0?????????J*
? ?
*__inference_conv2d_27_layer_call_fn_269631a??7?4
-?*
(?%
inputs?????????J%
? " ??????????J*?
E__inference_conv2d_28_layer_call_and_return_conditional_losses_269674o??7?4
-?*
(?%
inputs?????????JJ
? ".?+
$?!
0?????????J?
? ?
*__inference_conv2d_28_layer_call_fn_269664b??7?4
-?*
(?%
inputs?????????JJ
? "!??????????J??
D__inference_conv2d_2_layer_call_and_return_conditional_losses_268980nHI8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_2_layer_call_fn_268969aHI8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_269040nZ[8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_3_layer_call_fn_269029aZ[8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_268960nBC8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_4_layer_call_fn_268949aBC8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_269000nNO8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0??????????0
? ?
)__inference_conv2d_5_layer_call_fn_268989aNO8?5
.?+
)?&
inputs?????????? 
? "!???????????0?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_269060n`a8?5
.?+
)?&
inputs??????????0
? ".?+
$?!
0??????????@
? ?
)__inference_conv2d_6_layer_call_fn_269049a`a8?5
.?+
)?&
inputs??????????0
? "!???????????@?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_269094ojk9?6
/?,
*?'
inputs???????????
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_7_layer_call_fn_269084bjk9?6
/?,
*?'
inputs???????????
? "!??????????? ?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_269196p??8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????	?0
? ?
)__inference_conv2d_8_layer_call_fn_269185c??8?5
.?+
)?&
inputs?????????? 
? "!??????????	?0?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_269136nxy8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
)__inference_conv2d_9_layer_call_fn_269125axy8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
B__inference_conv2d_layer_call_and_return_conditional_losses_268930n898?5
.?+
)?&
inputs?????????&?
? ".?+
$?!
0?????????? 
? ?
'__inference_conv2d_layer_call_fn_268920a898?5
.?+
)?&
inputs?????????&?
? "!??????????? ?
G__inference_dropout_end_layer_call_and_return_conditional_losses_269733^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
G__inference_dropout_end_layer_call_and_return_conditional_losses_269745^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
,__inference_dropout_end_layer_call_fn_269723Q4?1
*?'
!?
inputs??????????
p 
? "????????????
,__inference_dropout_end_layer_call_fn_269728Q4?1
*?'
!?
inputs??????????
p
? "????????????
F__inference_final_pool_layer_call_and_return_conditional_losses_269712?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
F__inference_final_pool_layer_call_and_return_conditional_losses_269718b8?5
.?+
)?&
inputs?????????J?
? "&?#
?
0??????????
? ?
+__inference_final_pool_layer_call_fn_269701wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
+__inference_final_pool_layer_call_fn_269706U8?5
.?+
)?&
inputs?????????J?
? "????????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269470?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_269475k9?6
/?,
*?'
inputs?????????	??
? ".?+
$?!
0?????????J?
? ?
0__inference_max_pooling2d_1_layer_call_fn_269460?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_1_layer_call_fn_269465^9?6
/?,
*?'
inputs?????????	??
? "!??????????J??
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269171?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_269176j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????	? 
? ?
.__inference_max_pooling2d_layer_call_fn_269161?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_layer_call_fn_269166]8?5
.?+
)?&
inputs?????????? 
? "!??????????	? ?
G__inference_predictions_layer_call_and_return_conditional_losses_269765_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
,__inference_predictions_layer_call_fn_269754R??0?-
&?#
!?
inputs??????????
? "???????????
$__inference_signature_wrapper_268186?d89BCNOHITUZ[`ajkxy~????????????????????????????????????????D?A
? 
:?7
5
input_1*?'
input_1?????????&?"9?6
4
predictions%?"
predictions?????????