$	.?iB??@t?p?!??%??C?@!?t?V@$	K]*Wx-??
Aڧ??8??a?B??!2?H?U???"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0;?O??n	@?n?????Ao??ʡ@Y?I+???rtrain 11"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Q?@Zd;?O???AX9??v@Y+??????rtrain 12"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~j?t?@NbX9???Aj?t?@Y??~j?t??rtrain 13"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%??C?@;?O??n??AV-?? @Y{?G?z??rtrain 14"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0o??ʡ@o??ʡ??AB`??"?@YZd;?O???rtrain 15"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t?V@??/?$??A??C?l???Y+??????rtrain 16"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0V-??@Zd;?O??A+????@Y{?G?z??rtrain 17*	     ??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat
? ?rh???!G9?t?A@)=
ףp=??1=????3?@:Preprocessing2E
Iterator::Root`??"????!?氚?jB@)?/?$??1??X>4@:Preprocessing2T
Iterator::Root::ParallelMapV2??C?l???!s??%/?0@)??C?l???1s??%/?0@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?MbX9??!? x8@)?????K??1:;?젳+@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???x?&??!U?SOe$@)???x?&??1U?SOe$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???Q???!<$???C@)???Q???1<$???C@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipsh??|???!CQ0D9@)?I+???1??u???
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9L??y??I????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?&t;"???b?/?}??Zd;?O???!??/?$??	!       "	!       *	!       2$	u?)?[@	gWR?d????C?l???!o??ʡ@:	!       B	!       J$	wD?8:??????Ph???~j?t??!Zd;?O???R	!       Z$	wD?8:??????Ph???~j?t??!Zd;?O???b	!       JCPU_ONLYYL??y??b q????X@