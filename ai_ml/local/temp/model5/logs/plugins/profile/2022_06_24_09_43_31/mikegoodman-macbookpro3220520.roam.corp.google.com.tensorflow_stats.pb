"?@
BHostIDLE"IDLE1    ?AA    ?Aaw?Fy??iw?Fy???Unknown
?HostMatMul"2gradient_tape/sequential_14/dense_30/MatMul/MatMul(1    ??@9颋.??@A    ??@I颋.??@a?P??????i??~?V????Unknown
uHost_FusedMatMul"sequential_14/dense_30/Relu(1     ٱ@9F]t??y@A     ٱ@IF]t??y@a??S??i?Ê??S???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     О@9颋.?hf@A     О@I颋.?hf@a?9-Fs?i6??2z???Unknown
?HostRandomUniform"=sequential_14/dropout_13/dropout/random_uniform/RandomUniform(1     H?@9F]t?V@A     H?@IF]t?V@a^'>??b?i)O#?#????Unknown
^HostGatherV2"GatherV2(1     ??@9F]t?ET@A     ??@IF]t?ET@a*:?oa?icP&v?????Unknown
?HostMatMul"4gradient_tape/sequential_14/dense_31/MatMul/MatMul_1(1     @{@9?E]t?C@A     @{@I?E]t?C@a?????Q?i?'B????Unknown
xHost_FusedMatMul"sequential_14/dense_31/BiasAdd(1     Ps@9]t?E<@A     Ps@I]t?E<@a????)H?iM??#????Unknown
?	HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1     s@9?袋.?;@A     s@I?袋.?;@a????G?i4???????Unknown
?
HostMatMul"2gradient_tape/sequential_14/dense_31/MatMul/MatMul(1     @r@9?.?袋:@A     @r@I?.?袋:@a?
C-??F?i???θ???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      r@9/?袋.:@A      r@I/?袋.:@a?4L??F?i?"?-p????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(
1     ?l@9?????7@A     ?i@I     ?4@a??=a@?i{??r????Unknown
`HostGatherV2"
GatherV2_1(1     ?e@9t?E]t/@A     ?e@It?E]t/@aWό?;?i${??????Unknown
sHostSoftmax"sequential_14/dense_31/Softmax(1     ?b@9]t?E+@A     ?b@I]t?E+@a?u?~?L7?iD??Y?????Unknown
uHostMul"$sequential_14/dropout_13/dropout/Mul(1     ?a@9??.???)@A     ?a@I??.???)@an????6?i?Rj?~????Unknown
cHostDataset"Iterator::Root(1     @n@9      6@A     ?`@I/?袋.(@a??avb?4?i7?z????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      `@9F]t?E'@A      `@IF]t?E'@a???C84?iȖ??????Unknown
?HostGreaterEqual"-sequential_14/dropout_13/dropout/GreaterEqual(1     ?^@9]t?E]&@A     ?^@I]t?E]&@aZ*<3?i???? ????Unknown
lHostIteratorGetNext"IteratorGetNext(1      ]@9]t?E%@A      ]@I]t?E%@a??b??#2?if??D????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1     @[@9     @+@A     @[@I     @+@a?????1?iٛ?pf????Unknown
wHostCast"%sequential_14/dropout_13/dropout/Cast(1     ?X@9      "@A     ?X@I      "@a֡???.?iS(Q?U????Unknown
wHostMul"&sequential_14/dropout_13/dropout/Mul_1(1     ?X@9      "@A     ?X@I      "@a֡???.?iʹ?AE????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1     @X@9??.???!@A     @X@I??.???!@a???FeV.?ii?*????Unknown
ZHostArgMax"ArgMax(1     ?V@9?.?袋 @A     ?V@I?.?袋 @a?P  v,?ij$?????Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1     ?c@9?袋.?,@A     ?V@I?.?袋 @a?P  v,?ik)h?????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_14/dense_31/BiasAdd/BiasAddGrad(1     @V@9/?袋. @A     @V@I/?袋. @a?,2>??+?i???v????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_14/dense_30/BiasAdd/BiasAddGrad(1      V@9       @A      V@I       @a?:#]ͅ+?i???"/????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      T@9]t?E@A      T@I]t?E@a???TF)?i})5w?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1     @S@9      @A     @S@I      @a5?~?(?ijAp?@????Unknown
?HostReadVariableOp",sequential_14/dense_31/MatMul/ReadVariableOp(1      Q@9?袋.?@A      Q@I?袋.?@a!P??{D%?i??,?????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1     ?P@9     ? @A     ?P@I     ? @a^??j?$?i?/?V?????Unknown
? HostReluGrad"-gradient_tape/sequential_14/dense_30/ReluGrad(1     @P@9??.???@A     @P@I??.???@a?y?$IT$?i=|m?)????Unknown
?!HostMul"4gradient_tape/sequential_14/dropout_13/dropout/Mul_2(1      P@9F]t?E@A      P@IF]t?E@a???C8$?i???i????Unknown
|"HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1     ?N@9/?袋.@A     ?N@I/?袋.@aK???#?i ?K?????Unknown
u#HostFlushSummaryWriter"FlushSummaryWriter(1      M@9      M@A      M@I      M@a??b??#"?i.?{\?????Unknown?
V$HostSum"Sum_2(1      M@9]t?E@A      M@I]t?E@a??b??#"?i\m???????Unknown
?%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      J@9颋.??@A      J@I颋.??@a7.	?mC ?i?݆??????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1     ?H@9?E]t?@A     ?H@I?E]t?@a???'v??i?8?????Unknown
?'HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      H@9t?E]t@A      H@It?E]t@az˚eT?i?H?6?????Unknown
`(HostDivNoNan"
div_no_nan(1      H@9t?E]t@A      H@It?E]t@az˚eT?i`u~i?????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1     ?F@9]t?E]@A     ?F@I]t?E]@a?A?%?iio???????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_1(1      F@9      @A      F@I      @a?:#]ͅ?i?Xb?v????Unknown
?+HostReadVariableOp"-sequential_14/dense_30/BiasAdd/ReadVariableOp(1      E@9?.?袋@A      E@I?.?袋@ar?؉E?i???H????Unknown
?,HostMul"2gradient_tape/sequential_14/dropout_13/dropout/Mul(1     ?D@9?E]t?@A     ?D@I?E]t?@a΍?h??i??????Unknown
i-HostWriteSummary"WriteSummary(1     ?A@9     ?A@A     ?A@I     ?A@a^4????i?&?C?????Unknown?
?.HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      >@9      ??A      >@I      ??a,??????i?"?c[????Unknown
m/HostReadVariableOp"ReadVariableOp(
1      >@9      @A      >@I      @a,??????i?*??????Unknown?
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      =@9]t?E@A      =@I]t?E@a??b??#?i?	¡?????Unknown
X1HostCast"Cast_2(1      =@9]t?E@A      =@I]t?E@a??b??#?i??Y?????Unknown
X2HostEqual"Equal(1      =@9]t?E@A      =@I]t?E@a??b??#?i??ޤ????Unknown
b3HostDivNoNan"div_no_nan_1(1      =@9]t?E@A      =@I]t?E@a??b??#?i%ˉ?5????Unknown
?4HostReadVariableOp"-sequential_14/dense_31/BiasAdd/ReadVariableOp(1      :@9颋.??@A      :@I颋.??@a7.	?mC?in???????Unknown
s5HostReadVariableOp"SGD/Cast/ReadVariableOp(1      9@9/?袋.@A      9@I/?袋.@a?????F?i?*W35????Unknown
?6HostReadVariableOp",sequential_14/dense_30/MatMul/ReadVariableOp(1      8@9t?E]t@A      8@It?E]t@az˚eT?i3??L?????Unknown
w7HostDataset""Iterator::Root::ParallelMapV2::Zip(1     ?d@9??????K@A      6@IUUUUUU@a?:#]ͅ?i?5?c????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1      5@9?.?袋??A      5@I?.?袋??ar?؉E
?i^?z?????Unknown
u9HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      5@9?.?袋??A      5@I?.?袋??ar?؉E
?i??,??????Unknown
X:HostCast"Cast_3(1      4@9]t?E??A      4@I]t?E??a???TF	?i?OF?Q????Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1      .@9?E]t???A      .@I?E]t???a,??????i?M??????Unknown
?<HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      .@9?E]t???A      .@I?E]t???a,??????i?K???????Unknown
w=HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      *@9颋.????A      *@I颋.????a7.	?mC ?i?'??(????Unknown
e>Host
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@a???TF?>i-?/?Z????Unknown?
y?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      "@9/?袋.??A      "@I/?袋.??a?4L???>i?i???????Unknown
a@HostIdentity"Identity(1       @9F]t?E??A       @IF]t?E??a???C8?>i???????Unknown?
TAHostMul"Mul(1       @9F]t?E??A       @IF]t?E??a???C8?>i?x???????Unknown
wBHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9F]t?E??A       @IF]t?E??a???C8?>i      ???Unknown*?@
?HostMatMul"2gradient_tape/sequential_14/dense_30/MatMul/MatMul(1    ??@9颋.??@A    ??@I颋.??@a?<??gK??i?<??gK???Unknown
uHost_FusedMatMul"sequential_14/dense_30/Relu(1     ٱ@9F]t??y@A     ٱ@IF]t??y@a"?????i?0A?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     О@9颋.?hf@A     О@I颋.?hf@aq??@rs??i??N?h ???Unknown
?HostRandomUniform"=sequential_14/dropout_13/dropout/random_uniform/RandomUniform(1     H?@9F]t?V@A     H?@IF]t?V@a???.??i???j?a???Unknown
^HostGatherV2"GatherV2(1     ??@9F]t?ET@A     ??@IF]t?ET@a???S???i?p?? ????Unknown
?HostMatMul"4gradient_tape/sequential_14/dense_31/MatMul/MatMul_1(1     @{@9?E]t?C@A     @{@I?E]t?C@acg??!??i??鰱???Unknown
xHost_FusedMatMul"sequential_14/dense_31/BiasAdd(1     Ps@9]t?E<@A     Ps@I]t?E<@a&	b䢉?i?qB=????Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1     s@9?袋.?;@A     s@I?袋.?;@a?踂?M??i??| u????Unknown
?	HostMatMul"2gradient_tape/sequential_14/dense_31/MatMul/MatMul(1     @r@9?.?袋:@A     @r@I?.?袋:@aS????9??i??0O\G???Unknown
?
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      r@9/?袋.:@A      r@I/?袋.:@a???????i!?g??????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(
1     ?l@9?????7@A     ?i@I     ?4@a??0???iH8??????Unknown
`HostGatherV2"
GatherV2_1(1     ?e@9t?E]t/@A     ?e@It?E]t/@au?s??|?ig\a$???Unknown
sHostSoftmax"sequential_14/dense_31/Softmax(1     ?b@9]t?E+@A     ?b@I]t?E+@a????B?x?i?J??U???Unknown
uHostMul"$sequential_14/dropout_13/dropout/Mul(1     ?a@9??.???)@A     ?a@I??.???)@a@n?~oew?i????????Unknown
cHostDataset"Iterator::Root(1     @n@9      6@A     ?`@I/?袋.(@a????v?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      `@9F]t?E'@A      `@IF]t?E'@avH?7=u?i?g?h<????Unknown
?HostGreaterEqual"-sequential_14/dropout_13/dropout/GreaterEqual(1     ?^@9]t?E]&@A     ?^@I]t?E]&@adJ??ht?iev???Unknown
lHostIteratorGetNext"IteratorGetNext(1      ]@9]t?E%@A      ]@I]t?E%@a?Xy?z?s?ii7?*???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1     @[@9     @+@A     @[@I     @+@acg??!r?i??KH?N???Unknown
wHostCast"%sequential_14/dropout_13/dropout/Cast(1     ?X@9      "@A     ?X@I      "@a=?i-Ymp?ip???o???Unknown
wHostMul"&sequential_14/dropout_13/dropout/Mul_1(1     ?X@9      "@A     ?X@I      "@a=?i-Ymp?i?? ?n????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1     @X@9??.???!@A     @X@I??.???!@aФNdp?iD ?u?????Unknown
ZHostArgMax"ArgMax(1     ?V@9?.?袋 @A     ?V@I?.?袋 @a?j`3n?iˊ???????Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1     ?c@9?袋.?,@A     ?V@I?.?袋 @a?j`3n?iR?]?????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_14/dense_31/BiasAdd/BiasAddGrad(1     @V@9/?袋. @A     @V@I/?袋. @a0FС!?m?i?????
???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_14/dense_30/BiasAdd/BiasAddGrad(1      V@9       @A      V@I       @a?%??,4m?i?H???'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      T@9]t?E@A      T@I]t?E@aT"ȅ?j?i?b?`OB???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1     @S@9      @A     @S@I      @a
?2*??i?i????[???Unknown
?HostReadVariableOp",sequential_14/dense_31/MatMul/ReadVariableOp(1      Q@9?袋.?@A      Q@I?袋.?@a.?|P?f?i>nr???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1     ?P@9     ? @A     ?P@I     ? @a?|/q<f?i?Av)?????Unknown
?HostReluGrad"-gradient_tape/sequential_14/dense_30/ReluGrad(1     @P@9??.???@A     @P@I??.???@a?;??,?e?i??(V<????Unknown
? HostMul"4gradient_tape/sequential_14/dropout_13/dropout/Mul_2(1      P@9F]t?E@A      P@IF]t?E@avH?7=e?i??y????Unknown
|!HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1     ?N@9/?袋.@A     ?N@I/?袋.@a-?`5Y>d?i?1??????Unknown
u"HostFlushSummaryWriter"FlushSummaryWriter(1      M@9      M@A      M@I      M@a?Xy?z?c?i%??a?????Unknown?
V#HostSum"Sum_2(1      M@9]t?E@A      M@I]t?E@a?Xy?z?c?i~r`?6????Unknown
?$HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      J@9颋.??@A      J@I颋.??@aP??[?Aa?i??x????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1     ?H@9?E]t?@A     ?H@I?E]t?@a5ý?B`?iI?yx????Unknown
?&HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      H@9t?E]t@A      H@It?E]t@a1)????_?i^VXb????Unknown
`'HostDivNoNan"
div_no_nan(1      H@9t?E]t@A      H@It?E]t@a1)????_?is?6L?/???Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1     ?F@9]t?E]@A     ?F@I]t?E]@a?f??]?i&[wW?>???Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_1(1      F@9      @A      F@I      @a?%??,4]?i???m M???Unknown
?*HostReadVariableOp"-sequential_14/dense_30/BiasAdd/ReadVariableOp(1      E@9?.?袋@A      E@I?.?袋@a?NEY?[?iD{?[???Unknown
?+HostMul"2gradient_tape/sequential_14/dropout_13/dropout/Mul(1     ?D@9?E]t?@A     ?D@I?E]t?@a/c??o6[?i=?>ҫh???Unknown
i,HostWriteSummary"WriteSummary(1     ?A@9     ?A@A     ?A@I     ?A@a	??:W?i?)?LIt???Unknown?
?-HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      >@9      ??A      >@I      ??a??Vd?S?iy3??=~???Unknown
m.HostReadVariableOp"ReadVariableOp(
1      >@9      @A      >@I      @a??Vd?S?iF=?2????Unknown?
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      =@9]t?E@A      =@I]t?E@a?Xy?z?S?i??gnґ???Unknown
X0HostCast"Cast_2(1      =@9]t?E@A      =@I]t?E@a?Xy?z?S?i???+r????Unknown
X1HostEqual"Equal(1      =@9]t?E@A      =@I]t?E@a?Xy?z?S?iJs??????Unknown
b2HostDivNoNan"div_no_nan_1(1      =@9]t?E@A      =@I]t?E@a?Xy?z?S?i?/K??????Unknown
?3HostReadVariableOp"-sequential_14/dense_31/BiasAdd/ReadVariableOp(1      :@9颋.??@A      :@I颋.??@aP??[?AQ?iA??R????Unknown
s4HostReadVariableOp"SGD/Cast/ReadVariableOp(1      9@9/?袋.@A      9@I/?袋.@atU?ӗP?il??n?????Unknown
?5HostReadVariableOp",sequential_14/dense_30/MatMul/ReadVariableOp(1      8@9t?E]t@A      8@It?E]t@a1)????O?ivȶc?????Unknown
w6HostDataset""Iterator::Root::ParallelMapV2::Zip(1     ?d@9??????K@A      6@IUUUUUU@a?%??,4M?i?i?n?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1      5@9?.?袋??A      5@I?.?袋??a?NEY?K?i??8??????Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      5@9?.?袋??A      5@I?.?袋??a?NEY?K?i????????Unknown
X9HostCast"Cast_3(1      4@9]t?E??A      4@I]t?E??aT"ȅ?J?i??u????Unknown
u:HostReadVariableOp"div_no_nan/ReadVariableOp(1      .@9?E]t???A      .@I?E]t???a??Vd?C?i ?p????Unknown
?;HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      .@9?E]t???A      .@I?E]t???a??Vd?C?i? 'oj????Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      *@9颋.????A      *@I颋.????aP??[?AA?i?~޺????Unknown
e=Host
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@aT"ȅ?:?i?7o????Unknown?
y>HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      "@9/?袋.??A      "@I/?袋.??a?????7?i??	????Unknown
a?HostIdentity"Identity(1       @9F]t?E??A       @IF]t?E??avH?7=5?i?-??????Unknown?
T@HostMul"Mul(1       @9F]t?E??A       @IF]t?E??avH?7=5?i??YX????Unknown
wAHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9F]t?E??A       @IF]t?E??avH?7=5?i?????????Unknown2CPU