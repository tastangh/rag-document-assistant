## Sayfa 1

ASHEN: Automated Surveillance and Hostile Intent Evaluation Network — A
Multimodal Transformer Framework for Weakly Supervised Video Anomaly
Detection
Mehmet Taştana,b,∗, Hamza Osman İlhana
aDepartment of Computer Engineering, Faculty of Electrical and Electronics, Yıldız Technical University, 34220 Esenler,
Istanbul, Turkey
bTurkish Aerospace Industries Inc.
Abstract
Video anomaly detection (VAD) in surveillance systems is a critical task for public safety, yet it remains
challenging due to the scarcity and diversity of anomalous events, the cost of frame-level annotation, and
the semantic complexity of real-world hostile activities. To address these limitations, we propose the Au-
tomated Surveillance and Hostile Intent Evaluation Network (ASHEN), a multi–modal weakly supervised
framework that unifies vision–language representations with multi–modal fusion and temporal modeling.
ASHEN combines CLIP ViT-B/32 features as a semantic backbone and augments them with YOLOv8
object detection and human–object interaction (HOI) descriptors through a Gated Multi–modal Fusion
mechanism that learns adaptive softmax-weighted combinations of each modality. The proposed framework
achieves state-of-the-art performance on both binary and multiclass video anomaly classification using only
video-level labels. On the dedicated Ashen Dataset, ASHEN attains a binary AUC of 98.09%, a nine-class
classification accuracy of 80.83%, and a Macro-AUC of 95.69%. These results indicate that ASHEN ex-
hibits strong discriminative capability and suggests substantial promise for proactive threat identification
and monitoring in critical infrastructure environments.
Keywords:
Video Anomaly Detection, Weakly Supervised Learning, Multiple Instance Learning,
Contrastive Language-Image Pretraining, Multi–modal Fusion, Temporal Transformer, Intelligent
Surveillance
1. Introduction
1
The rapid expansion of surveillance camera sys-
2
tems across urban environments, transportation
3
networks, and critical infrastructure has generated
4
a massive volume of video data that far exceeds
5
the capacity of human operators to monitor in real
6
time. While the number of deployed cameras has
7
surpassed hundreds of millions worldwide, the vast
8
majority of recorded footage is never reviewed, ren-
9
dering continuous manual monitoring both imprac-
10
tical and economically infeasible. This disparity has
11
motivated significant research interest in automated
12
∗Corresponding author.
Email addresses: mehmet.tastan@std.yildiz.edu.tr
(Mehmet Taştan), hoilhan@yildiz.edu.tr (Hamza Osman
İlhan)
Video Anomaly Detection (VAD) (1; 2; 3), which
13
seeks to identify unusual, suspicious, or potentially
14
dangerous events in video streams without exhaus-
15
tive human oversight. VAD presents fundamental
16
challenges: anomalous events are inherently rare,
17
highly diverse in appearance and context, and often
18
ambiguous in interpretation. An event considered
19
anomalous in one setting—such as running in a hos-
20
pital corridor—may be entirely normal in another,
21
such as a sports arena. Consequently, developing
22
robust and generalizable anomaly detection systems
23
remains one of the most pressing open problems in
24
computer vision and intelligent surveillance.
25
Early approaches to VAD relied on fully super-
26
vised learning, which requires dense frame-level an-
27
notations indicating the precise temporal extent of
28
each anomalous event. While effective in controlled
29

## Sayfa 2

settings, the cost and subjectivity of frame-level la-
30
beling render this paradigm impractical at scale.
31
A transformative shift occurred with the seminal
32
work of Sultani et al., who introduced the weakly
33
supervised formulation using only video-level la-
34
bels (normal or anomaly) within a Multiple In-
35
stance Learning (MIL) framework (4; 5). In this
36
paradigm, each video is treated as a bag of tem-
37
poral segments; the model learns to identify the
38
most anomalous instances within each bag with-
39
out explicit segment-level supervision. This formu-
40
lation dramatically reduces annotation cost while
41
preserving discriminative capacity. Subsequent re-
42
search has built extensively upon this MIL-based
43
framework. Feature representation has been dom-
44
inated by motion-centric extractors: 3D Convolu-
45
tional Networks (C3D) (6) learns spatiotemporal
46
convolutions over short video clips, while Inflated
47
3D Convolutional Networks (I3D) (7), pretrained
48
on the Kinetics dataset (8), inflates 2D convolu-
49
tional filters into 3D, capturing richer temporal dy-
50
namics via two-stream processing with optical flow
51
(9).
Building on these representations, methods
52
such as Tian et al. (10) introduced robust temporal
53
feature magnitude learning, Feng et al. (11) pro-
54
posed multi-instance self-training to progressively
55
refine pseudo-labels, Li et al. (12) employed multi-
56
sequence learning for improved temporal context,
57
and Zhong et al. (13) modeled inter-segment rela-
58
tionships as graph structures.
59
Despite the success of MIL-based weakly su-
60
pervised methods, a fundamental limitation per-
61
sists in the utilized feature representations. Both
62
I3D and C3D are designed to capture low-level
63
spatiotemporal patterns—such as motion trajec-
64
tories, optical flow gradients, and local texture
65
changes—but they fundamentally lack the ability
66
to perform high-level semantic reasoning, a limi-
67
tation commonly referred to as the semantic gap.
68
The semantic gap denotes the mismatch between
69
pixel-level feature representations and the abstract,
70
context-sensitive understanding needed to differ-
71
entiate events that look similar visually but dif-
72
fer in their underlying meaning.
For instance, a
73
friendly handshake and the initiation of a robbery
74
may share nearly identical motion signatures, yet
75
they carry diametrically opposed semantic mean-
76
ings.
Conventional motion-based features cannot
77
reliably resolve such ambiguities. The emergence
78
of Vision-Language Models (VLMs), particularly
79
CLIP (14), has opened a promising direction for
80
bridging this semantic gap.
CLIP, pretrained on
81
400 million image–text pairs via contrastive learn-
82
ing, encodes visual content into a shared embed-
83
ding space with natural language descriptions, en-
84
riching its representations with extensive seman-
85
tic structure.
Recent works have begun explor-
86
ing CLIP features for VAD: CLIP-TSA (15) ap-
87
plies temporal self-attention over CLIP features,
88
TPWNG (16) combines text prompts with visual
89
features for weakly supervised detection, VadCLIP
90
(17) proposes dual-branch architectures leveraging
91
CLIP’s visual and textual encoders, and WSVAD-
92
CLIP (18) introduces comprehensive CLIP-based
93
pipelines for weakly supervised anomaly detection.
94
While these CLIP-based approaches represent a
95
significant advance, three critical research gaps re-
96
main unaddressed. First, many recent methods in-
97
troduce excessive architectural complexity on top of
98
pretrained CLIP features—including learnable text
99
prompts, dual-branch decoders, contrastive align-
100
ment heads, and multi-stage training procedures—
101
which paradoxically undermines the principal ad-
102
vantage of using powerful pretrained representa-
103
tions: simplicity and transfer efficiency. The ques-
104
tion of how much added complexity is truly neces-
105
sary, versus how much is architectural overhead, re-
106
mains largely unexplored. Second, the overwhelm-
107
ing majority of weakly supervised VAD methods
108
are restricted to binary classification (normal vs.
109
anomaly) and do not attempt detailed categoriza-
110
tion of anomaly types. From a practical surveillance
111
perspective, knowing what type of anomaly has oc-
112
curred (e.g., arson, assault, or robbery) is far more
113
actionable than a simple binary alert, yet multi-
114
class anomaly categorization under weak supervi-
115
sion remains a largely uncharted territory. Third,
116
current benchmarks exhibit significant limitations
117
for hostile-intent surveillance research. The widely
118
used UCF-Crime dataset (5), while valuable, in-
119
cludes categories such as Shoplifting, Road Acci-
120
dent, and Stealing that represent non-violent or ac-
121
cidental events rather than deliberate hostile activ-
122
ities. No curated benchmark specifically targeting
123
hostile-intent anomaly categories exists, hindering
124
focused evaluation and progress in this critical ap-
125
plication domain.
126
To address these gaps, we propose the Au-
127
tomated
Surveillance
and
Hostile
Intent
128
Evaluation Network (ASHEN), a multi–modal
129
weakly supervised framework for both binary and
130
multiclass video anomaly detection. The principal
131
contributions of this work are as follows:
132
2

## Sayfa 3

1. We propose ASHEN, a multi–modal weakly
133
supervised framework that combines CLIP
134
ViT-B/32 semantic features with YOLOv8
135
(19) object detection features and Human–
136
Object
Interaction
(HOI)
(20)
descriptors
137
through a Gated Multi–modal Fusion mech-
138
anism, enabling both binary and multiclass
139
video anomaly detection under video-level su-
140
pervision only.
141
2. We introduce a Gated Multi–modal Fusion
142
mechanism
that
learns
adaptive
softmax-
143
weighted combinations of heterogeneous fea-
144
ture streams (CLIP 512-d, YOLOv8 256-d,
145
HOI 256-d), enabling the model to dynamically
146
balance semantic, object-level, and interaction-
147
level information based on input content.
148
3. We present a Pure Temporal Transformer ar-
149
chitecture (21) (dmodel = 256, 8 attention
150
heads, 3 layers) with learnable positional en-
151
coding that captures long-range temporal de-
152
pendencies across video segments without re-
153
lying on recurrent structures.
154
4. We curate the Ashen Dataset,
a focused
155
surveillance benchmark comprising 1600 videos
156
across 9 classes (Normal and 8 hostile-intent
157
anomaly categories: Arson, Assault, Burglary,
158
Explosion, Fighting, Robbery, Shooting, and
159
Vandalism), derived from UCF-Crime with
160
rigorous class restructuring and a balanced
161
70/15/15 train/validation/test split.
162
5. We report strong results, achieving 98.09%
163
binary AUC on the created Ashen Dataset
164
and 96.22% video-level binary AUC on UCF-
165
Crime with the CLIP baseline.
In addition,
166
we obtain 80.83% nine-class classification ac-
167
curacy using a CLIP+YOLOv8+HOI triple-
168
fusion model, and 95.69% nine-class macro-
169
AUC with a CLIP+YOLOv8+Text trimodal
170
fusion setup. These findings are further sup-
171
ported by extensive ablation studies that verify
172
the contribution of each architectural compo-
173
nent.
174
The remainder of this paper is organized as
175
follows:
Section 2 reviews the related work on
176
weakly supervised video anomaly detection, vision-
177
language models, and the trend toward architec-
178
tural simplicity.
Section 3 describes the datasets
179
used in this study, including the curation process
180
for the Ashen Dataset.
Section 3.2 presents the
181
proposed ASHEN framework in architectural detail,
182
covering feature extraction, multi–modal fusion,
183
temporal modeling, and the training procedure.
184
Section 4 reports comprehensive experimental re-
185
sults, ablation studies, comparisons with state-of-
186
the-art methods, and a real-world case study on
187
the 2024 TUSAŞ terror attack (? ). Finally, Sec-
188
tion 5 concludes the paper and outlines directions
189
for future work.
190
2. Related Work
191
2.1. Weakly Supervised Video Anomaly Detection
192
The
weakly
supervised
paradigm
for
video
193
anomaly detection was formalized by Sultani et al.,
194
who introduced both the UCF-Crime benchmark
195
and the Multiple Instance Learning (MIL) frame-
196
work (4) which has subsequently emerged as the
197
dominant training strategy in this area (22; 5).
198
Prior to this, unsupervised approaches such as tem-
199
poral regularity learning (23) and future frame pre-
200
diction (?
)
dominated the field, but their re-
201
liance on reconstruction error limited their ability
202
to handle the diversity of real-world anomalies. Un-
203
der MIL, each untrimmed video is treated as a bag
204
of temporal segments (instances), with only video-
205
level labels available during training: a positive bag
206
contains at least one anomalous instance, while a
207
negative bag consists entirely of normal instances.
208
Sultani et al. proposed a ranking loss with a hinge
209
formulation that maximizes the score margin be-
210
tween the highest-scoring instance in the positive
211
bag and the highest-scoring instance in the negative
212
bag, thereby learning to localize anomalies without
213
temporal annotations (5).
214
Early MIL-based approaches predominantly re-
215
lied on motion-centric feature extractors. Tran et
216
al. introduced 3D convolutional filters (C3D) that
217
capture spatiotemporal patterns from raw video
218
(6), extending the earlier two-stream paradigm
219
(9), while Carreira et al.
inflated pretrained
220
2D convolutional weights (I3D) into the tempo-
221
ral dimension and incorporated two-stream pro-
222
cessing with optical flow (7).
These representa-
223
tions served as the backbone for a series of pro-
224
gressively refined methods.
Tian et al.
pro-
225
posed robust temporal feature magnitude learning
226
(RTFM) to better distinguish anomalous segments
227
by their feature norms (10). Feng et al. introduced
228
multi-instance self-training (MIST) with progres-
229
sive pseudo-label refinement, allowing the model
230
to iteratively improve its segment-level predictions
231
(11). Li et al. leveraged multi-sequence learning
232
3

## Sayfa 4

(MSL) with Transformer-based attention to cap-
233
ture long-range temporal dependencies across seg-
234
ments (12). Zhong et al. formulated the noisy label
235
problem inherent in MIL as a graph convolution
236
task (GCN), proposing a GCN-based label noise
237
cleaner (13).
Zhou et al.
introduced dual mem-
238
ory units (UR-DMU) with uncertainty regulation to
239
better model the distribution of both normal and
240
anomalous features (24). Chen et al. proposed a
241
magnitude-contrastive glance-and-focus mechanism
242
(MGFN) that explicitly models the feature magni-
243
tude distribution to improve discriminability (25).
244
Despite
their
progressive
refinements,
these
245
methods share a fundamental limitation: their re-
246
liance on motion-centric features creates a seman-
247
tic gap. Representations from C3D and I3D encode
248
spatiotemporal dynamics but lack the capacity to
249
distinguish visually similar events that carry dif-
250
ferent semantic meanings—for instance, a person
251
running in a sports context versus fleeing from a
252
crime scene. This semantic gap has motivated the
253
adoption of vision-language models that embed vi-
254
sual content within a linguistically grounded feature
255
space.
256
2.2. Vision-Language Models in Video Anomaly
257
Detection
258
The introduction of Contrastive Language-Image
259
Pretraining (CLIP) by Radford et al.
marked
260
a paradigm shift in visual representation learning
261
(14). Trained on approximately 400 million image-
262
text pairs through contrastive learning, CLIP pro-
263
duces embeddings that encode not only visual ap-
264
pearance but also semantic meaning aligned with
265
natural language descriptions.
This property di-
266
rectly addresses the semantic gap identified in
267
motion-centric features: CLIP representations can
268
intrinsically differentiate between visually similar
269
scenes that carry distinct semantic content, mak-
270
ing them particularly suitable for anomaly detec-
271
tion tasks where contextual understanding is criti-
272
cal.
273
The adoption of CLIP in video anomaly detec-
274
tion has progressed rapidly. Joo et al. was among
275
the first to apply temporal self-attention (CLIP-
276
TSA) over CLIP visual features for anomaly scor-
277
ing, though it excluded the text encoder entirely
278
and thus underexploited the cross-modal potential
279
(15). Yang et al. extended this line by introduc-
280
ing learnable text prompts (TPWNG) combined
281
with a normality guidance mechanism, employing
282
dual ranking losses and a Distribution Inconsis-
283
tency Loss to refine anomaly boundaries (16). Wu
284
et Al. proposed a dual-branch architecture (Vad-
285
CLIP) that leverages both CLIP encoders, process-
286
ing visual and textual streams in parallel to cap-
287
ture complementary cues (17). Concurrently, Lv et
288
al. updated the UR-DMU framework with CLIP
289
features in an unbiased MIL formulation, demon-
290
strating that simply replacing I3D with CLIP fea-
291
tures yields substantial performance gains (26). Pu
292
et al. explored language-guided open-world video
293
anomaly detection (LaGoVAD), enabling zero-shot
294
detection of novel anomaly categories via natural
295
language descriptions (27).
Most recently, Li et
296
al. proposed a comprehensive pipeline (WSVAD-
297
CLIP) incorporating Axial-Graph modules and a
298
Lightweight Graph Attention Network (LiteGAT)
299
with Gumbel Softmax selection to refine CLIP fea-
300
tures for anomaly detection (18).
301
These
developments
collectively
demonstrate
302
that CLIP features provide a substantially richer
303
foundation for video anomaly detection than tra-
304
ditional motion-centric alternatives.
However, as
305
the next subsection discusses, the architectural
306
additions introduced to process these features
307
have grown increasingly complex, raising questions
308
about whether such complexity is necessary.
309
2.3. Architectural Complexity and the Case for
310
Simplicity
311
While the methods reviewed above have ad-
312
vanced the state of the art, a closer examina-
313
tion reveals a consistent trend toward architec-
314
tural complexity that warrants scrutiny.
TP-
315
WNG (16) augments its pipeline with dual rank-
316
ing losses, a Distribution Inconsistency Loss, and
317
learnable text prompts requiring careful initializa-
318
tion. WSVAD-CLIP (18) introduces Axial-Graph
319
modules for spatial reasoning, a LiteGAT with
320
Gumbel Softmax for adaptive feature selection,
321
and a multi-stage training procedure. MGFN (25)
322
adds magnitude-contrastive mechanisms with fea-
323
ture magnitude separation constraints.
Each ad-
324
dition improves benchmark numbers, yet the cu-
325
mulative effect raises three concerns: (1) increased
326
computational and memory overhead that limits
327
deployment in real-time surveillance systems, (2) el-
328
evated risk of overfitting on smaller or domain-
329
specific datasets where the training signal is insuffi-
330
cient to support highly parameterized modules, and
331
(3) diminished transfer efficiency of CLIP, whose
332
pretrained representations may already capture the
333
4

## Sayfa 5

discriminative information that these modules at-
334
tempt to learn from scratch.
335
This raises a central question: is the extra ar-
336
chitectural complexity truly required, or is CLIP’s
337
semantic space already sufficiently expressive to en-
338
able effective anomaly detection with only min-
339
imal post-processing?
The Vision Transformer
340
(ViT) backbone Dosovitskiy et al. underlying CLIP
341
already encodes rich hierarchical visual features
342
through self-attention proposed by Vaswani et al.
343
(28; 21). The standard Transformer architecture,
344
which has proven remarkably effective across natu-
345
ral language processing, computer vision, and mul-
346
timodal tasks, offers a principled mechanism for
347
temporal modeling without task-specific architec-
348
tural inventions.
349
Beyond the architectural dimension, the existing
350
literature exhibits two additional gaps that moti-
351
vate this work.
First, the overwhelming major-
352
ity of weakly supervised methods focus exclusively
353
on binary anomaly detection—classifying each seg-
354
ment as normal or anomalous—without attempting
355
to categorize the type of anomaly.
In real-world
356
surveillance applications, knowing that an anomaly
357
involves an assault rather than vandalism carries
358
significant operational implications for response pri-
359
oritization and resource allocation.
Second, the
360
standard benchmark, UCF-Crime, includes cate-
361
gories such as road accidents, shoplifting, and abuse
362
that, while anomalous, do not constitute deliberate
363
hostile intent. No existing benchmark offers a fo-
364
cused evaluation on hostile-intent categories specif-
365
ically relevant to public safety.
366
The ASHEN framework proposed in this pa-
367
per concurrently addresses all three of the pre-
368
viously identified research gaps.
Architecturally,
369
it embraces Occam’s Razor by channeling CLIP’s
370
powerful 512-dimensional features directly into a
371
pure Temporal Transformer Vaswani et al. with-
372
out learnable text prompts, graph neural networks,
373
or multi-stage training procedures (21). The model
374
employs a gated multi–modal fusion mechanism
375
to optionally incorporate complementary modali-
376
ties (YOLOv8 object detection Jocher et al. and
377
human-object interaction features) and applies Fo-
378
cal Loss Lin et al.
combined with ranking loss
379
under the MIL framework (19; 29).
Function-
380
ally, ASHEN supports both binary anomaly detec-
381
tion and nine-class multiclass anomaly categoriza-
382
tion within the same architecture.
Furthermore,
383
we introduce the Ashen Dataset—a curated bench-
384
mark of 1600 surveillance videos focused exclusively
385
on hostile-intent anomaly categories—to provide a
386
more targeted evaluation protocol for public safety
387
applications.
388
To synthesize the trajectory of weakly super-
389
vised video anomaly detection discussed above, Ta-
390
ble 1 provides a comparative overview of key mile-
391
stones in the field.
As shown in the table, VAD
392
architectures have increasingly favored heavily pa-
393
rameterized and complex modules while remain-
394
ing largely confined to binary classification.
In
395
contrast, our proposed ASHEN framework adopts
396
a streamlined Temporal Transformer architecture
397
rather than convoluted multi-stage designs.
This
398
design choice preserves efficiency while extending
399
VAD from coarse binary detection to fine-grained
400
nine-class anomaly categorization.
The following
401
section details the architectural design, multimodal
402
fusion mechanism, and mathematical formulation
403
of the ASHEN framework.
404
3. Materials and Methods
405
3.1. Dataset Information
406
From a hostile-intent surveillance perspective,
407
the original UCF-Crime class taxonomy includes
408
non-hostile, accidental, or contextually ambigu-
409
ous
categories
(e.g.,
RoadAccidents,
Shoplift-
410
ing/Stealing, Abuse, and Arrest), which can weaken
411
evaluations centered on hostile-threat detection.
412
Therefore, the UCF-Crime dataset was re-curated
413
to retain only classes relevant to public institution
414
security. Specifically, classes representing terrorist
415
acts or actions likely to cause severe institutional
416
disruption were preserved, while the remaining
417
classes were removed. In addition, to reduce post-
418
filtering imbalance among anomaly classes, extra
419
videos were incorporated into the retained classes.
420
To evaluate the generalizability of the proposed
421
method and enable direct comparison with the lit-
422
erature, experimental analyses were conducted on
423
two different datasets.
424
The UCF-Crime dataset, introduced by (5), is
425
the de facto standard benchmark for weakly su-
426
pervised video anomaly detection (WSVAD). It
427
comprises 1900 untrimmed real-world surveillance
428
videos spanning 13 anomaly classes along with a
429
Normal class. Table 2 presents the per-class distri-
430
bution. The dataset contains a total of 950 anomaly
431
videos and 950 normal videos, providing a balanced
432
binary split.Although the UCF-Crime dataset ap-
433
pears balanced in terms of video count (950 nor-
434
5

## Sayfa 6

mal and 950 anomalous), there is a clear imbal-
435
ance at the frame level.
Normal videos contain
436
approximately 10 million frames, whereas anoma-
437
lous videos contain about 4 million frames in total;
438
this can increase exposure to normal samples dur-
439
ing training and make it harder to learn anomaly
440
representations.
441
However, a notable class imbalance exists within
442
the anomaly categories. RoadAccidents and Rob-
443
bery each contain 150 videos, while Arson, Assault,
444
Explosion, Fighting, Shooting, and Vandalism con-
445
tain only 50 videos each.
This three-fold dispar-
446
ity creates a long-tailed distribution that can bias
447
model training toward overrepresented categories.
448
To address the limitations identified in UCF-
449
Crime for hostile-intent surveillance research, we
450
introduce the Ashen Dataset—a curated bench-
451
mark of 1600 surveillance videos focused exclusively
452
on hostile-intent anomaly categories.
The Ashen
453
Dataset is derived from UCF-Crime through a prin-
454
cipled curation process involving three steps: (i)
455
removal of non-hostile classes, (ii) class balancing
456
through supplementary data collection, and (iii) re-
457
duction of overrepresented classes.
458
First, five classes that do not align with the
459
hostile-intent focus were entirely removed: Abuse,
460
Arrest, RoadAccidents, Shoplifting, and Stealing.
461
These categories were excluded because they rep-
462
resent non-violent property crimes, accidental in-
463
cidents, or ambiguous scenarios that do not fall
464
within the scope of deliberate hostile activities tar-
465
geted by our study.
466
Second, for the six classes that contained only 50
467
videos in UCF-Crime (Arson, Assault, Explosion,
468
Fighting, Shooting, and Vandalism), we collected
469
50 additional surveillance videos per class from pub-
470
licly available online sources to bring each class
471
to 100 videos. This supplementary collection was
472
conducted with careful quality control: each video
473
was manually verified to contain the target anomaly
474
type and to originate from real surveillance camera
475
footage. Robbery, originally containing 150 videos,
476
was reduced to 100 by random subsampling. Bur-
477
glary, already at 100 videos, required no modifi-
478
cation. The Normal class was reduced from 950 to
479
800 videos through random subsampling, yielding a
480
balanced binary distribution of 800 anomaly videos
481
and 800 normal videos.
Table 3 summarizes the
482
complete transition from UCF-Crime to the Ashen
483
Dataset.
484
Despite this video-level balance, the frame dis-
485
tribution remains skewed toward the Normal class.
486
In UCF-Crime, of the 13,768,423 total frames,
487
10,123,292 belong to the Normal class,
while
488
3,645,131 belong to anomaly classes. In the Ashen
489
Dataset, the total number of frames is 11,961,524,
490
of which 9,474,387 are Normal and 2,487,137 corre-
491
spond to anomaly classes. The complete transition
492
from UCF-Crime to the Ashen Dataset is summa-
493
rized in Table 3, and the Ashen class/split distri-
494
bution is reported in Table 4.
495
The resulting ASHEN Dataset contains nine
496
classes: eight hostile-intent anomaly categories (Ar-
497
son, Assault, Burglary, Explosion, Fighting, Rob-
498
bery, Shooting, and Vandalism) and one Normal
499
class.
Each anomaly class contains exactly 100
500
videos, providing a perfectly balanced anomaly dis-
501
tribution that eliminates the class imbalance prob-
502
lem present in UCF-Crime.
This focused de-
503
sign makes the Ashen Dataset a targeted bench-
504
mark for hostile-intent anomaly detection and cat-
505
egorization, complementing the broader scope of
506
UCF-Crime rather than replacing it.To provide
507
qualitative visual evidence, Figure 1 presents one
508
clearly representative frame from each class (Ar-
509
son, Assault, Burglary, Explosion, Fighting, Rob-
510
bery, Shooting, Vandalism, and Normal).
511
A critical yet often overlooked issue in the VAD
512
literature is data leakage between training and eval-
513
uation sets. Some prior works either lack explicit
514
train–test separation at the video level or inadver-
515
tently share temporal segments from the same video
516
across splits, leading to inflated performance esti-
517
mates. To ensure rigorous and reproducible evalua-
518
tion, we adopt a strict video-level separation proto-
519
col where no video appears in more than one split.
520
The Ashen Dataset is partitioned into training,
521
validation, and test sets following an approximate
522
70/15/15 ratio using stratified random sampling
523
with a fixed random seed for full reproducibility.
524
The stratification ensures that the class propor-
525
tions are preserved across all three splits.
Ta-
526
ble 4 presents the exact per-class distribution across
527
splits.
528
Each anomaly class contributes exactly 70 videos
529
to training, 15 to validation, and 15 to test. The
530
Normal class follows the same proportional split,
531
yielding 557 training, 119 validation, and 120 test
532
videos. All hyperparameter tuning and model se-
533
lection decisions are made exclusively on the vali-
534
dation set; the test set is used only for final per-
535
formance reporting. This strict separation between
536
tuning and evaluation prevents information leakage
537
and ensures that reported metrics reflect genuine
538
6

## Sayfa 7

Figure 1: Representative surveillance frames from the nine classes in the Ashen Dataset (Arson, Assault, Burglary, Explosion,
Fighting, Robbery, Shooting, Vandalism, and Normal). The images are arranged in a 3×3 layout, and the corresponding class
label is shown at the top-right corner of each panel.
7

## Sayfa 8

Figure 2: Overview of the ASHEN pipeline. Video segments are processed through three parallel feature extraction branches
(CLIP, YOLOv8, HOI), projected to a common 256-dimensional space, fused via a learned softmax gate, and passed through
a Temporal Transformer for binary or multiclass classification.
generalization performance.
539
3.2. Proposed Methodology
540
This section presents the ASHEN (Automated
541
Surveillance and Hostile Intent Evaluation Net-
542
work) architecture in detail. The proposed system
543
comprises four principal stages: (1) multi–modal
544
feature extraction, (2) gated feature fusion, (3) tem-
545
poral modeling via Transformer encoders, and (4)
546
Multiple Instance Learning (MIL) based classifica-
547
tion. Figure 2 illustrates the overall pipeline, and
548
the following subsections describe each component
549
formally.
550
3.2.1. CLIP-Based Visual Feature Extraction
551
The primary visual representation is obtained us-
552
ing the CLIP (Contrastive Language–Image Pre-
553
training) model (14) with a Vision Transformer
554
(ViT-B/32) backbone (28).
Given a surveillance
555
video V, we uniformly sample one frame every
556
16 consecutive frames, yielding approximately 2
557
frames per second for standard 30 fps CCTV
558
footage.
Each sampled frame is passed through
559
the frozen CLIP visual encoder to produce a
560
512-dimensional embedding.
Importantly, no L2-
561
normalization is applied to the output vectors;
562
we preserve the raw activation magnitudes, where
563
∥x∥2 denotes the L2 (Euclidean) norm of the 512-
564
dimensional feature vector and is typically in the
565
range of 10–12. Preliminary experiments indicated
566
that enforcing normalization removes this magni-
567
tude cue and weakens discriminative capacity.
568
To accommodate videos of varying lengths, all
569
feature sequences are temporally interpolated to a
570
fixed length of T = 200 segments using linear in-
571
terpolation along the temporal axis. This yields a
572
8

## Sayfa 9

feature matrix Xclip ∈RT ×512 per video, provid-
573
ing a uniform temporal resolution for downstream
574
processing.
575
3.2.2. Multimodal Feature Streams
576
Although CLIP embeddings encode rich, high-
577
level visual semantics informed by linguistic pre-
578
training, effective surveillance anomaly detection
579
also relies on additional cues that CLIP by itself
580
may underrepresent. To this end, ASHEN incorpo-
581
rates three distinct feature modalities:
582
• CLIP features (Xclip ∈RT ×512): Semantic
583
visual embeddings that jointly encode visual
584
appearance and linguistic meaning through
585
contrastive pre-training on 400M image–text
586
pairs (14).
587
• YOLOv8
features
(Xyolo
∈
RT ×256):
588
Object
detection
descriptors
derived
from
589
YOLOv8 (19), whose lineage traces back to
590
the original YOLO architecture (30), encoding
591
object categories, instance counts, and spatial
592
positions within each frame.
These features
593
capture the object-level scene composition that
594
is critical for detecting anomalies such as bur-
595
glary (presence of tools) or robbery (weapons).
596
• HOI features (Xhoi ∈RT ×256):
Human–
597
Object Interaction descriptors (20; 31) that en-
598
code the relational structure between detected
599
persons and objects, capturing interaction-
600
level semantics such as person-holding-weapon
601
or person-breaking-window.
602
Each modality provides complementary informa-
603
tion: CLIP (Xclip) captures global scene semantics,
604
YOLO (Xyolo) supplies explicit object-level compo-
605
sition, and HOI (Xhoi) encodes interaction dynam-
606
ics. The fusion of these streams is described in the
607
following subsection.
608
3.3. Gated Multimodal Fusion
609
The three feature streams are fused via a learned
610
gating mechanism (32) that adaptively weights each
611
modality at every temporal position. This approach
612
draws on the broader literature on multimodal fu-
613
sion (33) and enables the network to emphasize the
614
most informative modality depending on the local
615
temporal context—for instance, assigning higher
616
weight to HOI features during interpersonal inter-
617
actions and to YOLO features when specific object
618
configurations are anomalous.
619
Each stream is first independently projected to
620
a common dimensionality dmodel = 256 via a lin-
621
ear layer followed by layer normalization (34) and
622
GELU activation (35):
623
hclip = GELU(LayerNorm(Wclip Xclip + bclip)) ,
hyolo = GELU(LayerNorm(Wyolo Xyolo + byolo)) ,
(1)
hhoi = GELU(LayerNorm(Whoi Xhoi + bhoi)) ,
where Wclip ∈R256×512 and Wyolo, Whoi ∈
624
R256×256.
625
The projected representations are concatenated
626
along the feature dimension to form a joint context
627
vector:
628
ct =
h
h(t)
clip; h(t)
yolo; h(t)
hoi
i
∈R768,
t = 1, . . . , T.
(2)
A
two-layer
gating
network
computes
per-
629
timestep softmax weights over the three modalities:
630
gt = Softmax(W2 GELU(W1 ct + b1) + b2) ∈R3,
(3)
where W1 ∈R128×768 and W2 ∈R3×128.
The
631
fused representation is computed as the gated com-
632
bination:
633
zt = g(1)
t
⊙h(t)
clip + g(2)
t
⊙h(t)
yolo + g(3)
t
⊙h(t)
hoi ∈R256,
(4)
where g(i)
t
denotes the i-th gate scalar at time t,
634
broadcast across the feature dimension, and ⊙de-
635
notes element-wise multiplication. The resulting se-
636
quence Z = [z1, . . . , zT ] ∈RT ×256 is passed to the
637
temporal modeling stage.
638
3.3.1. Temporal Transformer Architecture
639
Temporal dependencies across the T = 200 video
640
segments are modeled using a Transformer en-
641
coder (21). Two architectural variants are employed
642
depending on the task:
643
Unimodal/fused
variant
(binary
detec-
644
tion): A single Transformer encoder processes ei-
645
ther the fused multimodal features Z or unimodal
646
CLIP features projected to dmodel = 256.
647
Independent-stream
variant
(multiclass
648
classification): Each modality stream is processed
649
by its own dedicated Transformer encoder before
650
gated fusion.
The three encoded representations
651
are then fused and classified. This variant preserves
652
modality-specific temporal patterns that might oth-
653
erwise be obscured by early fusion.
654
9

## Sayfa 10

In both variants,
each Transformer encoder
655
shares the same configuration.
A learnable posi-
656
tional embedding is added to the input:
657
pt = ht + PEt,
t = 1, . . . , T,
(5)
where PE ∈RT ×dmodel is a learnable parameter ma-
658
trix that encodes absolute temporal position.
659
The encoder consists of L = 3 identical lay-
660
ers.
Each layer applies multi-head self-attention
661
(MHSA) followed by a position-wise feed-forward
662
network (FFN), with residual connections and layer
663
normalization:
664
Attention(Q, K, V) = Softmax
QK⊤
√dk

V,
(6)
where Q = PWQ, K = PWK, V = PWV are
665
the query, key, and value projections, respectively,
666
and dk = dmodel/nheads = 32. The model employs
667
nheads = 8 attention heads. The FFN uses GELU
668
activation with an expansion factor of 4:
669
FFN(x) = GELU(xW1 + b1)W2 + b2,
(7)
where W1 ∈R256×1024 and W2 ∈R1024×256.
670
Dropout (36) with rate p = 0.3 is applied within
671
each sublayer.
The architectural configuration is
672
summarized as: dmodel = 256, nheads = 8, L = 3,
673
dff = 1024.
674
3.4. MIL Aggregation and Classification Heads
675
Since video-level labels do not specify which tem-
676
poral segments contain anomalies, the system op-
677
erates under the Multiple Instance Learning (MIL)
678
paradigm (4; 5).
The Transformer encoder pro-
679
duces per-segment representations {ot}T
t=1, which
680
are mapped to scores or logits by a classification
681
head.
682
Binary classification: The classifier consists
683
of two fully connected layers (256 →128 →1)
684
with GELU activation, dropout, and a final sig-
685
moid. Video-level aggregation employs top-k pool-
686
ing: the k = max(1, ⌊T/16⌋) = 12 highest-scoring
687
segments are averaged to obtain the video-level
688
anomaly score:
689
svideo = 1
k
k
X
i=1
s(i),
where s(1) ≥s(2) ≥· · · ≥s(T ).
(8)
Top-k pooling focuses the gradient signal on the
690
most anomalous segments (37), which is critical un-
691
der weak supervision where only a small fraction of
692
an anomaly video may contain the actual event.
693
Multiclass classification: The classifier maps
694
to 9 classes via two fully connected layers (256 →
695
128 →9) with LeakyReLU (negative slope = 0.2)
696
and dropout; no softmax is applied as it is handled
697
internally by the cross-entropy loss. Video-level log-
698
its are obtained by mean pooling over all temporal
699
segments:
700
ℓvideo = 1
T
T
X
t=1
ℓt.
(9)
Mean pooling is preferred over top-k for multiclass
701
classification because each category requires holis-
702
tic temporal context rather than focusing on peak
703
activations alone.
704
3.5. Loss Functions
705
The training objective combines multiple loss
706
terms that jointly enforce discriminative scoring,
707
temporal regularity, and sparsity. We describe the
708
binary formulation first, then note multiclass exten-
709
sions.
710
Focal Loss: To address class imbalance between
711
normal and anomalous segments, we employ Focal
712
Loss (29), which down-weights well-classified exam-
713
ples and focuses learning on hard negatives:
714
Lfocal(pt) = −αt(1 −pt)γ log(pt),
(10)
where pt is the predicted probability for the ground-
715
truth class, α = 0.75 balances positive and negative
716
contributions, and γ = 2.0 controls the focusing
717
strength.
718
MIL Ranking Loss.
Following Sultani et
719
al. (5), the ranking loss enforces a margin between
720
the maximum predicted anomaly score and the
721
maximum normal score within each mini-batch:
722
Lrank = max

0, 1 −max
t (sanom
t
) + max
t (snorm
t
)

,
(11)
with margin m = 1.0. This constraint ensures that
723
the highest anomaly activation in a positive bag
724
exceeds the highest activation in a negative bag,
725
which is the foundational principle of MIL-based
726
anomaly detection.
727
Sparsity Regularization.
Since anomalous
728
events are typically brief, a sparsity penalty encour-
729
ages the model to activate on only a small number
730
of segments:
731
Lsparse = 1
T
T
X
t=1
|st|.
(12)
10

## Sayfa 11

Temporal Smoothness.
To prevent erratic
732
score fluctuations between adjacent segments, a
733
smoothness term penalizes large temporal gradi-
734
ents:
735
Lsmooth =
1
T −1
T −1
X
t=1
(st+1 −st)2.
(13)
Total Loss. The overall binary training objec-
736
tive is a weighted combination of all terms:
737
L = Lfocal+λrank Lrank+λsparse Lsparse+λsmooth Lsmooth,
(14)
where λrank = 0.1, λsparse = 0.001, and λsmooth =
738
0.001.
739
For the multiclass setting, the focal loss oper-
740
ates on the cross-entropy formulation with per-
741
class weights (wnormal = 0.2228, wanomaly = 1.7730
742
for each anomaly category).
The ranking loss is
743
adapted so that the maximum probability of the
744
correct anomaly class exceeds that of the Normal
745
class by the margin m. Label smoothing (38) with
746
ϵ = 0.05 is applied to soften one-hot targets and
747
improve calibration.
748
3.6. Training Strategy
749
All models are optimized using AdamW (39) with
750
an initial learning rate of 10−4 and weight decay of
751
5 × 10−4. The learning rate follows a Cosine An-
752
nealing with Warm Restarts schedule (40) (T0 = 10,
753
Tmult = 2), which periodically resets the learning
754
rate to escape local minima.
755
Data augmentation is performed via MixUp (41)
756
with interpolation coefficient α ∈[0.2, 0.4], applied
757
with 50% probability after a warmup period of 3
758
epochs. MixUp constructs virtual training exam-
759
ples by linearly interpolating both features and la-
760
bels:
761
˜x = λxi+(1−λ)xj,
˜y = λyi+(1−λ)yj,
λ ∼Beta(α, α).
(15)
This regularization strategy improves generaliza-
762
tion by encouraging linear behavior between train-
763
ing examples and reducing overconfident predic-
764
tions.
765
Additional
training
details
include:
gradi-
766
ent clipping with maximum norm of 0.5 for
767
multiclass models,
mixed-precision training via
768
torch.amp.autocast for memory efficiency, batch
769
sizes of 16–32, and early stopping with patience of
770
10 epochs monitored on validation loss. Training
771
proceeds for a maximum of 50 epochs.
772
4. Experiments and Results
773
4.1. Experimental Configuration
774
All experiments are implemented in PyTorch
775
and executed on a workstation equipped with
776
an NVIDIA GeForce RTX 5070 Ti Laptop GPU
777
(12 GB VRAM). Feature extraction employs the of-
778
ficial OpenAI CLIP ViT-B/32 visual encoder (14)
779
to produce 512-dimensional segment-level represen-
780
tations, with YOLOv8
(19) object features (256-
781
d) and HOI interaction features (256-d) added in
782
multimodal variants. Mixed-precision training via
783
torch.amp.autocast is used throughout to reduce
784
memory consumption and accelerate computation.
785
All experiments are fully reproducible under fixed
786
random seeds applied to PyTorch, NumPy, and
787
CUDA backends.
788
Table 5 summarizes the key hyperparameters
789
across three representative configurations. The Bi-
790
nary Baseline follows a lightweight two-layer Tem-
791
poral Transformer trained with standard BCE-
792
based MIL loss, whereas the Binary Optimized vari-
793
ant deepens the architecture to three layers and in-
794
troduces Focal Loss (29) with ranking loss, AdamW
795
optimiser (39), MixUp augmentation (41), and la-
796
bel smoothing.
The Multiclass configuration in-
797
herits the optimised training recipe and adapts it
798
for nine-class classification with cross-entropy or
799
focal loss variants, gradient clipping, and class-
800
dependent regularization.
801
For binary anomaly detection, we report the area
802
under the receiver operating characteristic curve
803
(AUC-ROC), accuracy, precision, recall, and F1-
804
score.
For multiclass classification, we addition-
805
ally compute the macro-averaged one-vs-rest AUC
806
(Macro-AUC), weighted AUC, and per-class AUC
807
to capture class-level discrimination across imbal-
808
anced anomaly categories.
All metrics are com-
809
puted on held-out test sets that are strictly isolated
810
from training and validation data.
811
4.2. Binary Anomaly Detection
812
We first evaluate ASHEN on the widely used
813
UCF-Crime benchmark (5) to position our frame-
814
work against the state of the art.
Table 6 com-
815
pares representative methods spanning C3D, I3D,
816
and CLIP-based feature backbones. We follow the
817
official Sultani test split (290 videos: 140 anomaly
818
across 13 classes, 150 normal) without modification.
819
Our CLIP-only baseline achieves a video-level AUC
820
11

## Sayfa 12

of 96.22%, computed by assigning a single MIL-
821
aggregated anomaly score per video and ranking
822
anomalous videos above normal ones.
823
The prior methods listed in Table 6 report frame-
824
level AUC, which evaluates temporal localization
825
by scoring individual frames against dense temporal
826
annotations. ASHEN, by contrast, operates under
827
a MIL paradigm that produces a single video-level
828
score and does not perform frame-level temporal
829
localization. These two evaluation granularities are
830
not directly comparable: video-level AUC measures
831
the ability to rank anomalous videos above normal
832
ones (a coarser but practically relevant task), while
833
frame-level AUC measures within-video temporal
834
discrimination (a finer-grained task). We therefore
835
present ASHEN’s result separately in the table and
836
caution against drawing direct numerical compar-
837
isons across evaluation levels. The table is provided
838
to contextualise ASHEN within the broader land-
839
scape of weakly supervised VAD methods rather
840
than to claim strict superiority.
841
Table 7 presents the binary detection results of
842
five ASHEN variants on our curated Ashen Dataset
843
(240 test videos, balanced 120/120 normal/anomaly
844
split). The CLIP-only Baseline, using merely a two-
845
layer Temporal Transformer with standard BCE-
846
based MIL loss, achieves the highest accuracy of
847
95.00% and an AUC of 98.09%. The CLIP+YOLO
848
dual-fusion model marginally improves AUC to
849
98.16% yet accuracy drops to 91.67%, indicating
850
that object-level features introduce specificity that
851
does not generalize uniformly across anomaly types.
852
The CLIP+HOI variant attains 97.54% AUC, con-
853
firming that human–object interaction cues encode
854
meaningful behavioral signals, albeit at the cost
855
of slightly lower precision than the baseline. The
856
Triple-fusion model (CLIP+YOLO+HOI) achieves
857
a balanced precision and recall of 0.9333 each,
858
demonstrating that the gated fusion mechanism ef-
859
fectively reconciles the complementary information
860
streams.
861
Two noteworthy observations emerge from the bi-
862
nary experiments.
First, the Optimised model—
863
which
employs
a
deeper
three-layer
architec-
864
ture,
Focal Loss,
ranking loss,
and extensive
865
regularization—does not surpass the simpler Base-
866
line on the balanced Ashen Dataset. This counter-
867
intuitive result suggests an over-regularization ef-
868
fect: techniques designed to handle class imbalance
869
(label smoothing, class weights) may introduce un-
870
necessary noise when the dataset is already bal-
871
anced, leading to marginal performance degrada-
872
tion. Second, while multimodal fusion consistently
873
yields AUC values above 97.5%, the accuracy–AUC
874
trade-off reveals that fusion models tend to favor
875
higher specificity (fewer false positives) at the ex-
876
pense of recall.
The CLIP+YOLO model exem-
877
plifies this with the highest precision (0.9717) but
878
the lowest recall (0.8583), whereas the Triple model
879
achieves the most balanced operating point. These
880
findings motivate our multiclass experiments (Sec-
881
tion 4.3), where richer feature combinations become
882
more critical for detailed anomaly discrimination.
883
Figure 3 presents the ROC curve for the ASHEN
884
Baseline on the Ashen Dataset, illustrating the
885
near-perfect discrimination between normal and
886
anomalous videos at an AUC of 98.09%. The curve
887
closely hugs the upper-left corner, confirming that
888
the model maintains high sensitivity across a wide
889
range of decision thresholds.
890
Figure 3: ROC curve for ASHEN Baseline (CLIP) on the
Ashen Dataset binary test set (AUC = 98.09%).
4.3. Multiclass Anomaly Classification
891
Moving beyond binary detection, we evaluate
892
ASHEN’s capacity to discriminate among nine se-
893
mantic categories (Normal plus eight hostile-intent
894
anomaly types).
Table 8 summarizes the over-
895
all performance of seven model variants on the
896
Ashen Dataset test set (240 videos:
120 nor-
897
mal, 15 per anomaly class).
The Triple-fusion
898
model (CLIP+YOLO+HOI) achieves the high-
899
est overall accuracy of 80.83% and a Macro-
900
F1
of
0.6797,
whereas
the
Trimodal
variant
901
(CLIP+YOLO+TEXT) attains the best Macro-
902
AUC of 95.69% and Weighted-AUC of 96.81%.
903
This divergence highlights a fundamental trade-off:
904
textual caption embeddings improve per-class dis-
905
crimination boundaries (higher AUC) by providing
906
12

## Sayfa 13

rich semantic context, while spatial object and in-
907
teraction features (YOLO+HOI) yield more cali-
908
brated top-1 predictions (higher accuracy).
909
Notably, the CLIP-only Baseline already pro-
910
vides competitive accuracy (78.75%) and the high-
911
est single-modality Macro-F1 (0.6807), demonstrat-
912
ing that CLIP’s pre-trained visual–semantic repre-
913
sentations capture substantial anomaly-relevant in-
914
formation. The Optimized variant marginally im-
915
proves accuracy to 79.17% but decreases Macro-
916
AUC to 91.68%, echoing the over-regularization ef-
917
fect observed in the binary setting.
The Class-
918
Weight variant performs worst overall (71.25% ac-
919
curacy, 86.60% Macro-AUC), indicating that ex-
920
plicit class reweighting under MIL training disrupts
921
the bag-level loss optimization landscape.
922
Table 9 details the per-class precision, recall, and
923
F1-score for the best-accuracy Triple-fusion model.
924
The model achieves strong performance on high-
925
motion anomalies such as Fighting (F1 = 0.84),
926
Explosion (F1 = 0.81), and Arson (F1 = 0.79),
927
which exhibit distinctive visual and spatial signa-
928
tures readily captured by the CLIP+YOLO+HOI
929
feature combination.
In contrast, Vandalism (F1
930
= 0.37) and Burglary (F1 = 0.40) remain chal-
931
lenging, as these categories involve subtle, context-
932
dependent actions that share visual characteristics
933
with normal activities. The Normal class achieves
934
the highest F1 (0.94), benefiting from its larger sup-
935
port size and more consistent visual patterns.
936
An interesting disparity between AUC and F1
937
metrics emerges across classes. For instance, Bur-
938
glary achieves a respectable per-class AUC of
939
0.9200, indicating good ranking capability, yet its
940
F1-score is only 0.40 due to low recall (0.33). This
941
suggests that while the model learns discriminative
942
representations for rare anomaly types, the deci-
943
sion threshold optimized for overall accuracy tends
944
to under-predict minority classes. This observation
945
motivates future work on class-adaptive threshold-
946
ing or cost-sensitive loss formulations.
947
Figure 4 presents the one-vs-rest ROC curves for
948
the Triple-fusion model across all nine classes. Ar-
949
son (AUC = 0.9861) and Normal (AUC = 0.9801)
950
exhibit near-perfect curves, while Assault (AUC
951
= 0.9040) and Explosion (AUC = 0.9046) show
952
slightly lower but still strong discrimination. The
953
consistently high per-class AUC values (all above
954
0.90) confirm that the model captures robust class-
955
specific decision boundaries even when top-1 pre-
956
dictions are imperfect.
957
Figure 5 shows the normalized confusion matrix
958
Figure 4:
One-vs-rest ROC curves for the Triple-fusion
model (CLIP+YOLO+HOI) on the Ashen Dataset. All per-
class AUC values exceed 0.90.
for the Triple-fusion model.
The dominant diag-
959
onal confirms strong classification accuracy, with
960
Normal videos correctly identified 98% of the time.
961
The most frequent misclassification patterns involve
962
Vandalism being confused with Normal (67% of
963
Vandalism test videos are mislabeled) and Burglary
964
being confused with Normal and Robbery, reflect-
965
ing the inherent visual similarity between stealthy
966
criminal activities. Arson, Explosion, and Fighting
967
form a relatively well-separated cluster with recall
968
values of 87%, 87%, and 87%, respectively.
969
Figure 5: Normalised confusion matrix for the Triple-fusion
model on the Ashen Dataset (9 classes).
Figure 6 visualizes the t-SNE projections of the
970
13

## Sayfa 14

learned segment-level embeddings from the penulti-
971
mate layer of the Triple-fusion model. Normal seg-
972
ments form a dense, well-separated cluster, while
973
anomaly classes exhibit varying degrees of overlap.
974
Fighting and Assault clusters partially overlap, con-
975
sistent with their shared motion dynamics, whereas
976
Arson and Explosion occupy distinct regions reflect-
977
ing their unique visual signatures (fire/smoke vs.
978
blast patterns). The clear inter-class separation in
979
the embedding space—despite being trained under
980
weak MIL supervision without frame-level labels—
981
validates the feature discrimination capability of
982
the gated multi-modal fusion approach.
983
Figure 6: t-SNE visualisation of segment-level embeddings
from the Triple-fusion model. Colours indicate ground-truth
class labels.
4.4. Ablation Study: Modality Contributions
984
To quantify the contribution of each modality
985
stream, we conduct a systematic ablation study
986
progressing from single-modality CLIP to increas-
987
ingly richer multimodal configurations.
Table 10
988
organises the results as an incremental modality ad-
989
dition path.
990
Several key patterns emerge from the ablation re-
991
sults. First, adding YOLO alone to CLIP does not
992
improve accuracy (78.33% vs. 78.75%) and slightly
993
reduces Macro-AUC (93.11% vs. 93.69%), suggest-
994
ing that raw object detection features introduce
995
noise without sufficient complementary signal for
996
multiclass discrimination. Second, adding HOI fea-
997
tures to CLIP yields a meaningful improvement in
998
Weighted-AUC (+1.18 pp) while maintaining accu-
999
racy, indicating that human–object interaction pat-
1000
terns encode behavioural cues that enhance class-
1001
level separation.
Third, the Triple configuration
1002
(CLIP+YOLO+HOI) achieves the best accuracy
1003
(+2.08 pp over baseline), demonstrating that the
1004
gated fusion mechanism can effectively reconcile po-
1005
tentially conflicting modality signals when all three
1006
spatial-semantic streams are available.
1007
The most striking finding is the Trimodal variant
1008
(CLIP+YOLO+TEXT), which achieves the high-
1009
est Macro-AUC (95.69%) despite slightly lower ac-
1010
curacy than the Triple model.
The text modal-
1011
ity, derived from frame-level caption embeddings
1012
generated by a vision–language model, provides
1013
category-level semantic anchors that improve the
1014
model’s ranking capability across all classes. The
1015
per-class AUC values of the Trimodal variant are
1016
consistently high (all above 0.92), with the largest
1017
gains observed for classes that benefit from descrip-
1018
tive context such as Robbery (0.9547 vs. 0.9212 in
1019
Triple) and Assault (0.9514 vs. 0.9040). These re-
1020
sults suggest that textual semantics complement vi-
1021
sual and spatial features by providing higher-level
1022
scene understanding that is less sensitive to visual
1023
ambiguity.
1024
Figure 7 shows the per-class AUC comparison
1025
across the five ablation configurations.
The bar
1026
chart reveals that fusion consistently improves AUC
1027
for interaction-heavy classes (Fighting, Robbery)
1028
while maintaining high AUC for visually distinctive
1029
classes (Arson, Normal). Burglary and Vandalism
1030
remain the most challenging categories across all
1031
configurations, yet even these benefit from multi-
1032
modal fusion compared to the CLIP-only baseline.
1033
Figure 7: Per-class AUC comparison across ablation config-
urations on the Ashen Dataset.
4.5. Cross-Dataset Analysis
1034
To assess generalisability, we replicate the mul-
1035
ticlass experiments on the UCF-Crime dataset (5),
1036
which presents a substantially more challenging set-
1037
ting:
14 classes (13 anomaly + Normal), severe
1038
class imbalance, and noisier annotations. Table 11
1039
compares the best-performing configuration from
1040
each dataset.
1041
14

## Sayfa 15

The performance gap between the two datasets
1042
is substantial: Ashen achieves 80.83% accuracy and
1043
94.01% Macro-AUC versus UCF-Crime’s 63.10%
1044
and 87.50% for the best respective models.
Sev-
1045
eral factors explain this discrepancy. First, Ashen
1046
was purpose-built for hostile-intent detection with
1047
clean, category-specific annotations, yielding a se-
1048
mantically coherent label space.
UCF-Crime, in
1049
contrast, contains diverse anomaly types (e.g.,
1050
“Shoplifting”, “Road Accidents”) with considerable
1051
intra-class variation and ambiguous boundaries.
1052
Second, the Ashen Dataset is balanced (equal Nor-
1053
mal/Anomaly split), whereas UCF-Crime is heavily
1054
skewed toward Normal videos (∼63% of test set),
1055
which depresses minority-class recall and Macro-F1.
1056
Third, Ashen’s 9-class taxonomy was specifically
1057
designed to capture surveillance-relevant hostile-
1058
intent categories, enabling the model to learn more
1059
focused decision boundaries.
1060
Notably, multimodal fusion on UCF-Crime does
1061
not yield the same consistent improvements ob-
1062
served on Ashen. The CLIP Baseline achieves the
1063
highest Macro-AUC (87.50%) on UCF-Crime, out-
1064
performing both dual and triple fusion variants.
1065
This suggests that when the label space is large
1066
and noisy, CLIP’s generalised representations may
1067
be more robust than task-specific object and inter-
1068
action features, which can overfit to the training
1069
distribution. This finding underscores the impor-
1070
tance of dataset quality and annotation consistency
1071
in multimodal learning, and validates the purpose-
1072
built Ashen Dataset as a more suitable benchmark
1073
for hostile-intent surveillance research.
1074
4.6. Real-Time Detection Demonstration
1075
To demonstrate practical deployment feasibility,
1076
we evaluate the inference latency of the ASHEN
1077
pipeline on a single NVIDIA GeForce RTX 5070 Ti
1078
Laptop GPU. The complete pipeline—feature ex-
1079
traction (CLIP + YOLO + HOI) followed by tem-
1080
poral classification—processes a 200-segment video
1081
clip in approximately 180 ms end-to-end, corre-
1082
sponding to a throughput of ∼5.5 videos per sec-
1083
ond.
Feature extraction dominates the computa-
1084
tional cost (∼150 ms for CLIP forward pass over
1085
200 frames), while the Temporal Transformer clas-
1086
sification head adds only ∼30 ms of overhead. This
1087
latency profile is well within real-time requirements
1088
for surveillance applications, where typical camera
1089
feeds operate at 25–30 fps and segment-level predic-
1090
tions (one prediction per 16-frame segment) require
1091
only ∼1.9 classifications per second.
1092
The training dynamics of the Triple-fusion model
1093
are illustrated in Figure 8. The training loss con-
1094
verges smoothly within the first 30 epochs, while
1095
validation accuracy reaches its peak around epoch
1096
35–40, after which early stopping prevents over-
1097
fitting.
The cosine annealing with warm restarts
1098
schedule introduces periodic learning rate increases
1099
that help the optimizer escape local minima, con-
1100
tributing to the robust final performance.
1101
Figure 8: Training dynamics of the Triple-fusion model: loss
curves, validation accuracy, and learning rate schedule.
4.7. Real-World Case Study: TUSAŞ Terror Attack
1102
On October 23, 2024, a terrorist attack struck
1103
the headquarters of Turkish Aerospace Industries
1104
(TUSAŞ) in Ankara, Turkey—one of the country’s
1105
most strategically significant defense facilities. Two
1106
armed assailants arrived by taxi, disembarked, and
1107
opened fire on on-duty security personnel before one
1108
of the attackers detonated a suicide vest near the fa-
1109
cility entrance. The attack resulted in the deaths of
1110
five civilians, including security guard Atakan Şahin
1111
Erdoğan, whose name serves as the phonetic inspi-
1112
ration for the ASHEN acronym. This tragic event
1113
underscored the urgent need for AI-powered early-
1114
warning surveillance systems at critical infrastruc-
1115
ture sites, and provided the foundational motiva-
1116
tion for the present research.
1117
To validate the practical applicability of ASHEN
1118
in such a scenario, we conducted inference on two
1119
surveillance video clips sourced from the attack in-
1120
cident. The first clip (tai-a, 1,291 frames, ∼43 s at
1121
30 fps) captures the initial moments of the assault:
1122
the taxi arrival, the disembarkation of the attack-
1123
ers, and the ensuing gunfire exchange with security
1124
15

## Sayfa 16

personnel.
The second clip (tai-n, normal con-
1125
trol) depicts a routine operational scene at the same
1126
facility—groups of workers gathered around mili-
1127
tary fighter jets on the apron—representing a visu-
1128
ally complex but entirely non-threatening scenario.
1129
Both clips were processed through the full ASHEN
1130
pipeline: CLIP ViT-B/32 feature extraction (16-
1131
frame segments), YOLOv8 object detection, and
1132
HOI interaction features, followed by binary and
1133
multiclass inference using the trained Transformer-
1134
MIL and MulticlassGatedTripleMIL models respec-
1135
tively.
1136
The binary detection model classified tai-a as
1137
ANOMALY with a confidence score of 0.9837,
1138
flagging the video within the first few temporal seg-
1139
ments. The segment-level anomaly score timeline
1140
(Figure 9) reveals that the model consistently as-
1141
signs high anomaly scores throughout the clip, with
1142
scores exceeding 0.95 from the earliest segments—
1143
indicating that the system would have raised an
1144
alert within seconds of the attack’s onset. In con-
1145
trast, tai-n was correctly classified as NORMAL
1146
with a remarkably low score of 0.0103, demon-
1147
strating that the model does not generate false
1148
alarms even in visually crowded scenes involving
1149
large groups of people and military equipment.
1150
Figure 9: Binary anomaly score timelines for the TUSAŞ
case study. Top: attack video (tai-a, score = 0.9837). Bot-
tom: normal operational scene (tai-n, score = 0.0103).
The multiclass analysis yields a particularly strik-
1151
ing finding. The model classified tai-a as Explo-
1152
sion with 58.61% probability, followed by Assault
1153
(11.35%) and Shooting (9.90%). This prediction is
1154
remarkable because the analyzed clip captures only
1155
the pre-detonation phase: the gunfire exchange and
1156
the armed confrontation, with no explosion visible
1157
in the footage. Yet the model assigned the high-
1158
est probability to Explosion—the event that was
1159
about to occur moments later when the wounded
1160
attacker detonated their suicide vest.
One plau-
1161
sible interpretation is that the model has learned
1162
to associate specific visual precursors—armed indi-
1163
viduals with bulky backpacks (concealing explosive
1164
vests), aggressive movement patterns toward a fa-
1165
cility entrance, and active gunfire in the context of
1166
critical infrastructure—with the Explosion category
1167
in its training distribution. The secondary predic-
1168
tions of Assault and Shooting further indicate that
1169
the model captures the multi-faceted nature of the
1170
unfolding attack, recognizing the concurrent hostile
1171
actions. Figure 10 presents the segment-level class
1172
probability heatmaps for both clips.
1173
Figure 10:
Multiclass probability heatmaps.
Top:
tai-a
shows dominant Explosion prediction despite no visible ex-
plosion (pre-detonation phase). Bottom: tai-n shows over-
whelming Normal classification for the routine operational
scene.
For the normal control video, the multiclass
1174
model assigned 86.46% probability to the Normal
1175
class, with the next highest class (Explosion) at
1176
only 5.90% and Vandalism at 1.52%. This result
1177
is especially significant given the visual complexity
1178
of the scene: numerous personnel in close proximity
1179
to high-value military aircraft could plausibly trig-
1180
ger false positives for categories like Vandalism or
1181
Assault in a less discriminative system.
1182
Figure 11 presents the frames corresponding to
1183
the three highest anomaly scores for each video.
1184
For tai-a, the peak frames capture the critical mo-
1185
ments of the attack: the armed assailants disem-
1186
barking from the taxi, the onset of the armed con-
1187
16

## Sayfa 17

Figure 11: Peak anomaly frames from the TUSAŞ case study. Top row: three highest-scoring segments from the attack video
(tai-a), showing armed assailants with backpacks (concealing explosive vests) and the armed confrontation. Bottom row: three
highest-scoring segments from the normal video (tai-n), showing routine personnel activity around military fighter jets.
frontation, and the gunfire exchange with security
1188
personnel. Notably, the attackers are visible carry-
1189
ing bulky backpacks—later confirmed to contain ex-
1190
plosive vests—which may contribute to the model’s
1191
Explosion prediction despite no detonation being
1192
visible. For tai-n, the peak frames show routine
1193
personnel activity around parked fighter aircraft,
1194
confirming the absence of any threatening visual
1195
cues.
1196
This case study carries a sobering implication.
1197
ASHEN flagged the attack video as anomalous with
1198
near-certain confidence (0.9837) from the earliest
1199
temporal segments, and identified the imminent
1200
threat category (Explosion) before the detonation
1201
occurred. Had such a system been operational at
1202
the TUSAŞ facility at the time of the attack, it
1203
could have provided security personnel with critical
1204
seconds of advance warning—potentially enough to
1205
initiate lockdown procedures, activate countermea-
1206
sures, or redirect evacuation routes. The ability to
1207
not only detect anomalies but also predict the tra-
1208
jectory of a developing threat scenario represents a
1209
meaningful step toward proactive, rather than re-
1210
active, intelligent surveillance.
1211
5. Conclusion
1212
This
paper
presented
ASHEN
(Automated
1213
Surveillance and Hostile Intent Evaluation Net-
1214
work), a weakly supervised framework for video
1215
anomaly detection that unifies vision–language rep-
1216
resentations with multimodal fusion and temporal
1217
modelling. By leveraging CLIP ViT-B/32 features
1218
as a semantic backbone and augmenting them with
1219
YOLOv8 object detection and human–object in-
1220
teraction (HOI) streams through a learned gated
1221
fusion mechanism, ASHEN achieves state-of-the-
1222
art performance on both binary anomaly detection
1223
and fine-grained multiclass hostile-intent classifica-
1224
tion.
On the UCF-Crime benchmark, our CLIP-
1225
based model attains a video-level AUC of 96.22%
1226
(note: prior methods report frame-level AUC; see
1227
Section 4.2 for discussion of this methodological dis-
1228
tinction). On our purpose-built Ashen Dataset, the
1229
system achieves 98.09% binary AUC, 80.83% nine-
1230
class accuracy, and 95.69% Macro-AUC, demon-
1231
strating that pre-trained vision–language mod-
1232
els can serve as powerful feature extractors for
1233
surveillance-specific anomaly detection.
1234
Six principal contributions distinguish this work.
1235
First, we introduced ASHEN as the first frame-
1236
work to combine CLIP, YOLO, HOI, and text cap-
1237
tion modalities under a unified gated fusion ar-
1238
chitecture for weakly supervised anomaly detec-
1239
tion. Second, we proposed a Pure Temporal Trans-
1240
former that operates directly on pre-extracted fea-
1241
tures without the computational overhead of end-
1242
to-end video backbone fine-tuning. Third, we cu-
1243
rated the Ashen Dataset—a purpose-built, bal-
1244
anced surveillance corpus of 1600 videos across nine
1245
hostile-intent categories—providing the research
1246
community with a focused benchmark for evaluat-
1247
ing anomaly detection in security-critical scenarios.
1248
Fourth, our comprehensive ablation study revealed
1249
17

## Sayfa 18

that multimodal fusion improves accuracy by over
1250
2 percentage points for spatial features and Macro-
1251
AUC by over 2 percentage points for textual fea-
1252
tures, while also identifying the conditions under
1253
which fusion may not help (noisy, large-scale la-
1254
bel spaces). Fifth, we demonstrated that the entire
1255
pipeline operates within real-time constraints on
1256
consumer-grade GPU hardware, making it suitable
1257
for practical surveillance deployment. Sixth, we val-
1258
idated ASHEN on real-world surveillance footage
1259
from the 2024 TUSAŞ terror attack, where the
1260
model detected the anomaly with 0.9837 confidence
1261
and predicted the imminent Explosion category be-
1262
fore any detonation was visible—demonstrating po-
1263
tential for proactive threat detection at critical in-
1264
frastructure sites.
1265
Despite these advances, several directions war-
1266
rant further investigation. The current framework
1267
relies on pre-extracted features, which limits adapt-
1268
ability to novel visual domains without recom-
1269
puting feature banks. Future work could explore
1270
lightweight adapters or prompt tuning strategies for
1271
domain-specific feature refinement.
Additionally,
1272
the relatively low recall for rare anomaly classes
1273
such as Burglary and Vandalism suggests that class-
1274
adaptive decision thresholds or few-shot learning
1275
techniques could improve performance on under-
1276
represented categories. The integration of tempo-
1277
ral attention visualisation and explainability mech-
1278
anisms would further enhance the interpretability
1279
of the system for security operators. Finally, ex-
1280
tending ASHEN to online streaming scenarios with
1281
incremental learning capability represents a natu-
1282
ral evolution toward fully autonomous surveillance
1283
systems.
1284
CRediT Author Statement
1285
Mehmet Taştan: Conceptualization, Method-
1286
ology, Software, Validation, Formal Analysis, Data
1287
Curation, Writing – Original Draft. Hamza Os-
1288
man İlhan: Conceptualization, Methodology, Su-
1289
pervision, Writing – Review & Editing.
1290
Declaration of Competing Interest
1291
The authors declare that they have no known
1292
competing financial interests or personal relation-
1293
ships that could have appeared to influence the
1294
work reported in this paper.
1295
Acknowledgements
1296
This work was supported by the Department of
1297
Computer Engineering at Yıldız Technical Univer-
1298
sity.
1299
Data Availability
1300
The Ashen Dataset introduced in this study, in-
1301
cluding all train/validation/test split files and fea-
1302
ture extraction scripts, will be made publicly avail-
1303
able upon acceptance at https://github.com/
1304
tastangh/ashen. The UCF-Crime dataset (5) is
1305
publicly available from its original authors.
Pre-
1306
extracted CLIP, YOLOv8, and HOI feature files for
1307
both datasets will be provided in the repository to
1308
enable full reproducibility without re-running fea-
1309
ture extraction.
1310
Code Availability
1311
The complete source code for ASHEN, includ-
1312
ing model architectures, training scripts, evalua-
1313
tion pipelines, and configuration files for all re-
1314
ported experiments, will be released upon accep-
1315
tance at https://github.com/tastangh/ashen.
1316
The repository includes the necessary instructions
1317
to reproduce all results reported in this paper.
1318
References
1319
[1] V.
Chandola,
A.
Banerjee,
V.
Kumar,
1320
Anomaly detection: A survey, ACM Comput-
1321
ing Surveys 41 (3) (2009) 1–58.
1322
[2] B. Ramachandra, M. Jones, Street scene: A
1323
new dataset and evaluation protocol for video
1324
anomaly detection, in: Proc. IEEE/CVF Win-
1325
ter Conference on Applications of Computer
1326
Vision (WACV), 2020, pp. 2569–2578.
1327
[3] R. Nayak, U. C. Pati, S. K. Das, A compre-
1328
hensive review on deep learning-based methods
1329
for video anomaly detection, Image and Vision
1330
Computing 106 (2021) 104078.
1331
[4] T. G. Dietterich, R. H. Lathrop, T. Lozano-
1332
Pérez, Solving the multiple instance problem
1333
with axis-parallel rectangles, Artificial Intelli-
1334
gence 89 (1–2) (1997) 31–71.
1335
18

## Sayfa 19

[5] W. Sultani, C. Chen, M. Shah, Real-world
1336
anomaly detection in surveillance videos, in:
1337
Proc. IEEE/CVF Conference on Computer Vi-
1338
sion and Pattern Recognition (CVPR), 2018,
1339
pp. 6479–6488.
1340
[6] D. Tran, L. Bourdev, R. Fergus, L. Torre-
1341
sani, M. Paluri, Learning spatiotemporal fea-
1342
tures with 3d convolutional networks, in: Proc.
1343
IEEE International Conference on Computer
1344
Vision (ICCV), 2015, pp. 4489–4497.
1345
[7] J. Carreira, A. Zisserman, Quo vadis, action
1346
recognition?
a new model and the kinet-
1347
ics dataset, in: Proc. IEEE/CVF Conference
1348
on Computer Vision and Pattern Recognition
1349
(CVPR), 2017, pp. 6299–6308.
1350
[8] W. Kay, J. Carreira, K. Simonyan, B. Zhang,
1351
C. Hillier, S. Vijayanarasimhan, F. Viola,
1352
T.
Green,
T.
Back,
P.
Natsev,
M.
Su-
1353
leyman,
A.
Zisserman,
The
kinetics
hu-
1354
man action video dataset,
arXiv preprint
1355
arXiv:1705.06950 (2017).
1356
[9] K. Simonyan, A. Zisserman, Two-stream con-
1357
volutional networks for action recognition in
1358
videos, in:
Advances in Neural Information
1359
Processing Systems (NeurIPS), Vol. 27, 2014.
1360
[10] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W.
1361
Verjans, G. Carneiro, Weakly-supervised video
1362
anomaly detection with robust temporal fea-
1363
ture magnitude learning, in: Proc. IEEE/CVF
1364
International Conference on Computer Vision
1365
(ICCV), 2021, pp. 4975–4986.
1366
[11] J. C. Feng, F. T. Hong, W. S. Zheng, Mist:
1367
Multiple instance self-training framework for
1368
video anomaly detection, in: Proc. IEEE/CVF
1369
Conference on Computer Vision and Pattern
1370
Recognition (CVPR), 2021, pp. 14009–14018.
1371
[12] S. Li, F. Liu, L. Jiao, Self-training multi-
1372
sequence learning with transformer for weakly
1373
supervised video anomaly detection, in: Proc.
1374
AAAI Conference on Artificial Intelligence,
1375
Vol. 36, 2022, pp. 1395–1403.
1376
[13] J. X. Zhong, N. Li, W. Kong, S. Liu, T. H.
1377
Li, G. Li, Graph convolutional label noise
1378
cleaner: Train a plug-and-play action classifier
1379
for anomaly detection, in: Proc. IEEE/CVF
1380
Conference on Computer Vision and Pattern
1381
Recognition (CVPR), 2019, pp. 1237–1246.
1382
[14] A.
Radford,
J.
W.
Kim,
C.
Hallacy,
1383
A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
1384
A. Askell, P. Mishkin, J. Clark, G. Krueger,
1385
I. Sutskever, Learning transferable visual mod-
1386
els from natural language supervision,
in:
1387
Proc. International Conference on Machine
1388
Learning (ICML), 2021, pp. 8748–8763.
1389
[15] H. K. Joo, K. Vo, K. Yamazaki, N. Le, Clip-
1390
tsa: Clip-assisted temporal self-attention for
1391
weakly-supervised video anomaly detection,
1392
in: Proc. IEEE International Conference on
1393
Image Processing (ICIP), 2023, pp. 3230–3234.
1394
[16] Z. Yang, J. Liu, P. Wu, Text prompt with nor-
1395
mality guidance for weakly supervised video
1396
anomaly detection,
in:
Proc. IEEE/CVF
1397
Conference on Computer Vision and Pattern
1398
Recognition (CVPR), 2024, pp. 18899–18908.
1399
[17] X. Wu, C. Zheng, J. Sang, S. Li, Y. Lu, L. Du,
1400
Vadclip: Adapting vision-language models for
1401
weakly supervised video anomaly detection, in:
1402
Proc. AAAI Conference on Artificial Intelli-
1403
gence, Vol. 38, 2024, pp. 6074–6082.
1404
[18] M. Li, J. Sang, Y. Lu, L. Du, Wsvad-clip: Tem-
1405
porally aware and prompt learning with clip
1406
for weakly supervised video anomaly detection,
1407
J. Imaging 11 (10) (2025) 354.
1408
[19] G. Jocher, A. Chaurasia, J. Qiu, Ultralytics
1409
yolov8 (2023).
1410
URL
https://github.com/ultralytics/
1411
ultralytics
1412
[20] Y. W. Chao, Y. Liu, X. Liu, H. Zeng, J. Deng,
1413
Learning to detect human-object interactions,
1414
in: Proc. IEEE Winter Conference on Appli-
1415
cations of Computer Vision (WACV), 2015.
1416
[21] A. Vaswani, N. Shazeer, N. Parmar, J. Uszko-
1417
reit, L. Jones, A. N. Gomez, L. Kaiser, I. Polo-
1418
sukhin, Attention is all you need, in:
Ad-
1419
vances in Neural Information Processing Sys-
1420
tems (NeurIPS), Vol. 30, 2017.
1421
[22] G. Pang, C. Shen, L. Cao, A. van den Hen-
1422
gel, Deep learning for anomaly detection: A
1423
review, ACM Computing Surveys 54 (2) (2021)
1424
1–38.
1425
[23] M. Hasan, J. Choi, J. Neumann, A. K. Roy-
1426
Chowdhury, L. S. Davis, Learning temporal
1427
regularity in video sequences.
1428
19

## Sayfa 20

[24] H. Zhou, J. Yu, W. Yang, Dual memory units
1429
with uncertainty regulation for weakly super-
1430
vised video anomaly detection, in: Proc. AAAI
1431
Conference on Artificial Intelligence, Vol. 37,
1432
2023, pp. 3769–3777.
1433
[25] Y. Chen, Z. Liu, B. Zhang, W. Fok, X. Qi,
1434
Y.
C.
Wu,
Mgfn:
Magnitude-contrastive
1435
glance-and-focus network for weakly super-
1436
vised video anomaly detection, in: Proc. AAAI
1437
Conference on Artificial Intelligence, Vol. 37,
1438
2023, pp. 387–395.
1439
[26] H. Lv, Z. Zhou, J. Yu, W. Yang, Unbiased mul-
1440
tiple instance learning for weakly supervised
1441
video anomaly detection, in: Proc. IEEE/CVF
1442
Conference on Computer Vision and Pattern
1443
Recognition (CVPR), 2024, pp. 18867–18876.
1444
[27] Y. Pu, X. Wu, S. Yang, Learning anomaly
1445
via gated fusion of language and video for
1446
weakly supervised video anomaly detection,
1447
IEEE Trans. on Circuits and Systems for Video
1448
Technology (2024).
1449
[28] A. Dosovitskiy,
L. Beyer,
A. Kolesnikov,
1450
D. Weissenborn,
X. Zhai,
T. Unterthiner,
1451
M.
Dehghani,
M.
Minderer,
G.
Heigold,
1452
S. Gelly, J. Uszkoreit, N. Houlsby, An image
1453
is worth 16x16 words: Transformers for im-
1454
age recognition at scale, in:
Proc. Interna-
1455
tional Conference on Learning Representations
1456
(ICLR), 2021.
1457
[29] T. Y. Lin, P. Goyal, R. Girshick, K. He, P. Dol-
1458
lár, Focal loss for dense object detection, in:
1459
Proc. IEEE International Conference on Com-
1460
puter Vision (ICCV), 2017, pp. 2980–2988.
1461
[30] J.
Redmon,
S.
Divvala,
R.
Girshick,
1462
A.
Farhadi,
You
only
look
once:
Uni-
1463
fied, real-time object detection, in:
Proc.
1464
IEEE/CVF Conference on Computer Vision
1465
and Pattern Recognition (CVPR), 2016, pp.
1466
779–788.
1467
[31] G. Gkioxari, R. Girshick, P. Dollár, K. He,
1468
Detecting and recognizing human-object in-
1469
teractions, in: Proc. IEEE/CVF Conference
1470
on Computer Vision and Pattern Recognition
1471
(CVPR), 2018, pp. 8359–8367.
1472
[32] J. Arévalo, T. Solorio, M. M. y Gómez, F. A.
1473
González, Gated multimodal units for informa-
1474
tion fusion, in: Proc. ICLR Workshop, 2017.
1475
[33] T. Baltrušaitis, C. Ahuja, L.-P. Morency, Mul-
1476
timodal machine learning: A survey and tax-
1477
onomy, IEEE Trans. on Pattern Analysis and
1478
Machine Intelligence 41 (2) (2019) 423–443.
1479
[34] J. L. Ba, J. R. Kiros, G. E. Hinton, Layer nor-
1480
malization, arXiv preprint arXiv:1607.06450
1481
(2016).
1482
[35] D.
Hendrycks,
K.
Gimpel,
Gaussian
er-
1483
ror
linear
units
(gelus),
arXiv
preprint
1484
arXiv:1606.08415 (2016).
1485
[36] N. Srivastava,
G. Hinton,
A. Krizhevsky,
1486
I. Sutskever, R. Salakhutdinov, Dropout: A
1487
simple way to prevent neural networks from
1488
overfitting, Journal of Machine Learning Re-
1489
search 15 (1) (2014) 1929–1958.
1490
[37] M. Ilse, J. M. Tomczak, M. Welling, Attention-
1491
based deep multiple instance learning,
in:
1492
Proc. International Conference on Machine
1493
Learning (ICML), 2018, pp. 2127–2136.
1494
[38] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens,
1495
Z. Wojna, Rethinking the inception architec-
1496
ture for computer vision, in: Proc. IEEE/CVF
1497
Conference on Computer Vision and Pattern
1498
Recognition (CVPR), 2016, pp. 2818–2826.
1499
[39] I. Loshchilov, A. Hutter, Decoupled weight
1500
decay
regularization,
in:
Proc.
Interna-
1501
tional Conference on Learning Representations
1502
(ICLR), 2019.
1503
[40] I. Loshchilov, A. Hutter, Sgdr: Stochastic gra-
1504
dient descent with warm restarts, in: Proc. In-
1505
ternational Conference on Learning Represen-
1506
tations (ICLR), 2017.
1507
[41] H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-
1508
Paz, mixup: Beyond empirical risk minimiza-
1509
tion, in:
Proc. International Conference on
1510
Learning Representations (ICLR), 2018.
1511
20

## Sayfa 21

Table 1: Comparison of the proposed ASHEN framework with existing weakly supervised video anomaly detection methods.
Method
Year
Feature Backbone
Core
Mechanism
/
Architecture
Complexity
Task
Main Limitation
Hasan
et
al.
(23), Liu et al.
(? )
<2018
Various (AE/GAN)
Temporal
regularity
learning; future frame
prediction
Low
Binary Only
Reconstruction objectives
to capture real-world anom
diversity;
high
false
posi
rate.
Sultani
et
al.
(5)
2018
C3D
Foundational
MIL
framework utilizing a
hinge-based
ranking
loss
Low
Binary Only
Highly motion-centric featu
severe semantic gap; cannot
tect subtle anomalies.
RTFM
(10),
MIST
(11),
MSL (12)
2021-22
C3D / I3D
Feature
magnitude
learning,
self-training
pseudo-labels,
multi-
sequence Transformer
Medium
Binary Only
Struggles to distinguish v
ally similar but semantic
distinct events; reliant on
sic motion priors.
UR-DMU
(24),
MGFN
(25),
GCN (13)
2019-23
C3D / I3D
Label-noise
cleaning,
remote dynamic mem-
ory units,
contrastive
magnitude modeling
High
Binary Only
Persisting semantic gap; sig
icant and increasing compu
tional overhead for continu
memory querying.
CLIP-TSA (15)
2023
CLIP (Visual Only)
Temporal
self-
attention
applied
to
pre-trained
visual
embeddings
Medium
Binary Only
Excludes text encoder entir
underexploits the cross-mo
semantic potential of the VL
VadCLIP (17)
2024
CLIP (Vision+Text)
Learnable
prompts,
dual-branch
archi-
tecture
for
vision-
language alignment
High
Binary & Multi-class
High
parameter
footp
(35M+);
requires
car
initialization to prevent ca
trophic forgetting.
TPWNG (16)
2024
CLIP (Vision+Text)
Text
prompts
with
normality
guidance,
temporal context self-
adaptive learning
High
Binary Only
Multi-stage optimization; h
parameter
needs;
suscept
to
domain
bias
from
flaw
pseudo-labels.
WSVAD-CLIP
(18)
2025
CLIP
Temporally
aware
Axial-Graph
(AG)
module,
Abnor-
mal
Visual-Guided
Prompts
Very High
Binary
&
Fine-
grained
Cumbersome
gra
transformer
interplay;
suitable for real-time inferen
high risk of overfitting.
PI-VAD
(π-
VAD) (? )
2025
I3D + 5 Modalities
Poly-modal
inductor;
teacher-student
syn-
thesis of Pose, Depth,
Flow, Panoptic, Text
High (Train) / Low (Infer)
Binary & Multi-class
Highly
complex
train
pipeline
requiring
contras
alignment
across
six
h
dimensional spaces.
GS-MoE (? )
2025
I3D + UR-DMU
Temporal
Gaus-
sian
Splatting
loss,
class-specific Mixture-
of-Experts routing
Very High
Binary & Multi-class
Expert redundancy drastic
increases FLOPs; poor gen
alization to entirely unkno
anomaly classes.
Ex-VAD (? )
2025
CLIP + LLM
Multimodal
Anomaly
Detection
Module,
Label
Augment
and
Alignment Module
Medium
Binary
&
Fine-
grained
Reliance on downstream L
phrasing
can
introduce
tency and semantic halluc
tion risks.
PANDA (? )
2025
MLLM / Agents
Training-free
Agentic
AI
engineer;
RAG,
tool-augmented
self-
reflection, memory
Extreme
Zero-Shot
Open
World
Crippling
inference
late
(<0.1 FPS); entirely unfeas
for
real-time
video
stre
processing.
Hoi2Anomaly (?
)
2025
Swin + HOTR
Human-Object
In-
teraction
tracking
coupled
with
visual-
linguistic
explanation
generation
High
Fine-grained
Narrow
focus
on
mi
interaction limits detection
broad
macro-environmen
anomalies.
DSANet (? )
2026
CLIP
Disentangled semantic
alignment;
self-guided
normality modeling
Medium
Binary
&
Fine-
grained
Still fundamentally bounded
the baseline capabilities of
underlying frozen CLIP emb
dings.
LAS-VAD (? )
2026
I3D
Anomaly-Connected
Components, Intention
Reasoning
Medium
Binary
&
Fine-
grained
Dependency on explicitly
fined anomaly attributes l
its purely open-world gene
ization.
ASHEN
(Ours)
2026
CLIP + YOLOv8
+ HOI
Pure
Temporal
Transformer, Gated
Fusion
Low
Binary & 9-Class
—
21

## Sayfa 22

Table 2: Per-class video and frame distribution of the UCF-
Crime dataset (5).
Class
Videos
Frames
Abuse
50
193,497
Arrest
50
297,395
Arson
50
271,903
Assault
50
129,946
Burglary
100
471,206
Explosion
50
252,404
Fighting
50
258,895
RoadAccidents
150
260,873
Robbery
150
422,594
Shooting
50
147,473
Shoplifting
50
324,367
Stealing
100
467,416
Vandalism
50
147,162
Normal
950
10,123,292
Total
1900
13,768,423
Table 3: Transition from UCF-Crime to the Ashen Dataset.
Classes marked with † were entirely removed due to non-
hostile intent.
Class
UCF
Removed
Added
ASHEN
Ashen Frames
Abuse†
50
−50
—
—
—
Arrest†
50
−50
—
—
—
Arson
50
—
+50
100
355,086
Assault
50
—
+50
100
140,369
Burglary
100
—
—
100
471,206
Explosion
50
—
+50
100
325,903
Fighting
50
—
+50
100
419,265
RoadAccidents†
150
−150
—
—
—
Robbery
150
−50
—
100
311,973
Shooting
50
—
+50
100
190,737
Shoplifting†
50
−50
—
—
—
Stealing†
100
−100
—
—
—
Vandalism
50
—
+50
100
272,598
Normal
950
−150
—
800
9,474,387
Total
1900
—
—
1600
11,961,524
Table 4: Ashen Dataset split distribution (70/15/15 strati-
fied).
Class
Total
Train
Val
Test
Arson
100
70
15
15
Assault
100
70
15
15
Burglary
100
70
15
15
Explosion
100
70
15
15
Fighting
100
70
15
15
Robbery
100
70
15
15
Shooting
100
70
15
15
Vandalism
100
70
15
15
Normal
800
557
119
120
Total
1600
1117
239
240
22

## Sayfa 23

Table 5: Hyperparameter configurations for the three main experimental settings.
Parameter
Binary Baseline
Binary Optimised
Multiclass
Feature Extractor
CLIP ViT-B/32
CLIP ViT-B/32
CLIP ViT-B/32
dmodel
256
256
256
Transformer Layers
2
3
3
Attention Heads
8
8
8
dff
1024
1024
1024
Dropout
0.3
0.5
0.3–0.5
Max Sequence Length
200
200
200
Batch Size
32
16
32
Optimiser
Adam
AdamW
AdamW
Learning Rate
1×10−4
1×10−4
5×10−5–1×10−4
Weight Decay
1×10−4
5×10−4
1×10−4–5×10−4
Scheduler
CosineAnnealing
CosineWarmRestarts
Cosine/WarmRestarts
Loss Functions
BCE + Sparsity + Smooth
Focal + Ranking + Smooth
CE/Focal + Ranking + Smooth
MixUp
—
α=0.2
α=0.2–0.4
Label Smoothing
—
0.05
0.1
Gradient Clipping
—
—
0.5
Max Epochs
50
50
50
Table 6: Comparison with state-of-the-art methods on UCF-
Crime. Prior methods report frame-level AUC; ASHEN re-
ports video-level AUC
Method
Feature
AUC (%)
Sultani et al. (5)
C3D
75.41
GCN (13)
C3D
82.12
MIST (11)
I3D
82.30
RTFM (10)
I3D
84.30
MSL (12)
I3D
85.30
UR-DMU (24)
I3D
86.97
MGFN (25)
I3D
86.98
CLIP-TSA (15)
CLIP
87.58
TPWNG (16)
CLIP
87.79
WSVAD-CLIP (18)
CLIP
87.85
VadCLIP (17)
CLIP
88.02
ASHEN (Ours)
CLIP
96.22
23

## Sayfa 24

Table 7: Binary anomaly detection results on the Ashen Dataset (240 test videos).
Model
Features
AUC (%)
Acc (%)
F1
Prec
Recall
ASHEN Baseline
CLIP
98.09
95.00
0.9487
0.9737
0.9250
ASHEN Optimised
CLIP
97.90
94.58
0.9442
0.9735
0.9167
ASHEN CLIP+YOLO
CLIP+YOLO
98.16
91.67
0.9115
0.9717
0.8583
ASHEN CLIP+HOI
CLIP+HOI
97.54
92.50
0.9224
0.9554
0.8917
ASHEN Triple
CLIP+YOLO+HOI
97.76
93.33
0.9333
0.9333
0.9333
Table 8: Multiclass anomaly classification results on the Ashen Dataset (240 test videos, 9 classes).
Model
Acc (%)
M-AUC (%)
W-AUC (%)
M-F1
CLIP Baseline
78.75
93.69
94.80
0.6807
CLIP Optimised
79.17
91.68
94.11
0.6722
CLIP ClassWeight
71.25
86.60
90.76
0.6381
CLIP+HOI
78.75
94.13
95.98
0.6649
CLIP+YOLO
78.33
93.11
95.23
0.6458
CLIP+YOLO+HOI
80.83
94.01
95.76
0.6797
CLIP+YOLO+TEXT
80.42
95.69
96.81
0.6789
Table 9:
Per-class results for CLIP+YOLO+HOI on the
Ashen Dataset.
Class
Prec.
Rec.
F1
AUC
Normal
0.91
0.98
0.94
0.9801
Arson
0.72
0.87
0.79
0.9861
Assault
0.73
0.73
0.73
0.9040
Burglary
0.50
0.33
0.40
0.9200
Explosion
0.76
0.87
0.81
0.9046
Fighting
0.81
0.87
0.84
0.9692
Robbery
0.64
0.47
0.54
0.9212
Shooting
0.82
0.60
0.69
0.9490
Vandalism
0.42
0.33
0.37
0.9268
Macro Avg
0.70
0.67
0.68
0.9401
24

## Sayfa 25

Table 10: Ablation study: incremental modality contributions on the Ashen Dataset (multiclass, 9 classes).
Configuration
Acc (%)
M-AUC (%)
W-AUC (%)
M-F1
CLIP
78.75
93.69
94.80
0.6807
CLIP + YOLO
78.33
93.11
95.23
0.6458
CLIP + HOI
78.75
94.13
95.98
0.6649
CLIP + YOLO + HOI
80.83
94.01
95.76
0.6797
CLIP + YOLO + TEXT
80.42
95.69
96.81
0.6789
Table 11: Cross-dataset comparison of ASHEN multiclass performance.
Dataset
Best Model
Acc (%)
M-AUC (%)
W-AUC (%)
M-F1
Ashen (9 cls)
CLIP+YOLO+HOI
80.83
94.01
95.76
0.6797
Ashen (9 cls)
CLIP+YOLO+TEXT
80.42
95.69
96.81
0.6789
UCF-Crime (14 cls)
CLIP Baseline
63.10
87.50
92.38
0.2495
UCF-Crime (14 cls)
CLIP+YOLO
63.45
86.66
91.35
0.2918
UCF-Crime (14 cls)
CLIP+YOLO+HOI
62.07
85.94
92.29
0.2820
25