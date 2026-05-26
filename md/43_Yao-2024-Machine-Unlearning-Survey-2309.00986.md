## On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study


### Amir M. Mir
##### Delft University of Technology Delft, The Netherlands s.a.m.mir@tudelft.nl

#### ABSTRACT


### Mehdi Keshani
##### Delft University of Technology Delft, The Netherlands m.keshani@tudelft.nl


### Sebastian Proksch
##### Delft University of Technology Delft, The Netherlands s.proksch@tudelft.nl


Static call graph (CG) construction often over-approximates call
relations, leading to sound, but imprecise results. Recent research
has explored machine learning (ML)-based CG pruning as a means
to enhance precision by eliminating false edges. However, current
methods suffer from a limited evaluation dataset, imbalanced training data, and reduced recall, which affects practical downstream
analyses. Prior results were also not compared with advanced static
CG construction techniques yet. This study tackles these issues. We
introduce the NYXCorpus, a dataset of real-world Java programs
with high test coverage and we collect traces from test executions
and build a ground truth of dynamic CGs. We leverage these CGs
to explore conservative pruning strategies during the training and
inference of ML-based CG pruners. We conduct a comparative analysis of static CGs generated using zero control flow analysis (0-CFA)
and those produced by a context-sensitive 1-CFA algorithm, evaluating both with and without pruning. We find that CG pruning
is a difficult task for real-world Java projects and substantial improvements in the CG precision (+25%) meet reduced recall (-9%).
However, our experiments show promising results: even when we
favor recall over precision by using an F2 metric in our experiments, we can show that pruned CGs have comparable quality to a
context-sensitive 1-CFA analysis while being computationally less
demanding. Resulting CGs are much smaller (69%), and substantially faster (3.5x speed-up), with virtually unchanged results in our
downstream analysis.

#### KEYWORDS


call graphs, machine learning, pruning, software analysis, empirical
study

**ACM Reference Format:**
Amir M. Mir, Mehdi Keshani, and Sebastian Proksch. 2024. On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study.
In 21st International Conference on Mining Software Repositories (MSR ’24),
_April_ _15–16,_ _2024,_ _Lisbon,_ _Portugal._ ACM, New York, NY, USA, 12 pages.
[https://doi.org/10.1145/3643991.3644897](https://doi.org/10.1145/3643991.3644897)

#### 1 INTRODUCTION


Call graphs (CG) represent function invocations within programs [12,
45]. Their construction is a crucial component of static program

Permission to make digital or hard copies of part or all of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for third-party components of this work must be honored.
For all other uses, contact the owner/author(s).
_MSR ’24, April 15–16, 2024, Lisbon, Portugal_
© 2024 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-0587-8/24/04.
[https://doi.org/10.1145/3643991.3644897](https://doi.org/10.1145/3643991.3644897)


analysis, like security analysis, dead code identification, performance profiling, and more. An ideal CG would be both sound, i.e.,
not missing any legitimate function call, and precise, i.e., not containing unnecessary function calls. However, constructing a sound
and precise CG is challenging even for small programs [6]. In practice, static CG construction will over-approximate the call relations to boost soundness at the cost of precision: popular tools like
WALA [16] or Petablox [32] create imprecise CGs with up to 76%
false edges [54]. To address this imprecision, previous work [10, 32,
52] have enhanced pointer analysis, which builds the backbone of
numerous CG construction algorithms, by improving the contextsensitivity or flow-sensitivity. Unfortunately, a flawless pointer
analysis is principally infeasible [44], and pointer analyses often
require a tradeoffbetween scalability and precision [27]. For instance, WALA’s context-sensitive analysis only reduces the false
positive rate by 8.6% compared to a context-insensitive analysis,
despite significantly slowing down performance [54].
Recent work has introduced Machine Learning (ML)-based call
graph pruning approaches to improve the precision of call graphs
by pruning false edges in call graphs as a post-processing step.
Techniques like CGPruner [54] and AutoPruner [25] learn from
dynamic traces that are collected in actual program executions to
identify unnecessary edges in a static CG. CGPruner only leverages
features of the CG structure, while AutoPruner combines structural
features with automatically extracted semantic features from the
source code that are encoded with the code language model (CLM),
CodeBERT [15]. Although these previous approaches show intriguing results, they suffer from several limitations: (1) Both have been
trained and evaluated on the NJR-1 dataset [37], which lacks realworld projects and suffers from a notoriously low branch coverage
(68%). (2) The over-approximation of static call graph construction
results in many unnecessary edges [42] while dynamic CGs contain
much fewer edges. As a result, the training and evaluation of the ML
models have to deal with a highly imbalanced dataset. (3) After CG
pruning, the recall drops substantially by more than 25% [25], which
makes the pruned CGs impractical for client analyses, especially
for security-focused applications. (4) The previous work used a 0CFA algorithm to generate static CGs, which is context-insensitive
and less precise. It is unclear how advanced, context-sensitive CG
algorithms like 𝑘-CFA algorithms [48] perform in comparison.
In this paper, we will address these issues by (1) introducing a
meta dataset, NYXCorpus, which includes the existing datasets NJR1 [37], and XCorpus [14]. We also added YCorpus, which is based
on another dataset of projects with a high test coverage of 88% [24].
In contrast to NJR-1, both XCorpus and YCorpus contain real-world
projects. We combined these three datasets and generated dynamic
traces through test execution to create a unified benchmark. (2) To


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.


address the second and third issues, we explore a conservative
pruning strategy during the learning phase and different confidence
levels for the inference of ML models to prune CG edges. These
two strategies help to deal with an imbalanced dataset and mitigate
the recall drop after pruning. (3) In addition to 0-CFA, we also use
1-CFA to generate static CGs and compare both algorithms with
and without pruning in terms of quality and scalability.
We will answer the following research questions to investigate
the impact of these three improvements:

**RQ1:** How do ML-based CG pruning models generally perform at
a CG pruning task?
**RQ2:** Can conservative training/pruning strategies improve the
results?
**RQ3:** How do context-sensitive CG generators compare in terms
of quality and scalability?
**RQ4:** Is CG pruning practical for a security application like vulnerability analysis?

Our main results show that CG pruning is difficult on real-world
Java projects. Although ML-based call graph pruning techniques
are effective at boosting the precision of static CGs, the recall drops
as a result. Our experiments report F2 values to prioritize recall over
precision, but even then the tradeoff is in favor of the ML pruners.
Pruned CGs have comparable quality to a context-sensitive 1-CFA
analysis, while their creation is computationally less demanding.
Our pruners can be configured by incorporating weights in the
learning process or confidence levels when pruning to control the
resulting precision and recall. Our experiments show that a wellconfigured pruner can improve the quality of a 0-CFA CG more than
running a more advanced 1-CFA analysis would. We will show that
both have a similar execution time, but that the pruned CG has a
higher quality and is smaller. We use the resulting CGs in a use case
analysis of a security-focused application, in which we investigate
the reachability of vulnerable methods. We can show that analyses
using pruned CGs generate very minimal false negatives (less than
2%) while benefiting from a faster analysis time of up to 5 times
due to the reduced size of pruned CGs.
Overall, this paper makes the following main contributions.

 - We created a new benchmark dataset, NYXCorpus, from preexisting datasets and tailored it to the call graph pruning task.
It has Java programs of various sizes including real-world ones.

 - We adapt existing ML models to support weighted training and
customizable pruning through confidence levels.

 - We present an empirical study on the effectiveness of ML-based
call graph pruning, which studies current issues, proposes solutions, and evaluates their effects.

The rest of this paper is organized as follows. We describe related
work in section 2. We explain our research methodology in section 3.
The evaluation setup for this study is described in section 4. We
present the obtained empirical results in section 5. We discuss the
implications of the obtained results in section 6. We describe threats
to validity and limitations in section 7. Finally, we conclude our
empirical study in section 8.


#### 2 RELATED WORK

_Call Graph Construction._ Call graph construction has been widely
studied. ML-based call graph pruner does not utilize run-time information and hence it falls into the category of static approaches [35,
43, 51] for constructing call graphs. Approaches that use dynamic
analysis [19, 60] result in fewer false positives and higher precision,
but they are less scalable.
Also, research has been conducted to enhance the precision of
call graphs. Lhotak [26] created an interactive tool to help understand the root cause of discrepancies between different static and
dynamic analysis tools. Sawin and Rountev [46] proposed specific
heuristics to manage dynamic features like reflection, dynamic class
loading, and native method calls in Java. This approach improved
the precision of the Class Hierarchy Analysis (CHA) algorithm [13]
while maintaining decent recall levels. Moreover, Zhang and Ryder [62] worked on generating precise application-only call graphs
by distinguishing false-positive edges between the standard library
and the application. Similar to the described work, ML-based CG
pruners [25, 54] aim to improve CG precision as a data-driven
post-processing approach by removing false edges.

_Call graph comparison._ Xie and Notkin [60] quantitatively and
qualitatively compared dynamic and static call graphs from two
Java micro-benchmarks. They found that static call graphs tend
to be conservative but imprecise due to computational complexity.
Dynamic call graphs, on the other hand, are more straightforward
and reflect the actual invocations. Lhotak [26] presented a technique
to find the root causes of call graph differences and the PROBE
framework. PROBE facilitates comparing dynamic and static call
graphs to identify sources of imprecision. In this study, we compare
pruned static call graphs for Java programs to their dynamic call
graphs, and we analyze the differences between them in terms of
precision and soundness.

_Machine learning-based call graph pruning._ As of this writing,
there are currently two ML-based call graph pruning models, CGPruner [54] and AutoPruner [25]. Utture et al. [54] introduced an
ML-based technique called CGPruner, with the goal of reducing
the false-positive rate of static analysis tools, making them more
attractive to developers. CGPruner prunes the static call graph,
which is at the core of many static analyses, by removing falsepositive edges while retaining true edges. The technique achieves
this balance using an ahead-of-time learning process involving executing static and dynamic call-graph constructors. The dynamic
call graphs were only used during a training phase on a training
set of programs. CGPruner was shown to significantly decrease the
false-positive rate, in one case, from 73% to 23%.
CGPruner does not consider source code semantics. To address
this limitation, Le-Cong et al. proposed AutoPruner [25] to prune
false positives in call graphs by leveraging both structural and statistical semantic information. The semantic features extracted from
the caller and callee functions’ source code. Specifically, AutoPruner
uses CodeBERT [15], a pre-trained Transformer model [56] for code,
fine-tuning it to capture semantic features for each edge and combines them with handcrafted structural features, and employs a
neural classifier to classify each edge as true or false-positive.


-----

On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study MSR ’24, April 15–16, 2024, Lisbon, Portugal

_Machine Learning for Software Engineering._ In recent years, the Input datasets
application of machine learning for software engineering has been

NJR-1 XCorpus YCorpus

a hot topic of research [7, 47]. ML models have been used to perform various tasks, such as code completion, code summarization,
defect prediction, code classification, and code translation tasks.
Recently, large-scale code language models (CLMs) [61] such as

Executing unit tests

CodeBERT [15] and CodeT5 [58] have achieved state-of-the-art
performance on numerous SE tasks mentioned above. In general, Static CG Construction
ML can offer opportunities to improve or automate several aspects (WALA)
of the traditional software development process. The scale of soft- Dynamic CG Construction
ware artifact data, automated feature engineering provided by ML (Wiretap)
techniques, robustness and scalability of optimization techniques,
and transferability of traditional ML applications to SE artifacts all
indicate the potential of ML to improve the traditional software Post processing
development process. This research area is called Machine Learning Filtering edges Sampling edges
for Software Engineering (ML4SE).

#### 3 APPROACH Extracting code features

In this section, we first define the research problem under study. Structural Semantic Signature Combined
Then, we introduce the various ingredients of our research methodology: the datasets that we use in our experiments, a description
of the _call-graph generation_ (both static and dynamic), an expla- Model training
nation of ML models used in previous works [25, 54], and recent CGPruner AutoPruner CodeBERT
suitable code language models for this task; lastly, we describe the
different code features that we use for training call graph pruners. CodeT5 CodeT5+
Figure 1 shows an overview of our research methodology used in
this empirical study. Overall, our proposed methodology consists
of three datasets, static/dynamic CG construction, post-processing Evaluation
like filtering/sampling edges, training of ML models, and empirical
evaluation. All these steps are presented later in the paper. Overall The conservative

performance pruning strategy

#### 3.1 Problem definition CG pruning vs. more Vulnerability propagation

|Semantic|Signature|Combined|
|---|---|---|


This paper studies CG pruning, which takes a static call graph G as
initial input. A CG is a directed graph created using a static analysis
tool. The vertices _𝑉_ of the graph represent defined functions, which
are identified by a function signature (name, parameters, return
type). The edges 𝐸 represent calls from one function to another.
Each edge within 𝐸 is defined as a tuple, that consists of the calling
function (caller), the function being called (callee), and the site
within the caller where the call is made (offset).
The output G[′] is a refined version of the original CG, where
G[′] = (𝑉 [′], 𝐸[′]), 𝑉 [′] = _𝑉_, and _𝐸[′]_ is a subset of _𝐸._ The reduction is
achieved through a binary classifier, 𝐶, which is designed to decide
per edge 𝑒 ∈ _𝐸, whether the edge should be copied to G[′]_ or pruned.
Our validation is based on dynamic CGs that we construct from
traces of actual program and test executions and that we use to
validate the pruned call graph G[′].

#### 3.2 Datasets

In this section, we describe the three datasets of our study, that we
use to train and evaluate the ML-based CG pruning models.

_NJR-1._ Normalized Java Resource (NJR) [37] is an infrastructure
to leverage the potential of Big Code. The normalization enables
searchability, scriptability, and reproducibility. The NJR comprises
100,000 executable Java programs, a set of pre-existing tools, which
facilitate the development of novel research tools. For evaluating

|Input datasets<br>NJR-1 XCorpus YCorpus|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|Executing unit tests|Executing unit tests|Executing unit tests|Executing unit tests|Executing unit tests|
|Executing unit tests|Executing unit tests|Static CG Construction<br>(WALA)|Static CG Construction<br>(WALA)|Static CG Construction<br>(WALA)|
||||||
|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|
|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)|Dynamic CG Construction<br>(Wiretap)||
||||||
|Post processing<br>Filtering edges<br>Sampling edges|Post processing<br>Filtering edges<br>Sampling edges|Post processing<br>Filtering edges<br>Sampling edges|Post processing<br>Filtering edges<br>Sampling edges|Post processing<br>Filtering edges<br>Sampling edges|
||||||
|Extracting code features<br>Structural<br>Semantic<br>Signature<br>Combined|Extracting code features<br>Structural<br>Semantic<br>Signature<br>Combined|Extracting code features<br>Structural<br>Semantic<br>Signature<br>Combined|Extracting code features<br>Structural<br>Semantic<br>Signature<br>Combined|Extracting code features<br>Structural<br>Semantic<br>Signature<br>Combined|
||||||
|Model training<br>CGPruner<br>AutoPruner<br>CodeBERT<br>CodeT5<br>CodeT5+|Model training<br>CGPruner<br>AutoPruner<br>CodeBERT<br>CodeT5<br>CodeT5+|Model training<br>CGPruner<br>AutoPruner<br>CodeBERT<br>CodeT5<br>CodeT5+|Model training<br>CGPruner<br>AutoPruner<br>CodeBERT<br>CodeT5<br>CodeT5+|Model training<br>CGPruner<br>AutoPruner<br>CodeBERT<br>CodeT5<br>CodeT5+|
||||||
|Evaluation<br>Overall<br>performance<br>The conservative<br>pruning strategy<br>CG pruning vs. more<br>advanced static analysis<br>Vulnerability propagation<br>as a case study|Evaluation<br>Overall<br>performance<br>The conservative<br>pruning strategy<br>CG pruning vs. more<br>advanced static analysis<br>Vulnerability propagation<br>as a case study|Evaluation<br>Overall<br>performance<br>The conservative<br>pruning strategy<br>CG pruning vs. more<br>advanced static analysis<br>Vulnerability propagation<br>as a case study|Evaluation<br>Overall<br>performance<br>The conservative<br>pruning strategy<br>CG pruning vs. more<br>advanced static analysis<br>Vulnerability propagation<br>as a case study|Evaluation<br>Overall<br>performance<br>The conservative<br>pruning strategy<br>CG pruning vs. more<br>advanced static analysis<br>Vulnerability propagation<br>as a case study|


**Figure 1: Overview of our approach used in this study**

the ML-based call graph pruning models, we use a subset of the
NJR1 dataset, created by the work of Utture et al. [54]. The subset
contains 141 programs from the NJR-1 benchmark suite, of which
100 programs are used for training the models and 41 programs for
evaluation. The selection of 141 programs from NJR-1 programs
was based on criteria such as each program having at least 1,000
methods and 2,000 static call graph edges as per Wala, executing a
minimum of 100 distinct methods during runtime, and exhibiting
high coverage, i.e., executing a large portion of the methods that
can be reached from the main method (with an average coverage of
68%). On average, each selected program comprises around 560,000
lines of code, excluding the standard library [54].

_XCorpus._ The XCorpus dataset [14] contains a set of 76 executable Java programs, which includes 70 from the Qualitas Corpus [53]. This corpus combines both built-in and generated test
cases, offering better branch coverage than the DaCapo benchmark [9]. While the DaCapo benchmark and Qualitas Corpus are
curated datasets for benchmarking and static analysis, respectively,


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.


XCorpus combines the strengths of both—being executable (like
DaCapo) and diverse and extensive (like Qualitas Corpus). Such
a dataset is useful for research on program analysis, studies combining static and dynamic analyses, or studies on program transformations that evaluate impact through program execution preand post-transformation. The average coverage for XCorpus’ programs is moderate, with 62.35% for built-in and 60.25% for generated
tests by Evosuite [17]. On average, each program has an average of
around 36K lines of code. XCorpus has been used in the empirical
studies on the soundness of Java call graphs [43, 51].

_YCorpus._ For this work, we created a new dataset, namely, YCorpus, based on an existing dataset used in a recent empirical study by
Khatami and Zaidman [24], which investigates state-of-the-practice
in quality assurance in Java-based open source software development. Specifically, they have studied 1,454 popular Java projects
on GitHub with more than 100 stars. Given this, we selected 40
Java projects with the criteria that each project has higher than 80%
test coverage. These 40 projects have 88% test coverage on average,
which is substantially higher than that of XCorpus and NJR-1. Also,
YCorpus contains real-world Java projects such as Apache Commons IO, AssertJ, and MyBatis 3, and each project has an average
of around 50K lines of code. Both XCorpus and YCorpus have been
reduced to the programs that we could build and for which we were
able to construct both static and dynamic CGs.

_NYXCorpus._ In the remainder of the paper, we will refer to NYXCorpus as a dataset to indicate that we have based our experiments
on the joined data of all three corpora.

_Source Code Recovery._ The original NJR-1 dataset lacks source
code for the dependencies of its programs. However, call graphs
contain nodes/methods related to dependencies and code language
models need source code to learn the call graph pruning task. We
have identified dependencies in the NJR-1 dataset, located their
respective repositories (often on platforms like GitHub), and downloaded the necessary source code to extend the dataset with the
original code, including comments, for deeper code understanding.
For XCorpus and YCorpus, we downloaded sources JARs for their
programs via the Maven or Ant command-line tools.

#### 3.3 Call-Graph Generation

This subsection describes our dynamic and static CG generation
and explains our filter and sampling criteria for program edges.

_Dynamic Call Graphs._ To evaluate (pruned) static call graphs, we
need to establish an "oracle", a known ground truth that represents
actual program behavior. In this context, the oracle refers to vertices
in a call graph which represents methods. These methods are recognized using a mix of the class name where the method is defined,
the method’s name, and a descriptor, as per the Java language specification [28], to account for overloading. The edges in call graphs
are formed by pairs of source and target methods. To obtain such an
oracle, we utilize unit tests that are commonly available and can be
an effective way to initiate program executions. In fact, built-in test
cases offer a unique insight as they represent the intended behavior
of a program, mirroring the experience an end-user might have
when using the software in a real-world setting [51].


To collect the method calls of a program, we have instrumented
it via Wiretap [23], a tool to trace information from a running Java
program. Specifically, we wrote a recorder to insert probes at Java
method entries and exits to record call relationships. We then ran
all available unit tests to gather execution paths, creating dynamic
call graphs for the actual execution paths, serving as an oracle
or "ground truth" for evaluating ML-based call graph models. We
use this dynamic data to train a model for detecting and pruning
irrelevant or infeasible paths in static CGs.

_Static_ _Call_ _Graphs._ We employ the WALA framework [16] to
generate static CGs with the _context-insensitive_ 0-CFA (Zerothorder Control Flow Analysis) [48], and the context-sensitive 1-CFA
algorithm. For this study, we chose WALA over alternatives like
DOOP [49] and Soot [55] as it has better support for Java language
features such as lambda expressions and, as of this writing, it supports Java bytecode up to JDK 17 [5]. We follow prior research
work [25, 54] and do not use WALA’s handler for Java reflection,
which can potentially miss some execution paths that involve reflective calls. In short, given a Java program, we perform the following
steps to construct a static CG using the WALA framework:

 - We consider each project’s main JAR file as the application
scope and its transitive dependencies as the extension scope.

 - We perform a _Class_ _Hierarchy_ _Analysis_ [13], which involves
constructing the class inheritance hierarchy to facilitate the
resolution of method call targets.

 - All non-private methods within all public classes are used as
entry points for WALA’s call graph builder.

 - The obtained entry points and the CHA structure are used to
construct 0/1-CFA static CGs.

_Filtering edges._ Considering the call graph pruning problem, we
are interested in call graph edges related to the application itself and
its dependencies. We follow previous work and opt for removing
edges to/from the Java standard library as its enormous size would
dominate the dataset and skew the evaluation [54]. Specifically, we
remove all call edges that start with the following prefixes: java/,
javax/, sun/, com/sun/, jdk/.

_Large Programs._ Utture et al. [54] observed that a few programs
in the NJR-1 dataset have a very large number of call-graph edges
(over 20K), and they randomly sampled 20K edges from the edge
sets of those programs. Following this, we also randomly sampled
20K edges from the edge sets of 5 programs in the XCorpus and
YCorpus datasets to alleviate the skewness in the dataset distribution. This also prevents bias towards large programs when training
a model/classifier. Also, we do not remove or sample edges where
they exist in both dynamic and static call graphs, as fewer of these
true edges exist. A removal of true edges would harm the performance of the ML models at retaining true edges, i.e., recall.

#### 3.4 Call-Graph Pruning Models

In this subsection, we describe several machine learning techniques,
including code language models, which we extend and employ for
our CG pruning task.


-----

On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study MSR ’24, April 15–16, 2024, Lisbon, Portugal


_Random Forest [20]._ An ensemble learning method, constructs
decision trees on bootstrapped datasets using Bagging [11], considering a random feature subset at each node. Predictions are derived
from majority voting for classification or averaging for regression
tasks. The algorithm is versatile and adept at handling numerous
inputs, missing values, and errors in unbalanced datasets. However,
it can be a "black box" model and may overfit noisy datasets.

_CodeBERT [15]._ A bimodal pre-trained model for programming
language (PL) and natural language (NL) tasks, leverages a Transformer-based architecture [56] and a hybrid objective function
inclusive of replaced token detection during pre-training. It utilizes
both bimodal and unimodal data, helping to learn better generators.
Trained on CodeSearchNet [21], which contains GitHub repositories in six languages, it is similar to multilingual BERT without
explicit language markers. Empirical results show that fined-tuned
CodeBERT has superior performance on natural language code
search and code documentation generation tasks. Without parameter fine-tuning, zero-shot setting tests also indicate the superiority
of RoBERTa [29], suggesting its effective learning and application
in NL-PL tasks.

_CodeT5 [58]._ Built on the T5 architecture [41], utilizes denoising sequence-to-sequence pre-training for both understanding and
generation tasks in natural language. A novel identifier-aware pretraining task is introduced for better leveraging code semantics.
Similar to CodeBERT, it is pre-trained on CodeSearchNet [21] data
and additional data from open-source GitHub C/C# repositories.
It is fine-tuned on most CodeXGLUE benchmark tasks and supports multi-task learning. Experimental results reveal that CodeT5
outperforms CodeBERT on various tasks, demonstrating enhanced
capture of semantic information from code.

_CodeT5+_ _[57]._ An adaptable family of encoder-decoder Large
Language Models (LLMs) designed for code tasks, combining different pre-training objectives, including span denoising, contrastive
learning, text-code matching, and causal language modeling, for
flexible applications in various modes. Initiated with frozen off-theshelf LLMs [36], it circumvents training from scratch, promoting
efficient scaling. Upon evaluating 20+ code-related benchmarks,
CodeT5+ exhibits superior performance in tasks including code
generation, completion, and text-to-code retrieval.

#### 3.5 Code Features

We use the following features or code representations to train our
ML-based CG pruning models. The intuition behind all features is
to provide information about the usefulness of an edge.

_Structural._ Utture et al. [54] engineered a set of structural features encapsulating vital contextual and semantic call edge details,
adhering to three criteria: linear-time computational complexity, interpretability/generalizability, and black-box nature. The proposed
feature set is a combination of local and global information extracted from static call graphs (more info in [54]). The structural
features, _𝑓𝑠𝑡𝑟𝑢𝑐𝑡_, build a 𝑘𝑠 -dimensional vector (𝑘𝑠 = 11):

f𝑠𝑡𝑟𝑢𝑐𝑡 = [𝑥1[𝑠𝑡𝑟𝑢𝑐𝑡] _,𝑥2[𝑠𝑡𝑟𝑢𝑐𝑡]_ _, ...,𝑥𝑘[𝑠𝑡𝑟𝑢𝑐𝑡]𝑠_ ] (1)

_Semantic._ Semantic features are extracted from the source code
of the caller and callee functions, which is also used in the work


of Le-Cong et al. [25]. Unlike hand-crafted structural features, semantic features are automatically learned by using code language
models. They can generate a high-dimensional vector that captures
the statistical relationships between caller and callee functions.
Thus, each edge in the call graph has an associated embedding that
represents the semantic relationship between the caller and the
callee. Conceptually, semantic features are represented as follows:

[CLS]⟨caller’s source⟩[SEP]⟨callee’s source⟩[EOS] (2)

The semantic features, _𝑓𝑠𝑒𝑚, are represented as a 𝑘𝑐_ -dimensional
vector (𝑘𝑐 = 768), which are the output embeddings of a code
language model such as CodeT5.

f𝑠𝑒𝑚 = [𝑥1[𝑠𝑒𝑚],𝑥2[𝑠𝑒𝑚], ...,𝑥𝑘[𝑠𝑒𝑚]𝑐 ] (3)

_Signature-based._ AutoPruner [25] extracts features from caller
and callee method signatures to supplement CG nodes without
source code [2], namely, _class_ _&_ _method_ _name,_ _parameters,_ and
_return types. This code feature provides minimal code context but is_
helpful when source code is unavailable. Signature-based features,
_𝑓𝑠𝑖𝑔, are represented as a 𝑘𝑐_ -dimensional vector:

f𝑠𝑖𝑔 = [𝑥1[𝑠𝑖𝑔][,𝑥]2[𝑠𝑖𝑔][, ...,𝑥]𝑘[𝑠𝑖𝑔]𝑐 []] (4)

_Combined._ Le-Cong et al. [25] proposed a combined feature set,
which takes advantage of both structural and semantic features,
to prune call graph edges effectively. It has empirically shown
that AutoPruner [25], CodeBERT, with the combined feature set,
outperforms a RandomForest model trained on structural features.
The combined features, 𝑓𝑐𝑜𝑚𝑏, are represented as the concatenation
of two vectors x𝑠𝑒𝑚 and x𝑠𝑡𝑟𝑢𝑐𝑡 :

f𝑐𝑜𝑚𝑏 = x𝑠𝑡𝑟𝑢𝑐𝑡 ⊕ x𝑠𝑒𝑚 (5)

#### 3.6 Model Training

We fine-tune our code language models for two epochs. Specifically,
only the encoder module of the CLMs is fine-tuned, which generates
embedding for code features mentioned in subsection 3.5. To speed
up training, we utilized mixed precision training with a floating
point precision of 16-bit, which effectively reduces the GPU memory
consumption without sacrificing the model’s performance. We used
an initial learning rate of 1 × 10[−][5], which was found to be effective
for training such models without causing instability in the learning
process [25]. We use cross-entropy loss as the loss function, which
is suitable for binary classification problems:


_𝐿(𝑦,_ _𝑦ˆ)_ = − [1]

_𝑁_


_𝑁_
∑︁[𝑤1 ∗𝑦𝑖 ∗𝑙𝑜𝑔(𝑦 ˆ𝑖 ) +𝑤2 ∗(1 −𝑦𝑖 ) ∗𝑙𝑜𝑔(1 −𝑦ˆ𝑖 )] (6)

_𝑖=1_


Where 𝐿(𝑦, _𝑦ˆ) is the loss function comparing the true labels 𝑦_ and
the predicted labels _𝑦ˆ. 𝑁_ is the total number of samples. 𝑤1 and 𝑤2
are the weights associated with the positive and negative classes,
respectively. This allows us to define pruning strategies like "conservative" by giving a higher weight to the positive class. The conservative strategy prioritizes high recall by maintaining as many
edges as possible and only prunes when certain. We study the effect
of these weights on pruning call graphs in RQ2.
We used the AdamW optimizer [30], a variant of the Adam
optimizer that corrects its weight decay regularization, boosting
generalization and controlling over-fitting. A dropout rate of 0.25


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.


**Table 1: Stats for the datasets used in evaluation**

**Num. Edges** **Num. Tokens** 1
**Dataset** **P/R Ratio**

**Train** **Test** **Train** **Test**

NJR-1 859K 405K 262M 124M 18.7
XCorpus 269K 57K 22M 8M 2.4
YCorpus 242K 72K 35M 9M 3.7
NYXCorpus 1.37M 534K 319M 141M 7.6

1 Ratio of to-be-pruned and to-be-retained edges.

was applied to the models’ encoder output to prevent overfitting
further [50]. Also, we adopted a linear scheduling policy [18] with
a warmup phase of 100 steps. This method gradually ramps up the
learning rate from zero to the specified maximum (1 × 10[−][5] in this
case) during the warmup phase to avoid large gradient updates
early in training, thus aiding in better convergence.

#### 3.7 Model Inference

Given a fine-tuned code language model, we prune CG edges as
follows. Let x be an input to the CLM and the linear neural network
(NN) produces raw scores for the two classes, 𝑧0 and 𝑧1. Then, the
softmax function converts these raw scores into probabilities.

_𝑒[𝑧][𝑖]_
_𝑝_ (𝑦 = 𝑖 |x) = (7)

_𝑒[𝑧][0]_ + 𝑒[𝑧][1]

Where 𝑖 is the class label which can take values 0 or 1 and 𝑝 (𝑦 = 𝑖 |x)
is the probability of class 𝑖. Finally, we use a decision function with
a threshold (e.g., 𝜏 = 0.5) to decide the predicted class as follows.


where _𝛽_ determines the weight of precision in the combined
score. _𝛽_ _<_ 1 gives more weight to precision, while 𝛽 _>_ 1 favors
recall. We report 𝐹1 and 𝐹2 in our evaluation.

_Implementation and Environmental setup._ To parse and extract
Java methods’ source code, we utilized a Java parser [1]. We used PyTorch 2.0 [3, 38] with PyTorch Lightning 2.0 [4] to train and evaluate
code language models described in section 3.4. We used the pretrained code language models from HuggingFace’s transformers
library [59]. To implement the CGPruner model, i.e., RandomForest,
we employed the scikit-learn library [39]. We used NetworKit [8], a
toolkit for large-scale network analysis with optimized algorithms
to process graph data and extract structural features. We also used
JGraphT [33] in Java to do graph traversals and reachability analysis. We performed all the experiments on a Linux workstation
(Ubuntu 22.04 LTS) with Intel Core i9 13900KS@6GHz, an RTX
4090 24GB, and 2x48GB (96GB) DDR5 RAM.

_Datasets_ _characteristics._ Table 1 shows the characteristics of
the NJR-1, XCorpus, and YCorpus datasets. NJR-1 has 1.2M samples/edges, whereas XCorpus and YCorpus have 326K and 314K
edges, respectively. We also have a "meta" dataset, namely, NYXCor_pus, by combining the training and test sets of the three datasets._
This allows us to compare the performance of the models across all
datasets and their programs.
In Table 1, we also reported the P/R ratio for each dataset, representing the number of to-be-pruned edges divided by the number
of true edges. It can be seen that NJR-1 has a P/R ratio of 18.7, which
is much higher than that of XCorpus and YCorpus. This means that
the NJR-1 dataset is massively imbalanced. For XCorpus and YCorpus, a lower P/R ratio means that more static edges are observed
during test execution at run-time. Also, as expected, NYXCorpus is
placed between NJR-1 and Y/XCorpus given its P/R ratio.

#### 5 EVALUATION

This section presents the motivation, methodology, and empirical
results for all research questions that were defined before.

#### 5.1 RQ1: How do ML-based CG pruning models generally perform at a CG pruning task?

As the first step of our evaluation, we want to explore the overall
performance of the different ML models and their capacity to prune
static CGs. This first assessment provides insights into their abilities
and allows us to reduce the list to the most promising candidates
for the rest of the paper.

_Methodology._ We used our three datasets (NJR-1, XCorpus, and
YCorpus) for this experiment. Specifically, for the NJR-1 dataset,
we trained and evaluated the models in the same way prior work
did [54], using 100 programs for training and 41 for evaluation.
For XCorpus/YCorpus, we trained the models on 12/15 programs
and evaluated them on 4/3 programs, respectively. The results also
list NYXCorpus, which is a combination of the three datasets that
helps us with choosing the overall best-performing models. For
all datasets, Wala’s static CGs were constructed using 0-CFA and
there is no overlap of programs in the training and test sets. We
should also point out that, for the CLMs, we use semantic features


_𝑦ˆ_ =


�1 if 𝑝 (𝑦 = 1|x) _> 𝜏_
(8)
0 otherwise


In many cases with NNs having two output neurons, since the
probabilities produced by the softmax function for both classes sum
to 1, a threshold of 0.5 is commonly used. If 𝑝 (𝑦 = 1|x) _>_ 0.5, it
automatically implies 𝑝 (𝑦 = 0|x) _< 0.5 and vice versa._

#### 4 EVALUATION SETUP

In this section, we explain the evaluation metrics for evaluating
(pruned) call graphs, the implementation details, model training,
and the characteristics of the datasets used in this study.

_Evaluation Metrics._ Similar to the previous work [25, 54], to assess the accuracy of a static call graph, we use the common evaluation metrics, precision and recall. We denote the edge set generated
by a static call-graph constructor as 𝐸𝑆, and the edge set created by
Wiretap as 𝐸𝐷 . The proportion of incorrect identifications is represented by (1-Precision). To obtain the average precision and recall
values for the complete test set, we calculate the mean precision
and recall values of individual programs and the 𝐹𝛽 measure.

Precision = [|][𝐸][𝑆] [∩] _[𝐸][𝐷]_ [|] Recall = [|][𝐸][𝑆] [∩] _[𝐸][𝐷]_ [|]

|𝐸𝑆 | |𝐸𝐷 |


_𝐹𝛽_ = [(][1][ +]𝛽[ 𝛽][2] ×[2][) ×] 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛[ 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛]+ 𝑅𝑒𝑐𝑎𝑙𝑙[×][ 𝑅𝑒𝑐𝑎𝑙𝑙]


-----

On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study MSR ’24, April 15–16, 2024, Lisbon, Portugal

**Table 2: Comparison of the models on the NJR-1, XCorpus, YCorpus, and NYXCorpus datasets**

**NJR-1** **XCorpus** **YCorpus** **NYXCorpus**

**Models** P R F1 F2 P R F1 F2 P R F1 F2 P R F1 F2

**Random Classifier** 0.23 0.47 0.31 0.39 0.39 **0.48** 0.43 0.46 0.22 0.45 0.29 0.37 0.25 0.47 0.33 0.40
**CGPruner** **0.66** 0.48 0.56 0.51 0.49 0.28 0.36 0.30 **0.71** 0.24 0.36 0.28 0.61 0.43 0.50 0.46
**AutoPruner** 0.62 0.66 0.64 0.65 0.53 0.41 0.46 0.43 0.50 **0.51** **0.50** **0.51** 0.60 **0.61** 0.60 **0.60**
**CodeBERT** 0.62 0.68 0.65 0.67 0.52 0.47 **0.50** **0.48** 0.50 0.48 0.49 0.49 0.59 0.60 0.60 **0.60**
**CodeT5** 0.65 0.69 **0.67** 0.68 0.54 0.31 0.39 0.34 0.50 0.48 0.49 0.48 0.63 0.58 **0.61** 0.59
**CodeT5+** 0.63 **0.73** **0.67** **0.70** **0.61** 0.23 0.34 0.27 0.54 0.46 **0.50** 0.48 **0.65** 0.57 **0.61** 0.58

**Average** 0.57 0.62 0.58 0.60 0.51 0.36 0.41 0.38 0.49 0.44 0.44 0.43 0.55 0.54 0.54 0.54

**Wala** 0.24 0.95 0.38 0.59 0.39 0.95 0.55 0.73 0.22 0.90 0.35 0.55 0.25 0.95 0.39 0.60


if source code is available for an edge/sample. Otherwise, we use
the signature-based feature as a fallback.
To find the optimal hyper-parameters for CGPruner (i.e., RandomForest), similar to the work of Utture et al. [54], we performed
4-fold cross-validation with grid search. In addition to the all models described in subsection 3.4, we also considered a random binary
classifier, which prunes/retains edges with an equal probability, i.e.,
0.5. To assess the quality of pruned static call graphs by the ML
models, we used the evaluation metrics described in section 4.

_Results._ Table 2 shows the general performance of all the models
on four datasets, namely, NJR-1, XCorpus, YCorpus, and NYXCorpus. In addition to the traditional F1 score, we have also included
the F2 score in our evaluation, which puts a higher importance
on the recall of an approach. The results show that all the ML
models substantially outperform the random classifier across all
datasets and all metrics, with the exception of the _recall_ in the
XCorpus, which seems to be an outlier. The code language models
(i.e., AutoPruner, CodeBERT, and CodeT5(+)) generally perform
better than CGPruner at the CG pruning task. This is expected as
the CLMs leverage code semantics whereas CGPruner only relies
on structural features.
From Table 2, we also observe that all the models perform better
on the NJR-1 dataset compared to XCorpus and YCorpus. This is
because both XCorpus and YCorpus contain popular real-world Java
projects in contrast to NJR-1, which focused on automation over
popularity when selecting Java projects [37] and popular projects
seem to be more difficult for the models. In addition, the ML models
perform best on NYXCorpus after NJR-1 with an F2 score of 0.54.
This score shows that the gained precision comes at the price of a
reduced recall when compared to Wala. While the average F2 score
is only 0.54 on NYXCorpus, the best models can match the quality
of Wala’s 0-CFA analysis. Based on these results, we decided to use
CodeBERT and CodeT5 for the subsequent RQs. These two models
perform better than others concerning the F2 score and they do not
require structural features, unlike AutoPruner. Also, they are faster
compared to CodeT5+ considering both training and inference.
However, these results also bring two follow-up questions. First,
the comparatively low recall and many missing edges can prove
impractical for client analyses. We will explore the effect of more
conservative training and pruning strategies in **RQ2.** Second, a


context-sensitive CG generator might achieve the same performance gain with better soundness. As such, we will compare the
quality and scalability of the results when using a 1-CFA analysis
in RQ3.

#### 5.2 RQ2: Can conservative training/pruning strategies improve the results?

The three datasets of our evaluation are imbalanced, especially
NJR-1, as can be seen from the P/R ratio in Table 1. There are
substantially more edges that need to be pruned than edges to be
kept. If both true and false edges are treated equally in the training
phase, the ML models will get biased towards pruning. Indeed, the
results of RQ1 have shown that the recall drops significantly, so we
will experiment with more conservative strategies during training
and pruning to limit the effects of the imbalance. The goal is to keep
as many edges as possible to minimize the false pruning decisions.

_Methodology._ We use the two most suitable ML-based CGs pruners
from RQ1, i.e., CodeBERT and CodeT5, and experiment with two
conservative enhancements. First, we assign a weight 𝑤 during the
learning process to the positive class (i.e., retaining edges) in the
cross-entropy loss function (see Equation 6) to fine-tune two models separately. Second, we consider the confidence of the pruning
decision and require reaching a configurable threshold 𝜏, before
an edge gets pruned. We investigate the effects in two separate
experiments. In the first experiment, we perform a grid search over
the training weights {0, 6, 0.7, 0.8, 0.9, 0.95, 0.99} to investigate the
effect of the weighted loss function. In the second experiment, we
used the unweighted version of the two models that have been
used in RQ1 already, and performed another grid search over the
confidence threshold values {0, 6, 0.7, 0.8, 0.9, 0.95} in the decision
function defined in Equation 8 to find the best-performing confidence level. The higher the value of 𝜏, the more conservative we
are when pruning CG edges. Both experiments use 0-CFA-based
static CGs.

_Results._ Figure 2 shows the performance of the models while
fine-tuning them with different weights to the positive and negative
classes. For instance, a weight of 0.70 to the positive class means
that the negative class is given a weight of 0.30. Overall, for both
CodeBERT and CodeT5, we observe that the F2 score increases and
precision decreases by giving higher weight to the positive class.


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.


**(a) CodeBERT**

0.80

0.74 0.70 0.72

0.64 0.63

0.52

0.7 0.8
Confidence levels

**(a) CodeBERT**


**(b) CodeT5**

0.75

0.70

0.650.67 0.65

0.57

0.8
Confidence levels

**(b) CodeT5**


**Figure 3: Performance of the models by considering different confidence thresholds**


The results suggest that the weights of 0.95 and 0.99 are the most
interesting configurations. While the F2 score drops slightly when
moving from 0.95 to 0.99, it is to be expected that the resulting CG is
also larger. It seems that 0.95 can be considered as the "sweet spot" to
achieve a recall close to Wala’s while gaining higher precision. The
use case for 0.99 is the application where soundness is essential, like
vulnerability analysis. In short, with weights given to the classes, it
is possible to maintain a relatively high recall while having better
precision compared to Wala’s static call graphs.
Figure 3 indicates the performance of the models considering
different confidence levels for pruning call graph edges. It can
be seen that the higher the confidence level, the higher the F2
score is, which is expected as the model is more confident when
pruning edges. Overall, it becomes obvious that the differences to
the unweighted results in Figure 3 are minimal, and also here, the
0.95 and 0.99 levels are the best-performing configurations. The
confidence-based filtering seems to be as effective as using weights
in the loss function, while not requiring the additional overhead of
fine-tuning.
The similar F2 scores of the original and the pruned CG beg the
question of how large the effect of the pruning is in practice. We
will investigate the impact on a client analysis and the performance
implication of a substantially reduced CG on runtimes in RQ4.


#### 5.3 RQ3: How do context-sensitive CG generators compare in terms of quality and scalability?

The results of **RQ1** have shown that ML-based CG pruning can
improve a 0-CFA-based CG with a small computational overhead.
The interesting question is how this overhead compares to running
more advanced, context-sensitive CG algorithms like k-CFA (i.e.,
1-CFA), which has higher precision but is also computationally
more expensive. In this section, we will investigate how using a
1-CFA analysis in the CG generation compares to 0-CFA (with and
without pruning) in terms of performance and scalability.

_Methodology._ First, we generate static CGs using both 0-CFA
and 1-CFA algorithms for the training and test programs in the
NYXCorpus dataset. We reused the CodeBERT and CodeT5 models
that have been fine-tuned on the 0-CFA CGs in RQ1. We also finetuned both CLM models on 1-CFA CGs of the NYXCorpus training
set with no weight given to the loss function. When pruning static
CGs, we use a confidence threshold of 0.95, which we found to be
an effective configuration in RQ2.
In our second experiment, we measure the CPU time for generating a static call graph using 0-CFA and 1-CFA. In addition, we
show how long it takes to prune call graphs by measuring feature
extraction and model inference time separately. Lastly, we sum up


-----

On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study MSR ’24, April 15–16, 2024, Lisbon, Portugal


**Table 3: Model Performance with 0/1-CFA algorithm**

**NYXCorpus**
**Models**

P R F1 F2

_0-CFA_

**CodeBERT** 0.47 0.89 0.61 **0.76**
**CodeT5** **0.51** 0.85 **0.63** 0.75
**Wala** 0.25 **0.95** 0.39 0.60

_1-CFA_

**CodeBERT** 0.49 0.91 0.64 **0.78**
**CodeT5** **0.53** 0.86 **0.65** 0.76
**Wala** 0.34 **0.97** 0.50 0.71

the CPU time for both static CG generation and CG pruning to
show the total computational time of the whole process. For each
measurement, we report the average and the standard deviation
across the NYXCorpus programs.

_Results._ Table 3 shows the performance of the CodeBERT and
CodeT5 models when fine-tuned on 0-CFA and 1-CFA call graphs
for the call graph pruning task. The first observation is that, unsurprisingly, Wala’s context-sensitive 1-CFA CGs have a 9% higher
precision than Wala’s 0-CFA CGs. Also, the achieved recall is higher,
which in combination results in a substantial increase of both F1
and F2 scores by 11%. It is interesting to see that this CG improvement is barely visible in the pruned CGs, which only see a 1-2%
improvement in their F1 and F2 scores.
Table 4 provides an overview of the runtimes of the different
configurations of 0/1-CFA with and without pruning to allow an assessment of the results in terms of scalability. CG pruning consists
of feature extraction, i.e., tokenizing code sequences and creating
semantic features, and model inference, i.e., querying the CLM
model to prune call graph edges. Clearly, the CG pruning task adds
additional computational overhead on top of the static CG generation. The table therefore splits the different stages and shows the
averages and standard deviations for static CG generation, feature
extraction and inference, and the total runtime in seconds.
The results show that a 1-CFA-based static CG generation takes
42.3s, which is almost twice as long as the 0-CFA algorithm without
pruning (21.4s). The 1-CFA alternative is therefore as expensive as 0CFA with pruning (≈ 42s), however, its standard deviation is much
higher (120s vs. 65s). This is likely caused by the computational
complexity of static analysis, which, unlike ML models that have
a constant runtime per query, does not scale linearly with the
program size. The results suggest that context-sensitive analysis
can be beneficial for small programs, while ML-based approaches
scale better. It is worth noting that the improved CGs of the 1CFA analysis also have a positive impact on the runtime of the ML
approaches. As the CGs are smaller and contain fewer edges, the
runtime of the pruner goes down and deviates less.


**Table 4: Runtime of 0/1-CFA algorithms with CG pruning**

**Pruning [s]**
**Models** **CG Gen. [s]** **Total Time [s]**
Feature Infer.

_0-CFA_

**CodeBERT** 18.6 ± 30.9 42.7 ± 65.2
21.4 ± 57.0 2.8 ± 3.5
**CodeT5** 18.9 ± 31.4 43.0 ± 65.4

_1-CFA_

**CodeBERT** 11.7 ± 11.2 55.1 ± 122.6
42.3 ± 120.5 1.4 ± 0.8
**CodeT5** 14.5 ± 20.0 57.9 ± 123.9

#### 5.4 RQ4: Is CG pruning practical for a security application like vulnerability analysis?

All previous experiments have used statistical means to assess
the pruning performance of the ML models by comparing ground
truth and pruned CG through metrics such as F2-score. Previous
works have employed client analyses on the pruned CGs, like null
pointer exceptions (NPE), to show the effects of the pruning on
static analyses in practice [25, 54]. We follow this example and study
the effects of CG pruning on vulnerability propagation, a securitysensitive analysis that requires the traversal of call graphs [34]. We
will report on the resulting CG sizes and the runtime of the client
analysis. We expect a significant speed-up in finding vulnerable
call paths using pruned static call graphs, though pruned CGs may
be susceptible to false negatives, which needs to be investigated.

_Methodology._ We used WALAs 0-CFA static CGs as the baseline
and we employ the CodeBERT and CodeT5 models that have been
fine-tuned on the training set of NYXCorpus. We use two configurations for the comparison. The conservative configuration does
not add weights to the loss function but uses a 0.95 confidence
threshold. The paranoid configuration reuses the CodeBERT and
CodeT5 models that have been fine-tuned with a weight of 0.99
to the positive class and applies a confidence level of _>_ 0.95 for
pruning.
Our experimental design builds upon the availability of methodlevel vulnerability information, which is provided by tools like
Prospector [40]. We did not use real-world vulnerabilities for programs in NYXCorpus though, as the NJR-1 dataset does not include
Maven coordinates for its dependencies, plus many projects without vulnerabilities would have to be filtered. As such, we decided
to randomly mark 100 methods as vulnerable in each program
of the NYXCorpus test set. All marked CG nodes represent nonapplication nodes that are defined in the dependencies of a program.
Our experimental goal is then to measure how long it takes to compute all paths that start in the application and reach a vulnerability
with a simple reachability analysis through a Breadth-first-search
(BFS). We calculate the fraction of vulnerable methods that are
reachable in a given CG, before and after pruning. Also, there is no
reason to believe that our artificial vulnerabilities are easier to reach
than actual vulnerabilities. To accurately measure the analysis time,
we first run the code three times to warm up the JVM and let the JIT
compilation do its optimization. We then compute the reachability
another three times and average the required execution time.


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.


**Table 5: Vulnerability Propagation Analysis on (pruned) CGs**

**CG Size** **Reachable ...**
**Models** **Time (ms)**
Edges Nodes[1] .. Paths .. Nodes

**Wala** 5227.1 1420.2 853.9 99.8% 8.1 ± 26.6

_Conservative Pruning_ (> 0.95 confidence)

**CodeBERT** 1736.4 1048.2 515.3 86.0% 2.3 ± 8.1
**CodeT5** 1498.2 950.5 388.6 82.0% 1.5 ± 4.8

_Paranoid Pruning_ (0.99 weight, > 0.95 confidence)

**CodeBERT** 3728.4 1392.2 778.9 98.4% 6.6 ± 23.0
**CodeT5** 3503.5 1337.3 832.5 96.9% 6.7 ± 23.0

1 #Nodes with at least an incoming and/or out-going edge

It is worth pointing out that while the absolute number of identified paths is lower in a pruned CG, we believe that the crucial
information is whether a vulnerable method is reached at all. Moreover, it is irrelevant for the actionability of the results, whether 1
or 10 affected paths can be found for a given vulnerability.

_Results._ Table 5 shows the results of vulnerability propagation for
both 0-CFA-based static CGs and their pruned version. Note that the
reported numbers for CG size, the number of reachable vulnerable
paths and nodes are average per test program in NYXCorpus.
WALA is the baseline for the comparison. It is obvious that
both pruning strategies are able to substantially reduce the original
CG size from 5.2K edges to 1.5-1.7K (≈ 33%) with the conservative
setting and 3.5-3.7K (≈ 69%) in the paranoid setting. This results
in substantial reductions in the runtime of the client analysis to
only 1.5ms (5.4x speedup) in CodeT5 and 2.3ms (3.5x speedup) in
CodeBERT. While the concrete reachability analysis is very fast
even on the original CG, we have already seen earlier in the paper
that static analyses scale non-linearly, so every reduction in the size
of a CG will have a substantial impact in more advanced analyses.
The substantial reduction of the conservative setup comes at the
price of only reaching 86% of the vulnerable nodes. However, the
_paranoid_ setup is able to retain the reachability of 96-98% of the
vulnerable nodes, which comes very close to the WALA baseline.
We find that the reduced size and substantial speedup make this
result very attractive for large-scale analyses, but the best CG choice
always depends on the task and the context. A security-focused
application might accept the slower execution time of an unpruned
1-CFA-based CG analysis to gain a sound result. For other use cases,
a paranoid setup might be all that is required, or even a conservative
analysis could work, when performance is the main issue. It is
noteworthy that, at least in the presented analysis, the pruning
does not introduce any false positives and only introduces a small
fraction of false negatives.

#### 6 DISCUSSION

When reflecting on the obtained evaluation results, we believe
that several points are noteworthy and should be considered by
researchers and practitioners.


_Call graph pruning is an open problem._ As shown throughout the
paper, code language models like CodeBERT and CodeT5 have the
potential to substantially improve the precision of static CGs. However, we have also seen that CG pruning is challenging, especially
for the real-world programs in the XCorpus and YCorpus. While
the precision is good, the main challenge is achieving a reasonably
good recall as well. The probabilistic nature of CLM models makes
it easy to introduce pruning thresholds, however, the parameters
of our current approaches must be fine-tuned in a small range at
the extremes. Our models can present a promising step in the right
direction, but they do not give an exhaustive answer to the larger
problem. More work is required to find a more robust approach
with a more differentiated confidence measure and better results
overall. Future work could explore hybrid approaches that combine
heuristic (non-) pruning rules with a CLM model.

_Data_ _Imbalance._ We believe that the main limitation that we
have faced in our experiments is the massive imbalance of the
dataset, as seen in the P/R ratio in Table 1). Naturally, trained ML
models will be biased towards pruning edges rather than retaining
them. We believe that future work should continue to emphasize
_recall_ over _precision,_ as we have done by using the _F2_ measure
when optimizing their models. However, this is only the first step
and further approaches need to be taken to counter the imbalance.
Our technique for building the ground truth was executing test
suites for collecting relevant edges. Future work could extend this
endeavor and trace more extensive program executions and build
more complete dynamic CGs.

_Hybrid Static Analysis._ Recent works have introduced advancements in ML-based CG pruning, but also advanced program analysis
approaches that consider call site sensitivity and more context to
improve the precision of static CGs [22, 31]. Unfortunately, the MLbased approaches and advanced static analyses are still often seen
as related, but separate solutions to the CG generation problem.
Likely, because both are very advanced topics in their respective
fields and because it is hard to find researchers who are experts in
both areas. We strongly believe though that a hybrid static analysis
that integrates ML-based approaches into the static analysis instead
of running it as a separate step would be very promising. Our experiments have shown that even the best static CG generator that
we included could not reach 100% of the vulnerable nodes. This
is, for example, caused by calls through the Java reflection API,
which could be suggested through complementing probabilistic approaches. Another potential combination would be the use of ML to
improve the performance of advanced static analyses, which would
otherwise not scale to large programs, for example, by accepting
unsound results for less important parts of the program.

_Feature_ _Engineering._ In contrast to mostly structural features
and graph metrics that have been used in previous work, we used
semantic features that are based on the source code that surrounds
the potential call site. Overall, we believe that the obtained results
are positive, but the feature engineering idea should be investigated
more closely. It is likely that considering full methods for sources
and targets exceeds the attention of ML models, therefore important
information is not taken into consideration by the model. Future
work could investigate new ways to encode the surrounding context


-----

On the Effectiveness of Machine Learning-based Call Graph Pruning: An Empirical Study MSR ’24, April 15–16, 2024, Lisbon, Portugal


of a method call to find better, semantic features, which might carry
more relevant information about the likelihood of a call relation
between two methods.

#### 7 THREATS TO VALIDITY AND LIMITATIONS

In this section, we describe the limitations of our work, possible
threats to the validity of our empirical findings, and how we address
them. We have picked F2 as the main metric to judge the CG quality
of our pruned CGs. While it could be possible that the metric still
does not emphasize the importance of recall enough, we believe
that our results in the vulnerability analysis confirm the suitability
of the metric in our experiments.
Our experimental result rests on the ground truth that we have
generated through the instrumentation and execution of test suites,
which might not be complete or representative of other programs.
We selected programs with a high test coverage, which makes us
confident that the results are reliable. Larger benchmark datasets
will certainly contain more cases that might be missed in this paper,
but they would also provide more data to train the ML models.
Overall, we are confident about the representativeness of our data,
confirming the data with larger datasets will be left for future work.
We have chosen a vulnerability analysis as a client analysis that
is built upon a CG. We do not even start to object that this choice
might not be representative of other analysis tasks. However, we
think that the generated results and the insights still hold, as the
described downsides of static analyses only get worse with larger
programs or more advanced static analysis algorithms.
Lastly, we filter call graph edges based on package names as
described in subsection 3.3. This may cause the exclusion of call
graph edges related to Graphical User Interface (GUI) components
or event-driven programming aspects from the evaluation. This
is not a threat but a limitation of the filtering strategy we used,
following the previous work [25, 54], which filters such edges if
their CG node’s URI starts with any of those filtered packages like
java/. Future work should propose a more robust approach to
filtering call graph edges before the training phase.

#### 8 SUMMARY

This paper presents an empirical study on the effectiveness of machine learning-based call graph pruning. We identified several key
issues in the current state of research on ML-based call graph pruning such as a lack of a suitable benchmark dataset, data imbalance
due to static analysis over-approximation, significant recall drop in
CG pruning, and no comparison between pruned 0-CFA-based call
graphs with context-sensitive algorithms like 𝑘-CFA. To address
these challenges, we have introduced (1) the NYXCorpus dataset,
combining NJR-1, XCorpus, and YCorpus. (2) and a conservative
strategy to prune CG edges more confidently, which can be tuned
by giving weights to classes in the learning process or considering
different confidence levels when pruning. Our empirical findings
show substantial improvement in CG precision. Specifically, MLbased CG pruning can boost precision by 24-34% while reducing the
recall by 2-10%. Even though our experiments favor recall over precision, we can show through a comparison with a more advanced
1-CFA-based CG generation that the overall tradeoff is in favor of
the ML-based approaches. We show in a client analysis that by


tweaking our model parameters to a paranoid setup, it is possible
to achieve virtually identical results to a static analysis while being
3.5x faster and operating on a reduced CG with 69% of its original
size.

#### DATA AVAILABILITY

The datasets including NYXCorpus, all the fine-tuned CLMs, and
the source code to replicate the results of the experiments in this
paper are publicly available.
[Data & Models: https://zenodo.org/doi/10.5281/zenodo.10638852](https://zenodo.org/doi/10.5281/zenodo.10638852)
[Source code: https://github.com/mir-am/ml4cgp_study](https://github.com/mir-am/ml4cgp_study)

#### ACKNOWLEDGMENTS

The FASTEN project has received funding from the European Union’s
Horizon 2020 research and innovation program under grant agreement number 825328. Also, we would like to thank the anonymous
reviewers and Georgios Gousios for providing valuable feedback to
improve this paper.

#### REFERENCES

[1] [n. d.]. Java 1-17 Parser and Abstract Syntax Tree for Java with advanced analysis
functionalities. [https://javaparser.org/](https://javaparser.org/) Accessed: 2023-07-31.

[2] [n. d.]. The official open-source implementation of AutoPruner. [https://github.](https://github.com/soarsmu/AutoPruner/)
[com/soarsmu/AutoPruner/](https://github.com/soarsmu/AutoPruner/) Accessed: 2023-08-01.

[3] [n. d.]. PyTorch 2.0. https://pytorch.org/blog/pytorch-2.0-release/. 2023-06-13.

[4] [n. d.]. _PyTorch Lightning._ [https://lightning.ai/docs/pytorch/latest/](https://lightning.ai/docs/pytorch/latest/)

[5] [n. d.]. T.J. Watson Libraries for Analysis, with frontends for Java, Android, and
JavaScript, and may common static program analyses. [https://github.com/wala/](https://github.com/wala/WALA/releases)
[WALA/releases](https://github.com/wala/WALA/releases) Accessed: 2023-11-17.

[6] Karim Ali and Ondrej Lhotak. 2012. Application-only call graph construction. In
_ECOOP 2012–Object-Oriented Programming: 26th European Conference, Beijing,_
_China, June 11-16, 2012. Proceedings 26. Springer, 688–712._

[7] Miltiadis Allamanis, Earl T Barr, Premkumar Devanbu, and Charles Sutton. 2018.
A survey of machine learning for big code and naturalness. _ACM Computing_
_Surveys (CSUR) 51, 4 (2018), 1–37._

[8] Eugenio Angriman, Alexander van der Grinten, Michael Hamann, Henning
Meyerhenke, and Manuel Penschuck. 2022. _Algorithms for Large-Scale Network_
_Analysis and the NetworKit Toolkit._ Springer Nature Switzerland, Cham, 3–20.
[https://doi.org/10.1007/978-3-031-21534-6_1](https://doi.org/10.1007/978-3-031-21534-6_1)

[9] Stephen M Blackburn, Robin Garner, Chris Hoffmann, Asjad M Khang, Kathryn S
McKinley, Rotem Bentzur, Amer Diwan, Daniel Feinberg, Daniel Frampton,
Samuel Z Guyer, et al. 2006. The DaCapo benchmarks: Java benchmarking development and analysis. In Proceedings of the 21st annual ACM SIGPLAN conference
_on Object-oriented programming systems, languages, and applications. 169–190._

[10] Martin Bravenboer and Yannis Smaragdakis. 2009. Strictly declarative specification of sophisticated points-to analyses. In Proceedings of the 24th ACM SIGPLAN
_conference on Object oriented programming systems languages and applications._
243–262.

[11] Leo Breiman. 1996. Bagging predictors. _Machine learning 24 (1996), 123–140._

[12] David Callahan, Alan Carle, Mary W. Hall, and Ken Kennedy. 1990. Constructing
the procedure call multigraph. _IEEE Transactions on Software Engineering 16, 4_
(1990), 483–487.

[13] Jeffrey Dean, David Grove, and Craig Chambers. 1995. Optimization of objectoriented programs using static class hierarchy analysis. In ECOOP’95—Object_Oriented Programming, 9th European Conference, Åarhus, Denmark, August 7–11,_
_1995 9. Springer, 77–101._

[14] JB Dietrich, Henrik Schole, Li Sui, and Ewan Tempero. 2017. XCorpus–an executable corpus of Java programs. (2017).

[15] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong,
Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020. CodeBERT:
A Pre-Trained Model for Programming and Natural Languages. In Findings of
_the Association for Computational Linguistics: EMNLP 2020. Association for Com-_
putational Linguistics, 1536–1547. [https://doi.org/10.18653/v1/2020.findings-](https://doi.org/10.18653/v1/2020.findings-emnlp.139)
[emnlp.139](https://doi.org/10.18653/v1/2020.findings-emnlp.139)

[16] Stephen Fink and Julian Dolby. 2012. WALA–The TJ Watson Libraries for Analysis.

[17] Gordon Fraser and Andrea Arcuri. 2011. Evosuite: automatic test suite generation
for object-oriented software. In Proceedings of the 19th ACM SIGSOFT symposium
_and the 13th European conference on Foundations of software engineering. 416–419._


-----

MSR ’24, April 15–16, 2024, Lisbon, Portugal Mir et al.



[18] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski,
Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. 2017. Accurate,
large minibatch sgd: Training imagenet in 1 hour. _arXiv preprint arXiv:1706.02677_
(2017).

[19] Joseph Hejderup, Arie van Deursen, and Georgios Gousios. 2018. Software
ecosystem call graph for dependency management. In Proceedings of the 40th
_International Conference on Software Engineering: New Ideas and Emerging Results._
101–104.

[20] Tin Kam Ho. 1995. Random decision forests. In Proceedings of 3rd international
_conference on document analysis and recognition, Vol. 1. IEEE, 278–282._

[21] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
Brockschmidt. 2019. Codesearchnet challenge: Evaluating the state of semantic
code search. _arXiv preprint arXiv:1909.09436 (2019)._

[22] Minseok Jeon and Hakjoo Oh. 2022. Return of CFA: call-site sensitivity can be
superior to object sensitivity even for object-oriented programs. _Proceedings of_
_the ACM on Programming Languages 6, POPL (2022), 1–29._

[23] Christian Gram Kalhauge and Jens Palsberg. 2018. Sound deadlock prediction.
_Proceedings of the ACM on Programming Languages 2, OOPSLA (2018), 1–29._

[24] Ali Khatami and Andy Zaidman. 2023. State-Of-The-Practice in Quality Assurance in Java-Based Open Source Software Development. _arXiv_ _preprint_
_arXiv:2306.09665 (2023)._

[25] Thanh Le-Cong, Hong Jin Kang, Truong Giang Nguyen, Stefanus Agus Haryono, David Lo, Xuan-Bach D Le, and Quyet Thang Huynh. 2022. AutoPruner:
transformer-based call graph pruning. In Proceedings of the 30th ACM Joint Eu_ropean Software Engineering Conference and Symposium on the Foundations of_
_Software Engineering. 520–532._

[26] Ondrej Lhotak. 2007. Comparing call graphs. In _Proceedings_ _of_ _the_ _7th_ _ACM_
_SIGPLAN-SIGSOFT workshop on Program analysis for software tools and engineer-_
_ing. 37–42._

[27] Yue Li, Tian Tan, Anders Møller, and Yannis Smaragdakis. 2018. Scalabilityfirst pointer analysis with self-tuning context-sensitivity. In Proceedings of the
_2018 26th ACM joint meeting on european software engineering conference and_
_symposium on the foundations of software engineering. 129–140._

[28] Tim Lindholm, Frank Yellin, Gilad Bracha, Alex Buckley, and Daniel Smith. 2021.
The Java Virtual Machine Specification: Java SE 17 Edition. (2021).

[29] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A
robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_
(2019).

[30] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization.
_arXiv preprint arXiv:1711.05101 (2017)._

[31] Wenjie Ma, Shengyuan Yang, Tian Tan, Xiaoxing Ma, Chang Xu, and Yue Li.
2023. Context Sensitivity without Contexts: A Cut-Shortcut Approach to Fast
and Precise Pointer Analysis. _Proceedings of the ACM on Programming Languages_
7, PLDI (2023), 539–564.

[32] Ravi Mangal, Xin Zhang, Aditya V Nori, and Mayur Naik. 2015. A user-guided
approach to program analysis. In Proceedings of the 2015 10th Joint Meeting on
_Foundations of Software Engineering. 462–473._

[33] Dimitrios Michail, Joris Kinable, Barak Naveh, and John V. Sichi. 2020. JGraphT–
A Java Library for Graph Data Structures and Algorithms. _ACM Trans. Math._
_Softw. 46, 2, Article 16 (May 2020), 29 pages._

[34] Amir M Mir, Mehdi Keshani, and Sebastian Proksch. 2023. On the Effect of
Transitivity and Granularity on Vulnerability Propagation in the Maven Ecosystem. In 2023 IEEE International Conference on Software Analysis, Evolution and
_Reengineering (SANER). IEEE, 201–211._

[35] Gail C Murphy, David Notkin, William G Griswold, and Erica S Lan. 1998. An
empirical study of static call graph extractors. _ACM Transactions on Software_
_Engineering and Methodology (TOSEM) 7, 2 (1998), 158–191._

[36] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou,
Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language
model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474
(2022).

[37] Jens Palsberg and Cristina V Lopes. 2018. Njr: A normalized java resource. In
_Companion Proceedings for the ISSTA/ECOOP 2018 Workshops. 100–106._

[38] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.
2019. Pytorch: An imperative style, high-performance deep learning library. In
_Advances in neural information processing systems. 8026–8037._

[39] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss,
Vincent Dubourg, et al. 2011. Scikit-learn: Machine learning in Python. _the_


_Journal of machine Learning research 12 (2011), 2825–2830._

[40] Serena E. Ponta, Henrik Plate, Antonino Sabetta, Michele Bezzi, and C´edric
Dangremont. 2019. A Manually-Curated Dataset of Fixes to Vulnerabilities of
Open-Source Software. In _Proceedings_ _of_ _the_ _16th_ _International_ _Conference_ _on_
_Mining Software Repositories._

[41] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of
transfer learning with a unified text-to-text transformer. _The Journal of Machine_
_Learning Research 21, 1 (2020), 5485–5551._

[42] Michael Reif. 2021. Novel Approaches to Systematically Evaluating and Constructing Call Graphs for Java Software. (2021).

[43] Michael Reif, Florian Kübler, Michael Eichberg, Dominik Helm, and Mira Mezini.
2019. Judge: Identifying, understanding, and evaluating sources of unsoundness
in call graphs. In Proceedings of the 28th ACM SIGSOFT International Symposium
_on Software Testing and Analysis. 251–261._

[44] Henry Gordon Rice. 1953. Classes of recursively enumerable sets and their
decision problems. _Transactions of the American Mathematical society 74, 2 (1953),_
358–366.

[45] Barbara G Ryder. 1979. Constructing the call graph of a program. _IEEE Transac-_
_tions on Software Engineering 3 (1979), 216–226._

[46] Jason Sawin and Atanas Rountev. 2011. Assumption hierarchy for a CHA call
graph construction algorithm. In 2011 IEEE 11th International Working Conference
_on Source Code Analysis and Manipulation. IEEE, 35–44._

[47] Tushar Sharma, Maria Kechagia, Stefanos Georgiou, Rohit Tiwari, Indira Vats,
Hadi Moazen, and Federica Sarro. 2021. A survey on machine learning techniques
for source code analysis. _arXiv preprint arXiv:2110.09610 (2021)._

[48] Olin Grigsby Shivers. 1991. _Control-flow analysis of higher-order languages or_
_taming lambda._ Carnegie Mellon University.

[49] Yannis Smaragdakis. 2021. Doop-framework for Java pointer and taint analysis
(using p/taint). _Retrieved Jan 10 (2021), 2021._

[50] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan
Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from
overfitting. _The journal of machine learning research 15, 1 (2014), 1929–1958._

[51] Li Sui, Jens Dietrich, Amjed Tahir, and George Fourtounis. 2020. On the recall of
static call graph construction in practice. In Proceedings of the ACM/IEEE 42nd
_International Conference on Software Engineering. 1049–1060._

[52] Tian Tan, Yue Li, and Jingling Xue. 2016. Making k-object-sensitive pointer
analysis more precise with still k-limiting. In Static Analysis: 23rd International
_Symposium, SAS 2016, Edinburgh, UK, September 8-10, 2016, Proceedings. Springer,_
489–510.

[53] Ewan Tempero, Craig Anslow, Jens Dietrich, Ted Han, Jing Li, Markus Lumpe,
Hayden Melton, and James Noble. 2010. The Qualitas Corpus: A curated collection of Java code for empirical studies. In 2010 Asia pacific software engineering
_conference. IEEE, 336–345._

[54] Akshay Utture, Shuyang Liu, Christian Gram Kalhauge, and Jens Palsberg. 2022.
Striking a balance: pruning false-positives from static call graphs. In Proceedings
_of the 44th International Conference on Software Engineering. 2043–2055._

[55] Raja Vallée-Rai, Phong Co, Etienne Gagnon, Laurie Hendren, Patrick Lam, and
Vijay Sundaresan. 2010. Soot: A Java bytecode optimization framework. In
_CASCON First Decade High Impact Papers. 214–224._

[56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. _Advances in neural information processing systems 30 (2017)._

[57] Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and
Steven CH Hoi. 2023. Codet5+: Open code large language models for code
understanding and generation. _arXiv preprint arXiv:2305.07922 (2023)._

[58] Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. 2021. CodeT5: Identifieraware Unified Pre-trained Encoder-Decoder Models for Code Understanding and
Generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural
_Language Processing. 8696–8708._

[59] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.
2019. Huggingface’s transformers: State-of-the-art natural language processing.
_arXiv preprint arXiv:1910.03771 (2019)._

[60] Tao Xie and David Notkin. 2002. An empirical study of java dynamic call graph
extractors. _University of Washington CSE Technical Report_ (2002), 02–12.

[61] Daoguang Zan, Bei Chen, Fengji Zhang, Dianjie Lu, Bingchao Wu, Bei Guan,
Yongji Wang, and Jian-Guang Lou. 2022. When Neural Model Meets NL2Code: A
Survey. _arXiv preprint arXiv:2212.09420 (2022)._

[62] Weilei Zhang and Barbara G Ryder. 2007. Automatic construction of accurate
application call graph with library call abstraction for Java. _Journal of Software_
_Maintenance and Evolution: Research and Practice 19, 4 (2007), 231–252._


-----

