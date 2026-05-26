# xLAM: A Family of Large Action Models to Empower AI Agent Systems

> **arXiv:** [2409.03215](https://arxiv.org/abs/2409.03215)
> **TeX source:** [arXiv-2409.03215v1/](arXiv-2409.03215v1/)
> **PDF:** [xLAM-arXiv-2409.03215v1.pdf](xLAM-arXiv-2409.03215v1.pdf)

---

**Figure:** *Figure* _(image: media/xLAM_2400px.png)_

## Abstract

Autonomous agents powered by large language models (LLMs) have attracted significant research interest. However, the open-source community faces many challenges in developing specialized models for agent tasks, driven by the scarcity of high-quality agent datasets and the absence of standard protocols in this area.
We introduce and publicly release **xLAM**, a series of large action models designed for AI agent tasks. The **xLAM** series includes five models with both dense and mixture-of-expert architectures, ranging from 1B to 8x22B parameters, trained using a scalable, flexible pipeline that unifies, augments, and synthesizes diverse datasets to enhance AI agents' generalizability and performance across varied environments.
Our experimental results demonstrate that **xLAM** consistently delivers exceptional performance across multiple agent ability benchmarks, notably securing the 1st position on the Berkeley Function-Calling Leaderboard, outperforming GPT-4, Claude-3, and many other models in terms of tool use. By releasing the **xLAM** series, we aim to advance the performance of open-source LLMs for autonomous AI agents, potentially accelerating progress and democratizing access to high-performance models for agent tasks.
\\\\
**Models:** [huggingface.co/Salesforce/xLAM-models](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4) \\
**GitHub:** [github.com/SalesforceAIResearch/xLAM](https://github.com/SalesforceAIResearch/xLAM)\\

% — BEGIN 1-introduction —

## Introduction

The field of autonomous agents has witnessed significant advancements in recent years, with large language models (LLMs) playing a crucial role in enhancing agent capabilities across diverse tasks. Researchers have made substantial progress in developing sophisticated frameworks [hong2023metagpt,xagent2023,wu2023autogen,xie2023openagents] and specialized environments [deng2023mind2web,yao2022webshop,zhou2023webarena] to enhance agent capabilities, such as tool use [qin2023toolllm] and web browsing [zhou2023webarena]. Concurrently, comprehensive benchmarks like AgentBench [liu2023agentbench], ToolBench [qin2023toolllm], and AgentBoard [ma2024agentboard] have been established to rigorously assess agent performance in reasoning, planning, and multi-turn interactions.

While proprietary LLMs developed by industry leaders have demonstrated competitive performance in various agent tasks [anthropic2024claude,openai2023gpt4,reid2024gemini,touvron2023llama], the open-source community faces limited choices for specialized models in this domain. This scarcity stems from several challenges in adapting open-source LLMs to agent tasks, primarily due to the lack of comprehensive, high-quality datasets and the heterogeneity of existing data formats. These factors complicate the unification of diverse datasets and obstruct the learning of transferable knowledge across different agent tasks.

Recently, the agent research community has intensified efforts in open-source agent data processing and model training [qin2023toolllm,chen2023fireact,xu2023lemur,patil2023gorilla,zeng2023agenttuning,yin2023lumos,zhang2024agentohana]. 
However, these works still face challenges in managing complex environments and generalizing to new scenarios, primarily due to limitations in the collected agent data.
A major obstacle is the homogeneity of content and format in existing datasets, resulting in models that lack diversity across various tasks and struggle to adapt to new or slightly different data structures in practical applications. While previous efforts have attempted to design pipelines for unifying data, they typically cover only a few scenarios or lack flexibility in their unified formats. For instance, Lumos [yin2023lumos] primarily addresses question answering, web agents, and mathematical tasks involving planning and grounding; while AgentOhana [zhang2024agentohana], despite encompassing a more diverse range of environments, lacks an extendable unified format to accommodate new environments.

Moreover, open-source datasets often suffer from quality issues, such as incorrect agent outputs, hallucinated actions, and repeated interaction turns within trajectories [zhang2024agentohana, chen2024agentflan]. The lack of detailed analysis and understanding of agent data further complicates these challenges, hindering the development of robust and versatile open-source agent models. Addressing these challenges is crucial for advancing the field of open-source agent models and bridging the performance gap with proprietary LLMs in agent tasks.

% — BEGIN figures_tables/fig_data_pipeline —

**Figure:** *Overview of the data processing, training and evaluation of xLAM. We take the diagnostic feedback from the model evaluation results to iteratively improve the data quality.* () _(image: media/xlam_release_v2_icon.drawio.pdf)_

% — END figures_tables/fig_data_pipeline —

% — BEGIN figures_tables/fig_gorilla_rank —

**Figure:** *An overview of xLAM model performances on the Berkeley Function Calling Leaderboard v2 (cutoff date 09/03/2024). Our 8x22b model secures the top-1 position with wide margin on the leaderboard.* () _(image: media/gorilla_rank_v2.pdf)_

% — END figures_tables/fig_gorilla_rank —

In this work, we introduce and open-source **xLAM**, a series of powerful models with varying sizes. This diverse set is tailored for a variety of applications, with smaller models (1B and 7B) optimized for on-device deployment, while larger models (8x7B and 8x22B) are designed to tackle more challenging tasks. Alongside the model release, we offer several insights and lessons learned from our experience in agent model training:

- **Data Processing:** We highlight the importance of data unification and augmentation in enhancing dataset diversity and mitigating overfitting. Our developed dataset preprocess and augmentation pipeline significantly improves the generalizability of agent models across diverse environments.

We evaluate the **xLAM** series on public agent benchmarks, demonstrating exceptional performance across various agent tasks. By open-sourcing these models, we aim to advance open-source agent models and provide valuable insights into data processing and synthesis techniques, addressing key challenges in developing competitive alternatives to proprietary models.

 
 
 

 
 

 
 

 

% — END 1-introduction —

% — BEGIN 2-related_work —

## Related Work

### LLM Agents

Recent advancements in LLMs have significantly enhanced their utility in various agent tasks. Several innovative prompt techniques have been developed to improve performance, including Chain of Thought (COT) [wei2022chain], ReACT [yao2023react], and Reflection [shinn2023reflexion]. Additionally, considerable efforts have been made to fine-tune open-sourced agent models for better capabilities [qin2023toolllm, chen2023fireact,patil2023gorilla, zeng2023agenttuning, zhang2024agentohana]. These include enhancements in data collection and processing to facilitate effective agent learning [zeng2023agenttuning, li2023api, tang2023toolalpaca, yin2023lumos, zhang2024agentohana, chen2024agentflan], covering a range from simple question answering to more complex scenarios like web interactions, tool operations, reasoning, and planning. 
However, many of these agent frameworks still depend on proprietary models as their core engine to achieve optimal performance, revealing a substantial gap in the availability of high-quality open-source models for these tasks.

### Agent Benchmarks

A variety of benchmarks have been established to assess the abilities of LLM agents across diverse scenarios [yao2022webshop, qin2023toolllm, liu2023agentbench,
ma2024agentboard,huang2023metatool, liu2023bolaa, wang2023mint, liu2024agentlite, du2024anytool, bfcl]. Notably, AgentBench [liu2023agentbench], Mint-Bench [wang2023mint], and AgentBoard [ma2024agentboard] encompass environments ranging from code generation and games to web interactions and reasoning tasks. ToolBench [qin2023toolllm] specifically evaluates multi-turn reasoning and tool-usage abilities, while the Berkeley Function-Calling Leaderboard [bfcl] broadly assesses models' capabilities in function calling across various contexts. These recent advancements in benchmarking have made the evaluation of agent models more accessible and standardized.

% — END 2-related_work —

% — BEGIN 3-data —

## Data Processing Pipeline

In this section, we discuss the data pipeline for training xLAM, including data unification, augmentation, quality verification, general instruction data synthesis, and preference data generation.

### Data Unification

Existing agent datasets are collected from diverse environments and designed in various formats, introducing noise and complicating data augmentation and verification. Models like NexusRaven [srinivasan2023nexusraven], Gorilla-Openfunctions [gorilla-openfunctions-v2], and AgentOhana [zhang2024agentohana] have demonstrated superior performance in function-calling, suggesting that a well-defined, universal format could significantly enhance model performance. By standardizing the format of existing data, we can reduce noise and facilitate easier data augmentation and quality verification, leading to a more efficient and robust framework for model training and evaluation. Furthermore, a standardized format ensures consistency, simplifies model training, and enhances the model's ability to generalize across various benchmarks.

Function-calling formats form the basis for how models understand and execute tasks, motivating us to design our unified data format in a function-calling style. 
As illustrated in Figure §fig:unified_format, 
the unified format consists of several modules: task instruction, available tools, format instruction, few-shot examples, query, and steps. 
Specifically, the available tools define the agent's action space, and the format instruction specifies the output format the agent should follow when generating a response. In each step, the agent's output, the environment's feedback/execution results, and the user's follow-up input are organized into a dictionary. 
It's quite common for there to be purely conversational interactions between users and agents that don't trigger any APIs or receive corresponding observations. In these instances, the related entry values would simply remain empty.

This unified format is compatible with various environments and tasks, making our data processing pipeline adaptable to different datasets and scalable to large amounts of data. Moreover, the modularized design allows for fine-grained data augmentation and quality verification, which are essential in improving agent data quality. For example, by unifying all the available tools and tool calls, we can easily inspect for hallucination and function-call errors, and apply various augmentation techniques.

### Data Augmentation

Our data augmentation strategy focuses on improving the diversity of the data. It involves applying various transformations to the existing dataset, thereby generating new, synthetic data samples. The data unification step significantly simplifies the application of various augmentation techniques. A standardized data format ensures consistency and ease of implementation, allowing for more efficient augmentation processes. Specifically, the augmentation techniques we adopted can be categorized as prompt format augmentation and instruction-following augmentation.

**Prompt Format Augmentation:** Prompt format augmentation focuses on creating various prompt formats based on the structured, unified data format. The format augmentation can be further divided into two categories: 1) *Order Shuffling*. In the unified format, the available tools are provided in a list, and each tool contains the name, description, and parameters. To avoid model overfitting to the specific order of the tools, we randomly shuffle the tool list. Furthermore, we also shuffle the order of the name, description, parameters, and within the parameters to present the information in different ways. We do the same thing within the tool_calls in each step. Additionally, we also shuffle the order of different sections of the input, including task instruction, tools, format instruction, few-shot examples etc. 2) *Concatenation Tokens*. Each training data point is a pair of input and output sequences. To convert the structured unified format to the training prompt, we use special tokens to concatenate different sections into one sequence. We create several different special token styles, including "[START/END OF QUERY]", "<query></query>", and plain text. 

**Instruction-Following Augmentation:**
Instruction-following augmentation focuses on adding diversity to the instructions in order to improve the model's instruction-following capability. It involves rephrasing existing instructions and adding new instructions, without introducing inaccuracy and inconsistency. Therefore, verification of the new instructions is a crucial step for this type of augmentation. We employ two methods for instruction-following augmentation: 1) *Task Instruction Rephrasing.* We rephrase the task instructions using powerful LLMs to accommodate various input styles from users. To ensure the rephrased instructions still align with the original version, we verify them by prompting the LLMs with the rephrased instructions and check if the LLMs can still follow them and generate correct function calls. 2) *Format Instruction-Following.* In our unified format, the output format is a JSON string with `thought` and `tool_calls`. To avoid the model overfitting on JSON format and to enable the model to follow various output formats upon different format instructions, we prepare 15 different output formats along with their corresponding format instructions and format converters. The output formats include JSON, XML, YAML, plain text, etc. 

### Data Quality Verification

To further understand of the data quality and to thoroughly investigate the sources of errors in the evaluation, we conduct a detailed analysis of the unified dataset. We identify a list of errors in the data using both rule-based and LLM-as-a-judge approaches.

**Undefined Function Call:** 
In function-calling, a list of available functions is provided, and the model should generate a function_call using one of the given functions. However, we found that in many cases, the predicted function_call is not from the given list. We match the predicted function with the given functions by comparing the function names and the list of parameter names. When the function_call name does not match any given functions, we refer to it as *Undefined Functions Invoked*. When the function name matches but the argument list contains undefined arguments, we refer to it as *Undefined Arguments Passed*. We also take into consideration optional parameters. 

**Incorrect Argument Type:**
Other than the error types mentioned above, we also observe that sometimes the model generates the correct argument's value, but in the wrong types. For example, when a parameter expects a `[val1, val2, val3]`, the generated arguments is `"[val1, val2, val3]"`, which is a string version of the list. When executing the function call, errors will occur due to incorrect data type. We identify trajectories containing the incorrect argument type error by comparing the parameter type in the available tools and the actual argument type. We also found that most argument type errors can be fixed by converting the arguments to the correct parameter types. 

**Argument Hallucination:** Upon examining the unified dataset from public sources, we discovered that tool calls frequently include argument values not present in the user query or prior steps. This issue arises because much of this data is generated by LLMs, which are prone to hallucination, a common problem in LLM-generated content.
We identified two types of hallucination: 1) the generated tool names or argument names do not appear in the provided tool and argument list; and 2) the argument values do not align with the user query or observations from previous steps. The first type of hallucination is straightforward to address by searching the generated tool call and argument names and matching them with the provided tool list, as they are all structured in JSON, making this process efficient. However, detecting the second type, where argument values are misaligned, is more challenging, as simple string matching is ineffective for complex queries and tasks. To tackle this, we use LLMs as judges to perform step-wise argument hallucination detection, detecting if there is a mismatch between the arguments and the intended query or prior observations.

**Low-Quality Reasoning and Planning:** We observe many data trajectories where the reasoning and planning steps are of low quality, which is a common issue in the outputs of many LLMs. To address this, we first filter out low-quality data using rule-based methods informed by heuristics, then prompt models like Mixtral-8x22b-Instruct-v0.1 [jiang2024mixtral] and DeepSeek-V2 [deepseekv2] to evaluate both the overall trajectory and individual thought steps on the selected data. A portion of these rating results is then sampled and verified by humans. We also attempted to iterate on this process using specifically fine-tuned models.

### Data Synthesis

Based on our findings in Sec. §sec:data-verification, we observe that most of these publicly available datasets have several limitations. First, these datasets are often static, synthesized by weak models, limited in scope, and, more importantly, not verified by execution. 
Second, these datasets mainly focus on a single type of function-calling category, i.e., outputting a single function call based on the provided tools. However, real-world scenarios might consist of many other types of use cases, such as the parallel function-calling scenario [bfcl], where the user query contains multiple requests and the model should respond with concurrent function calls in parallel within a single response.

To address these two issues, we utilize a systematic data synthesis framework called APIGen [liu2024apigen], which can generate verifiable datasets based on a collection of executable APIs.
The key idea is a multi-stage verification process to ensure the accuracy and quality of the generated data. This process includes format verification as introduced in Sec. §sec:data-verification, execution verification, and semantic verification, which collectively help to identify and filter out low-quality data points, such as those with hallucination issues or inaccurate argument parameter values.

 
We utilize over 3,673 APIs across 21 categories from ToolBench [qin2023toolllm] to generate a total of 60,000 high-quality data. These samples are generated using several strong open-source language models: DeepSeek-V2-Chat [deepseekv2] and Mixtral-8x22B-Inst [jiang2024mixtral]. 
This synthesis framework greatly improves the robustness and applicability of the dataset, as the majority of low-quality data can be identified by the multi-stage verification process. 

### Data Mixture

For supervised fine-tuning (SFT), our dataset combines training samples from three main sources: cleaned and augmented agent datasets, a synthetic function-calling dataset, and general instruction-tuning datasets. These sources are used to train the general xLAM models. 

Specifically, to enhance the general instruction capability of xLAM, we integrate diverse instruction-tuning datasets from DialogStudio [zhang2023dialogstudio] and Data Provenance [longpre2023data,longpre2024consent]. We employe rule-based techniques to filter out low-quality data, such as repetitive words and turns, which are common and often produced by less powerful models. We also remove data with inappropriate contents, responses and non-commercial licenses. Additionally, we deduplicate examples with similar user queries and organized the data by domain or category. We then prompt Mixtral-8x22b-Instruct-v0.1 and DeepSeek-V2 to assess both the entire dialogue and individual system responses on the selected data. This instruction data comprises 20% to 30% of our training set. To further enhance model robustness, we preserve the original formats of the general instruction-tuning data. 

To enhance the function-calling capability of xLAM-7b-fc-r and xLAM-1b-fc-r, we employ a targeted training approach, with 50% of their training data drawn from our high-quality synthetic function-calling dataset. The remaining 50% of the training data is sampled from other tasks within our training set.

For Direct Preference Optimization (DPO) [rafailov2023direct], we prompt less powerful models to generate and rate responses for selected data from each source, then sample a subset for human verification. After adjustments to models and prompts, we classify the selected responses as rejected samples. 

% — END 3-data —

% — BEGIN 4-method —

## Model Training

### Modeling

We use a supervised fine-tuning 
(SFT) approach, further aligning model checkpoints with the DPO method, and leverage the robustness of our flexible data pipeline. Our training code is based on the HuggingFace Transformers and Accelerate libraries[wolf2020transformers, accelerate], as well as PyTorch FSDP[zhao2023pytorchfsdpexperiencesscaling]. During training, the model undergoes multiple epochs, with datasets randomly shuffled each time. When using data parallelism across multiple devices, we diversify random seeds based on process IDs, ensuring balanced data distribution through partitioning, shuffling, and interleaving, thereby enhancing the robustness and reproducibility of our training process.

The fine-tuning of general xLAM models is conducted on Nvidia H100 GPUs. For SFT, we use a full fine-tuning framework that employs the fully sharded data parallel algorithm [zhao2023pytorch]. In the case of xLAM-8x22b-r, we integrate LoRA [hu2021LoRA,dettmers2023qLoRA] to better preserve the model's original capacities and prevent catastrophic forgetting [liu2023tail]. LoRA is also used for DPO alignment across all xLAM models. Additionally, we use a cosine learning rate scheduler with 100 warm-up steps to optimize performance.

The xLAM-FC models target various categories of function-calling agents, including simple, multiple, parallel, and parallel multiple. These categories are designed to enhance the models' performance in different scenarios. For instance, a simple query like retrieving the weather for a location (e.g., "What is the weather in Palo Alto today?") can be handled by calling `get_weather("Palo Alto", "today")`. Multiple queries involve selecting the appropriate function from several APIs, while parallel queries require executing multiple function calls simultaneously. Additionally, the models are trained in relevance detection to ensure alignment between function calls, execution results, and query objectives.

% — BEGIN figures_tables/tab_xlam_series —

**Table:** *Overview of xLAM model series.*

| lcccc
Model | Base Model | # Total Params | Context Length | Category |
| — | — | — | — | — |
| xLAM-7b-fc-r | DeepSeek-Coder-7b | 6.91B | 4k | Function-calling |
| xLAM-7b-r | Mistral-7b | 7.24B | 32k | General |
| xLAM-8x7b-r | Mistral-8x7b | 46.7B | 32k | General |
| xLAM-8x22b-r | Mistral-8x22b | 141B | 64k | General |

% — END figures_tables/tab_xlam_series —

### xLAM Model Series

We introduce a series of agent models tailored for different use cases. Our flagship model series, xLAM, is built upon the Mixtral Instruct [jiang2024mixtral] models and aims to achieve balanced performance across a diverse range of agent tasks, from complex multi-turn interactions to function-calling applications. To ensure its versatility, xLAM is trained on uniformly sampled data from our training dataset as introduced in Sec. §sec:training-data.

In addition to general xLAM models, we develop two specialized models for function-calling use cases, xLAM-7b-fc-r and xLAM-1b-fc-r, based on DeepSeek-Coder-7B-instruct-v1.5 and DeepSeek-Coder-1.3B-instruct, respectively [guo2024deepseek]. The smaller model sizes offer increased accessibility, allowing users to easily host them on a single GPU to address various function-calling tasks, ranging from simple user queries to parallel concurrent requests.

By offering a suite of models with varying sizes and specializations, the xLAM series caters to a wide range of user needs and computational resources, making powerful agent capabilities more accessible and adaptable to real-world applications.

% — END 4-method —

% — BEGIN 5-experiments —

## Experiments

### Benchmarks

After considering the stability of environments and research budget limitations, we evaluate the performance of models across four rigorous benchmarks: Webshop [yao2022webshop], ToolQuery [ma2024agentboard], ToolBench [qin2023toolllm], and the Berkeley Function-Calling Benchmark [bfcl]. Each benchmark is designed to assess different aspects of model capabilities under a variety of settings and constraints.

 **Webshop** is an interactive web environment designed to mimic online shopping experiences, testing an agent's ability to navigate and assist in e-commerce tasks. Webshop comprising approximately 250 test cases. 

 **ToolQuery** evaluates an agent's skills in using tools to retrieve and process information across domains. ToolQuery features 60 test cases across three distinct settings: Weather, Movie, and Academia. 

We use the testing configurations from AgentBoard [ma2024agentboard] for both Webshop and ToolQuery. These configurations assess overall performance using the Success Rate and evaluate progressive performance across interactive turns with the Progress Rate, with Success Rate being the more critic metric.

We additionally evaluate on **ToolQuery-Unified**, which is essentially ToolQuery but requires an agent to ingest the task instruction and tools following the augmented prompt format described in §ss:data_augmentation and likewise solve the task following the unified format. 
The purpose of testing agents in this setting is to assess their reasoning and tool-use abilities when evaluated on structured formats [tam2024let].

 **ToolBench** is developed for real-time evaluation of multi-turn reasoning and interactive capabilities via RapidAPI, and includes around 1,000 test cases. It uses Pass Rate as the metric, where the trajectory and final response are sent to GPT-4-0125-preview to determine whether the agent's final response successfully addresses the given user query. The evaluations cover both in-domain and out-of-domain settings, including unseen instructions with familiar tools, unseen tools within previously known categories, and entirely new categories of unseen tools.

 **Berkeley Function-Calling Leaderboard (BFCL) Benchmark** [bfcl] provides a comprehensive evaluation framework for assessing an agent's capability to reason about and execute function calls across a variety of programming languages and application domains. The benchmark comprises over 2,200 test cases, challenging models with complex scenarios such as parallel and multiple function calls in languages like Java, JavaScript, and Python. The evaluation metrics include Abstract Syntax Tree (AST) accuracy for non-executable test queries, executable accuracy by running APIs to obtain results, and a relevance detection score that measures the agent's ability to distinguish non-relevant queries and provided tools.

 Importantly, our evaluation utilizes the most recent BFCL v2 version, as of the cutoff date 09/03/2024. The v2 version introduces live function calls and real-world scenarios contributed by users, addressing issues such as data contamination, bias, and fairness by leveraging user-provided data. This updated dataset better reflects real-world distributions, characterized by a higher demand for selecting among multiple functions and a reduced demand for parallel function calls. For instance, our analysis indicates that in the v2 benchmark, the average number of available functions has doubled, while the average number of function calls has been halved compared to the non-live v1 data. It is important to note that all our models were trained prior to the release of the BFCL v2 live data.

### Experimental Results

#### Webshop and ToolQuery

% — BEGIN figures_tables/tab_agentboard —

**Table:** *Testing results on Webshop and ToolQuery. **Bold** and {0.55}
 
[width=0.99]{media/fig_model_comparison.png}
 
 
 
 


% — END figures_tables/fig_ablation_study —

We conducted an ablation study on the 7B models to measure the impact of various steps in our data pipeline.
Three datasets were prepared for this analysis: raw data, augmented data, and augmented + cleaned data. The raw data represents the dataset before data unification, while the other two datasets are post-unification.
Figure §fig:ablation presents the evaluation results of models trained on these three datasets. The metrics used for this evaluation are G1_instruction from ToolBench and success_rate from both Webshop and ToolQuery. The results indicate that augmented data consistently outperforms raw data across all metrics, with improvements of 2.3% on ToolBench, 5.8% on Webshop, and 18.3% on ToolQuery. Furthermore, the addition of data cleaning leads to a substantial performance increase on ToolQuery, with a further improvement of 23.4%. The results highlight the effectiveness of data augmentation and cleaning processes in the data pipeline.

% — END 5-experiments —

% — BEGIN 7-conclusion —

## Conclusion

This paper introduces xLAM series, a set of large action models for autonomous AI agents. Our models, ranging from 1B to 8x22B parameters, were trained with a scalable and flexible data pipeline that unifies, augments, and synthesizes diverse datasets. 
Our evaluations show that xLAM models consistently perform exceptionally across various benchmarks. 
The insights we learned from training these models highlight the importance of rigorous data processing and the potential of data synthesis in developing capable AI agents.
By releasing the xLAM series to the public, we aim to democratize access to high-performance models for agent tasks, thereby accelerating progress in the field. 

% — END 7-conclusion —




% — BEGIN appendix-2 —

## Appendix

% — BEGIN figures_tables/fig_unified_format —

**Figure:** *Unified function calling data format.* () _(image: figure)_

% — END figures_tables/fig_unified_format —

% — BEGIN figures_tables/fig_training_template —

}

**Figure:** *Example prompt and output for function-calling using xLAM.* () _(image: figure)_

% — END figures_tables/fig_training_template —

% — END appendix-2 —