# ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis

> **arXiv:** [2410.18447](https://arxiv.org/abs/2410.18447)
> **TeX source:** [arXiv-2410.18447v1/](arXiv-2410.18447v1/)
> **PDF:** [ToolFlow-arXiv-2410.18447v1.pdf](ToolFlow-arXiv-2410.18447v1.pdf)

---

## Abstract

Supervised fine-tuning (SFT) is a common method to enhance the tool calling capabilities of Large Language Models (LLMs), with the training data often being synthesized. The current data synthesis process generally involves sampling a set of tools, formulating a requirement based on these tools, and generating the call statements. However, tools sampled randomly lack relevance, making them difficult to combine and thus reducing the diversity of the data. Additionally, current work overlooks the coherence between turns of dialogues, leading to a gap between the synthesized data and real-world scenarios. 
To address these issues, we propose a Graph-based Sampling strategy to sample more relevant tool combinations, and a Planned-generation strategy to create plans that guide the synthesis of coherent dialogues. 
We integrate these two strategies and enable multiple agents to synthesize the dialogue data interactively, resulting in our tool-calling data synthesis pipeline . Data quality assessments demonstrate improvements in the naturalness and coherence of our synthesized dialogues.
Finally, we apply SFT on LLaMA-3.1-8B using 8,000 synthetic dialogues generated with . Results show that the model achieves tool-calling performance comparable to or even surpassing GPT-4, while maintaining strong general capabilities.

% — BEGIN chapters/introduction —

## Introduction

Enabling Large Language Models (LLMs) to perform tool calling significantly enhances their capabilities and practical applications. This requires the models to possess strong understanding, reasoning, and instruction-following abilities.
Customized fine-tuning is a widely used method to improve the tool-calling capabilities of LLMs [abdelaziz2024granitefunctioncallingmodelintroducing, patil2023gorillalargelanguagemodel, schick2023toolformerlanguagemodelsteach, qin2023toolllmfacilitatinglargelanguage]. However, access to fine-tuning data can be limited. One viable solution is to utilize LLMs for data synthesis [basu-etal-2024-api, wang2023selfinstructaligninglanguagemodels, xu2023wizardlmempoweringlargelanguage, yu2024metamathbootstrapmathematicalquestions].

A typical tool-calling data synthesis process involves three steps: (1) selecting candidate tool(s), (2) generating requirements based on those tools, and (3) creating the call statements [tang2023toolalpacageneralizedtoollearning, liu2024apigenautomatedpipelinegenerating]. However, the data synthesized through this method often lacks realism and naturalness. Randomly sampled tools frequently fail to interconnect, making it difficult to combine them for complex tasks. Consequently, the requirements for subsequent synthesis tend to be simplistic, which reduces the diversity and complexity of the data. Furthermore, much of the existing research focuses solely on generating single-turn tool-calling instructions, neglecting the coherence between dialogue turns [qin2023toolllmfacilitatinglargelanguage,yang2023gpt4toolsteachinglargelanguage]. In real-world interactions, LLMs typically engage with users through dialogues rather than single-round Q&A sessions. This creates a gap between Q&A-type training data and its practical application, ultimately diminishing the naturalness of the synthesized data.

To address these two challenges, we propose , a tool-calling data synthesis pipeline that employs a graph-based sampling algorithm to improve the correlation among the selected tools and a planned-generation strategy to enhance the naturalness and coherence of the synthesized tool call dialogues.

Specifically, we consider tools with similar parameters or return values to be related. For instance, both "*book_flight*" and "*get_weather*" require parameters related to location. In practical scenarios, these two tools are indeed interconnected, as they often occur together in travel contexts. Based on this assumption, we construct a tool graph that represents the similarity between parameters and return values of the tools. Each node in the graph represents a tool, while the edges indicate the relevance between pairs of tools. When sampling tools, we randomly select a subgraph from this tool graph, ensuring that the sampled tools are more likely to interact effectively, thereby facilitating the generation of complex requirements.

On the other hand, before synthesizing dialogues, we first have the LLM create a plan based on the selected subset of tools. This plan outlines the requests that users need to make in each turn of the dialogue. While constructing the plan, the model focuses on establishing the dialogue framework without worrying about phrasing and details. This approach allows the model to concentrate on the logical relationships and interactions between requirements, resulting in more coherent demands.
Additionally, we enable the LLM to incorporate non-tool-call requests into the plan. This not only enhances the diversity of the conversation content but also facilitates seamless transitions between topics, naturally leading to new requirments.

We generate dialogues using three agents: *User*, *Assistant*, and *Tool*. Based on the selected tool subset and the established plan, these agents interact to complete the dialogue. By iterating through the "sampling-planning-generation" process, we synthesized a total of 8,000 dialogues.
To evaluate the effectiveness of our proposed method, we conduct a comprehensive ablation study on the graph-based sampling and planning strategy by generating dialogues of the same size selectively without these modules. We perform a thorough evaluation of the data quality, which demonstrates that can effectively enhance the naturalness, coherence, and diversity of the generated dialogues. Finally, we apply supervised fine-tuning to LLaMA-3.1-8B-Instruct [dubey2024llama3herdmodels] using the synthesized data and validate improvements in the model's tool-calling capabilities while preserving its general abilities, with .

We summarize our contributions into the following three key points: 

- We propose a Graph-based Sampling strategy to select related tools, aiming to enhance the diversity and complexity of synthetic tool calling requirements.
- We introduce a Planned-Generation strategy to improve the naturalness and coherence of synthetic dialogues.

% — END chapters/introduction —

% — BEGIN chapters/related_work —

## Related Works

Integrating external tools with large language models (LLMs) significantly broadens their functional scope, allowing for more specialized, precise, and reliable solutions to complex problems [qin2023toolllmfacilitatinglargelanguage]. There are generally two main strategies for embedding tool-use capabilities into LLMs: prompt-based methods and tool-augmented SFT. Prompt-based methods enable LLMs to utilize tools by providing descriptions and examples of the tools in the prompt, without any incremental training [ruan2023tptulargelanguagemodelbased,hsieh2023tooldocumentationenableszeroshot, mialon2023augmentedlanguagemodelssurvey]. 
ReAct [yao2023reactsynergizingreasoningacting] is one of the notable methods within this category. It allows LLMs to switch between reasoning and executing actions to tackle challenging tasks. However, the effectiveness of these approaches can be limited by the inherent capabilities of the model.
On the other hand, tool-augmented tuning is attracting increasing interest due to its direct enhancement of LLM’s tool usage capabilities [abdelaziz2024granitefunctioncallingmodelintroducing, patil2023gorillalargelanguagemodel, schick2023toolformerlanguagemodelsteach, qin2023toolllmfacilitatinglargelanguage]. As the limited availability of tool calling datasets, basu-etal-2024-api [basu-etal-2024-api] adapted data from various other domains for application in tool calling studies. Others [liu2024apigenautomatedpipelinegenerating, tang2023toolalpacageneralizedtoollearning, qin2023toolllmfacilitatinglargelanguage] mainly synthesized single-turn instructions that involve basic tool calling requirements. However, LLMs typically interact with users through dialogue rather than single-turn Q&A. This mismatch means the data is unnatural, creating a gap with real-world scenarios. Our focuses on enhancing the coherence and naturalness of dialogues in data synthesis, making it more aligned with actual applications.

% — END chapters/related_work —

% — BEGIN chapters/methodology —

## Methodology

**Figure:** *The pipeline of dialogue synthesis. The left side shows the Tool Graph with blue boxes representing tools and purple boxes representing parameters or return values. In the middle is the dialogue synthesis plan generated according to sampled tools. On the right is an example of data synthesis by the *User*, *Assistant*, and *Tool* agents.* () _(image: images/pipeline-c.pdf)_

To generate realistic and coherent dialogues, we propose a three-step data synthesis process: (1) Selecting a tool subset using graph-based sampling; (2) Generating a dialogue plan based on the selected tool subset; (3) Synthesizing dialogues guided by the tool subset and dialogue plan.

### Graph-based Sampling for Tool Selection

Tool calling data synthesis generally starts by selecting one or more tools from the available toolset. While much of the previous work overlooks the significance of tool selection [qin2023toolllmfacilitatinglargelanguage, patil2023gorillalargelanguagemodel], relying solely on random sampling, the chosen tools play a crucial role in shaping the quality of the synthesized dialogues. In real-world scenarios, user requirements often necessitate the combined use of multiple tools to achieve a solution. To synthesize more complex user needs, it is vital to ensure that the selected tools can work together. To address this, we propose a Graph Sampling strategy to identify relevant and compatible tool combinations.

#### Tool Graph Construction

We first construct a graph, $G = (V, E)$, where node $v_iV$ represents the tool $i$, and the edge $e_{i,j}E$ represents whether tool $i$ and tool $j$ are related. The left side of Figure §fig:pipeline shows an example of the tool graph.
We consider tools with similar parameters or return values to be related to each other:

**P-P Similarity:**
When two tools share similar parameters, there is a high probability that the tools are related. For instance, based on "*location*" and "*destination*", two semantically similar parameters, we can identify tools like "*get_weather*" and "*book_flight*", which are frequently used together in travel-related contexts.

**P-R Similarity:**
If the return value of one tool is similar to the input parameters of another, there is also a high likelihood that the two tools are related. For example, the *"check_calendar"* tool typically returns the location of events, while the *"navigate"* tool requires a location as input. When a user requests to *"navigate to the location of this afternoon's meeting,"* both tools would be called.

To derive similarity between parameters or return values, we first concatenate the name and description of a parameter or a return value using the template *"{Name\*: {Description}"}. For example, the parameter "Date" of one specific tool is represented as *"Date: Departure date, format as dd/mm/yyyy."* 
Then, we encode these strings using `Sentence-BERT` [reimers2019sentencebertsentenceembeddingsusing] to obtain the corresponding embeddings.
We use $p^i_k$ and $r^i_l$ to denote the $k$-th parameters and $l$-th return values of the tool $v_i$, respectively. And we use $^i_k$ and $^i_k$ to represent the embeddings of $p^i_k$ and $r^i_l$, respectively.
The similarity between $v_i$'s parameter $p^i_k$ and $v_j$'s parameter $p^j_l$ is defined as:

$$
 (^i_k, ^j_l) = ^i_{k} ^j_{l}}{\|^i_{k}\| \|^j_{l}\|}
$$

Similarly, the similarity between $v_i$'s return value $r^i_k$ and $v_j$'s parameter $p^j_l$ is defined as $(^i_k, ^j_l)$.
If the similarity is greater than a predefined threshold $$, we consider the two tools to be correlated. We set $$ to be 0.82 according to our preliminary study. 
This means that when the similarity between any pair of parameters from two tools exceeds $$, or when the similarity between a return value and an input parameter of two tools exceeds $$,
we assign an edge between the two tools:

$$
e_{i, j} = 
 
 1, & p^i_kv_i, p^j_lv_j : (^i_k, ^j_l) > \\
 &  \\
 & r^i_kv_i, p^j_lv_j : (^i_k, ^j_l) > \\
 0, & 

$$

where $i, j=1... N$ and $i j$. We use $$ to represent a parameter or a return value is included in the tool.

#### Graph-based Sampling

With the constructed tool graph, we are able to sample a subset consisting of $n$ tools that might be correlated. Generally, we randomly select a node as the starting point on the graph, and then perform a random walk along the edges of the graph. We stop when the path length reaches $n$, and the nodes included in the path constitute the sampled subset of tools. Details are shown in Algorithm §alg:graph_sample.



### Dialogue Plan Generation

Coherent dialogues usually involve complex tool-calling scenarios, such as cases where the current tool calling relies on the return value of a previous one or where parameters for the current tool calling are already present in the dialogue history. Moreover, realistic dialogues between humans and AI assistants often don’t always require tool use; they frequently involve non-tool-related exchanges, such as chitchat, interspersed with tool-based tasks. To enhance an LLM's performance in such realistic scenarios, training examples that reflect this balance are essential.

To address the need and enhance the coherence of synthesized dialogues, we propose a Planned-Generation strategy, which indicates planning before generating. We have the LLM first formulates a set of user requirements based on the tool subset to create a dialogue plan. These requirements can involve tool call requests – tasks that necessitate the use of these tools – or non-tool interactions, such as chitchat. The middle part of Figure §fig:pipeline shows an example of a plan. At this stage, the LLM focuses solely on the logic, coherence, and natural flow of the requirements, without delving into the phrasing of interactions or other nuances.
Compared to directly synthesized dialogues, those generated based on a dialogue plan show markedly improved coherence. We provide a detailed assessment of this coherence in the subsequent sections. Please refer to Table §tab:plan_prompt for the plan synthetic prompt.

### Multi-Agent Dialogue Synthesis

We set up three agents, *user*, *assistant*, and *tool*, with LLMs to collaboratively synthesize dialogues. The right side of Figure §fig:pipeline illustrates the synthesis process for one dialogue turn.

The *user* agent is responsible for initiating requests based on the dialogue plan. It first checks whether the current request in the plan has been completed, which determines whether to continue with the current task or move on to the next one. This ensures that the dialogue stays aligned with the plan's flow and sequence.

The *assistant* agent evaluates the user’s request to determine if a tool is required. In cases where no tool is needed, such as chitchat, the assistant agent responds directly. If a tool call is necessary, the assistant verifies whether all required parameters are present based on the tool documentation. If any parameters are missing, the assistant requests clarification from the user; otherwise, it generates the tool call statement.

The *tool* agent simulates the return values of the requested tool based on the tool documentation and the assistant's call statement. 

The interaction among the three agents continues for each dialogue turn until all the requests in plan are addressed or the preset turn limit is reached.
Afterward, all dialogue turns are collected and a rule-based data filtering module is applied to remove low-quality data [liu2024apigenautomatedpipelinegenerating, liu2024toolacewinningpointsllm]. 
The filtering rules primarily check the format of tool call statements, as well as other issues such as incomplete dialogues or missing tool call turns.

### Implementation Details

In this work, for tool selection, we directly utilize the tools from ToolBench [qin2023toolllmfacilitatinglargelanguage] (including over 16,000 RESTful APIs) as our available tools. To standardize tool descriptions, we follow the setting of OpenAI Function Calling [^note: https://platform.openai.com/docs/guides/function-calling] and prompt an LLM (Llama-3.1-8B) to convert all these tools into JSON format. For cases where information is incomplete, such as missing parameter descriptions, we rely on the LLM to infer and fill in the missing details during the conversion process. A demonstration tool is presented in Figure §fig:tool_demo.

We use GPT-4 [openai2024gpt4] for all generative tasks, including dialogue plan generation and agent simulation, unless otherwise specified. Multiple versions of GPT-4 are randomly selected for each dialogue, potentially enhancing diversity.



## Data Quality Assessment

### Basic Data Information

In this section, we evaluate the quality of the synthetic data. To assess the effectiveness of the Graph-based Sampling (referred to as Graph) and the Planned-Generation strategy (referred to as Plan), we synthesized three additional sets of comparative data under different conditions: removing Graph, removing Plan, and removing both Graph and Plan. Each dataset contains 8,000 dialogues. Table §tab:statistics presents the total number of tokens, the number of tool calls, and the number of dialogue turns containing tool calls for these datasets.

As shown in Table §tab:statistics, the total number of tokens generated by the different strategies is similar, at approximately 8 million. Dialogues synthesized using the Planned-Generation strategy include more non-tool interactions, resulting in a lower proportion of tool call requests. In contrast, the Graph-based Sampling strategy increases the number of tool calls. 
This can be attributed to the connections among tools, where relevant information for subsequent tool calls is generally contained in the dialogue history, thereby reducing the need for additional turns to ask for missing information.

### Quality Evaluation

To further assess the quality of the synthetic dialogues, we implemented both an automatic evaluation and a model-based evaluation. 

The automatic evaluation primarily assesses the coherence and diversity of the dialogues. 
Following dziri-etal-2019-evaluating [dziri-etal-2019-evaluating], we assess the coherence of the dialogue as a Natural Language Inference (NLI) task. 
We treat two consecutive turns in the dialogue as the premise and hypothesis, respectively, and calculate the ratio of entailment relation (*EnR*) as well as the semantic similarity (*SS*) between them. A higher EnR or SS between turns indicates that the dialogue is more coherent.

Regarding diversity, we calculate the text's Shannon entropy (*H*) based on the word frequency [6773024]. 
We also compute the Distinct-N Score [li-etal-2016-diversity] for the dataset, with $N=3$ (*D-3*). Higher entropy or Distinct-N Score indicates that the dataset contains more information and has greater diversity.

In addition, we randomly sampled 200 dialogues in each dataset for the model-based evaluation. We used GPT-4 [openai2024gpt4] to carefully evaluate each dialogue based on four dimensions: naturalness (*NAT*), coherence (*COH*), helpfulness (*HELP*), and accuracy (*ACC*). The prompt for GPT-4 evaluation is shown in Table §tab:eval_prompt.

The evaluation results are shown in Table §tab:data_quality. There are two key observations:

- *H* and *D-3* Score demonstrate that Graph Sampling enhances the diversity of the data.

For more detailed settings, analysis and explanations, please refer to Appendix §sec:data_quality_assess_app.



### Comparison with Natural Dialogue Dataset

To better understand how our synthetic dialogues compare with human-created ones, we conducted a comparative study with an established dataset. We chose the MultiWOZ dataset [budzianowski-etal-2018-multiwoz] as a natural dialogue dataset for comparison. MultiWOZ (Multi-Domain Wizard-of-Oz) is a well-known task-oriented dialogue dataset, and we believe comparisons with this dataset would be convincing. We repeated the GPT-4 evaluation experiment by first randomly sampling 200 dialogues from MultiWOZ. Then, using the same scoring prompt, we had GPT-4 evaluate these dialogues across the four dimensions. The results are shown in Table §tab:multi_woz_comp.

MultiWOZ scores slightly higher on naturalness, coherence, and accuracy compared to the ToolFlow dataset, though the differences are minimal (average score differences between 0.1-0.2). Regarding helpfulness scores, ToolFlow outperformed MultiWOZ by 0.3 points. These results suggest that our synthetic dialogues in *ToolFlow* achieve comparable quality to human-created dialogues in MultiWOZ, with particularly strong performance in task-oriented aspects such as helpfulness. 


% — END chapters/methodology —

% — BEGIN chapters/experiment —

## Experiments

### Settings



#### Datasets
 We conducted experiments on the following three tool-calling datasets to validate the tool call capability of the model trained with .

- **BFCL-v2** [patil2023gorillalargelanguagemodel] primarily consists of Python-style tool call data, divided into four categories *Simple*, *Multiple*, *Parallel*, and *Parallel Multiple*. Version 2 adds more data from dynamic, real world scenarios. We selected the categories that can be evaluated with the Abstract Syntax Tree (AST), which are statistically stable and easy to evaluate.
 The accuracy is reported.
- **API-Bank** [li-etal-2023-api] is a dialogue-style tool call dataset, including two settings: *Call* and *Retrieve + Call*. The model is required to call predefined local Python tools based on user requirements in the dialogue. Accuracy is measured by evaluating whether the tool return values match the ground truth.

Additionally, to examine changes in general performance, we evaluated the model's reasoning and conversational abilities using MMLU [hendrycks2021measuring], BBH [suzgun2022challenging], and MTBench [10.5555/3666122.3668142].

#### Models

In our main experiments, we use LLaMA-3.1-8B-Instruct [dubey2024llama3herdmodels] as base model to examine the effectiveness of the synthetic dialogues generated with . For simplicity, we use to refer to the fine-tuned model throughout the remainder of this paper.
The models we compared include GPT-3.5, GPT-4, GPT-4o [openai2024gpt4], Claude [bai2022constitutionalaiharmlessnessai], LLaMA-3.1 [dubey2024llama3herdmodels], etc, as well as baselines from the paper of the evaluated datasets, such as Lynx-7B [li-etal-2023-api] and ToolAlpaca-7B [tang2023toolalpacageneralizedtoollearning], etc. For specific checkpoint information, please refer to the experimental result tables.

### Results

#### achieves tool-calling capability comparable to GPT-4o.

We evaluated 's tool-calling ability on the BFCL. This dataset contains questions from four categories.
In the **Simple** category, each question contains one tool, which the LLM must correctly call based on requirements. The **Multiple** question includes 2-4 tools, requiring the model to choose and call the most suitable one. In the **Parallel** category, several tools should be called in one turn. **Multiple Parallel** adds distracting candidate tools to the Parallel setup. 

The results are shown in Tabel §tab:bfcl. Overall, achieves performance comparable to GPT-4o. On the Non-Live subset, outperformed GPT-4 and GPT-4o, but was slightly weaker than Claude-3.5-Sonnet. On the Live subset, still lags behind these leading closed-source LLMs. 
This is because the Live subset added more user-contributed test cases from the real world, thus making it more challenging for the model.
We attribute this gap primarily to differences in model size, given that only has 8B parameters.

The ablation experiment shows that the model trained on data synthesized by strategies including both Graph-based Sampling and Generated Plan performs the best. This is especially evident in Parallel Multiple type questions.

While different models or training strategies exhibit some variance in certain categories, the differences in average performance are not significant. Therefore, we conducted further comparative experiments on additional tool call datasets.

#### achieves SOTA on dialogue data.

BFCL tests the model's tool calling capability in the form of Q&A. However, we believe that a conversational format is closer to real-world application scenarios. 
On the other hand, our synthesized training data is also in the form of dialogue. Therefore, in the BFCL test, the advantage of cannot be fully demonstrated.

API-Bank is a dialogue dataset. During evaluation, the model needs to make tool call requests after receiving user demands and provides a response based on the tool's return value. This process may occur multiple times within a single dialogue. It includes two test settings: *Call* and *Retrieve + Call*. In the *Call* setting, the assistant selects tools from a candidate tool set to fulfill user requests. In the *Retrieve + Call* setting, the assistant only has access to a search tool. The assistant needs to search for the relevant tools first, and then call them. 



From the results in Table §tab:api_bank, achieves state-of-the-art average accuracy. Under the *Call* setting, outperforms all baselines. In the *retrieve + call* setting, is inferior to GPT-3.5-turbo but superior to other baselines, including GPT-4.
The ablation experiments show that Graph-based Sampling strategy can improve the model's accuracy in tool call under this setting. This is because tools obtained through Graph sampling often have sequential correlations. As a result, the training data includes more examples of sequentially calling tools, aligning better with the requirements of the *retrieve + call* setting.



#### can correct mistakes based on error messages.

Correcting errors is a key capability of LLM tool calls [wang2024llmsimaginariumtoollearning]. We conducted the tests in the *Simulated* setting of ToolAlpaca dataset. This dataset established a simulation environment that utilizes GPT-4 to mimic the return values of tools, including the error messages when calls fail. The model is allowed to self-correct based on these error messages and then retry the call. We assess the tool call and correction capabilities of on this dataset.

The dataset evaluates the accuracy of the tool call *Procedure* and the model's final *Response*. The procedure is considered accurate when the model's call matches the ground truth. The response is considered accurate when the model's response can satisfy the user’s instruction. If they are both accurate, the model's *Overall* performance is considered accurate. This evaluation was conducted by GPT-4. We presented the results in Table §tab:tool_alpaca.

's *Procedure* accuracy reached 85%, surpassing GPT-3.5's 77%. In *Procedure* evaluation, correcting errors is considered as redundant actions and therefore judged as incorrect. Hence, this accuracy implies that in most cases, the first tool call of is accurate. On the other hand, 's *Response* accuracy of 88% is higher than the *Procedure* accuracy of 85%, indicating that corrected errors in some test cases. This suggests that has the ability to self-correct based on error messages, even though error correction samples are not included in the training data.

#### 's General Ability Is NOT Compromised by Fine-tuning.

The fine-tuned model risks catastrophic forgetting, where the capability for tool call is enhanced, but other abilities decline. As an AI assistant, LLM's reasoning and conversational abilities are equally important. Therefore, we tested the tuned model on the MMLU, BBH, and MTBench datasets to examine whether catastrophic forgetting issues have occurred. The results are shown in Table §tab:mtbench_mmlu_bbh.

The test results on MMLU and BBH show that there is no significant difference in performance between the models before and after training. However, on the MTBench dataset, models trained on data without Graph-based Sampling or Plan-Generation exhibited a decline in performance.
Notably, in the evaluation of *Turn 2*, models trained on synthetic data using the Plan-Generation strategy exhibited a slight performance improvement. This improvement is due to Plan-Generation enhancing the naturalness and coherence of synthetic dialogues, thereby boosting the model's conversational abilities.


% — END chapters/experiment —

% — BEGIN chapters/analysis —

## Correlation Analysis

To further investigate the impact of the diversity and coherence of the dialogue data on model performance, we conducted additional correlation analysis. In the previous experiments, we synthesized a total of $8,000 4 = 32,000$ dialogues by using and its ablation settings. We randomly sampled from these data 10 times, each time selecting 4,000 dialogues to form 10 new training sets. We calculated the diversity metrics *D-3* and H, and the coherence metrics *SS* and *EnR* for each dataset. 
Then, we used this data to fine-tune the Llama3.1-8B-Instruct. We tested these ten fine-tuned models on the BFCL and MTBench. 
Finally, we calculated the Pearson correlation coefficient between the evaluation metrics of the training data and the model performance and reported it in Table §tab:corr.

The average results on the BFCL show that both diversity and coherence of the training data contribute a lot to enhancing the model's tool-calling capabilities. MTBench results show a strong positive correlation between data coherence and the model's conversational performance, consistent with our assumption. Notably, while we use entropy and Distinct-N scores to assess diversity, the inconsistent correlation between these metrics and model performance suggests they may reflect different dimensions of diversity.
On the other hand, coherence does not appear to positively impact the parallel test sets in the BFCL, likely due to the nature of these tests involving multiple calls within a single turn.
Nevertheless, while our has demonstrated the benefits of increasing diversity and coherence in Section §sec:quality_evaluation, the correlation results in this section further validate their positive effects on overall performance.



% — END chapters/analysis —

% — BEGIN chapters/overlap_analysis —

## Dataset Overlap Analysis

To ensure the reliability of the evaluation, we conducted an overlap analysis between training and test datasets. This examination helps verify the independence of these test data and prevents potential data leakage issues. We employed both N-gram-based and similarity-based methods to demonstrate that there is no significant data leakage in the *ToolFlow* dataset. We also included the well-known xLam agent training set [zhang2024xlamfamilylargeaction] as a control group for comparison.

**N-gram-based method** Following the approach used in LLaMA-2 [touvron2023llama2openfoundation], we considered a token contaminated if it appeared in any token n-gram longer than 10 tokens in both the evaluation sample and the training set. A tool was classified as leaked if more than 10% of the tokens in its JSON string were contaminated.

**Similarity-based method** We defined a tool as leaked if the cosine similarity between the given tool and any tool in the evaluation dataset exceeded 0.9. We used the 

`all-MiniLM-L12-v2`
encoder from HuggingFace [^note: https://huggingface.co/] to obtain representations for all tools.

We present the proportions of data leakage across different evaluation metrics in Table §tab:dataset_overlap. These results suggest that there is no severe data leakage between *ToolFlow* as a training set and the test sets.

% — END chapters/overlap_analysis —

% — BEGIN chapters/conclusion —

## Conclusion

In this work, we propose Graph-based Sampling and Planned Generation strategies to enhance the diversity and coherence of synthetic data. Based on these two strategies, we introduce a pipeline called for synthesizing tool calling data and generate 8,000 training samples. Using this dataset, we conduct SFT on Llama3.1-8B-Instruct, resulting in improved tool calling capability of the model. Subsequently, we conduct correlation analysis to demonstrate the influence of data diversity and coherence on model performance. This provides a reference for the composition of training data for the tool-enhanced agent.

% — END chapters/conclusion —

% — BEGIN chapters/limitation —

## Limitations

We summarize the limitations in two points. 

As described in Section §sec:imp_details, the seed data is a pre-collected tool set including 16,000 APIs. Although our can synthesize more diverse data, it is undeniable that the size and diversity of the tool set also affect the diversity of the data. However, how to enrich the seed data has not yet been studied in this work.

On the other hand, utilizes GPT-4 for data synthesis, and then uses this data to train a 8B-model. Therefore, it still falls under the paradigm of using strong models to train weak models. Whether the model can be improved by training on its own synthesized data is still unknown. We believe that this weak-to-strong setting is more challenging but also more meaningful.

% — END chapters/limitation —

% — BEGIN chapters/ethic_statement —

## Ethic Statement

In this research, GPT-4 was employed as an evaluator and generator in a manner consistent with ethical guidelines. Transparency about its usage, accountability for its outputs, and mitigation of potential biases were prioritized. Data privacy and security were strictly maintained, and the AI's limitations were acknowledged, ensuring it supplemented rather than replaced human judgment. This approach aimed to enhance the research quality while upholding academic integrity and ethical standards. 

% — END chapters/ethic_statement —

% — BEGIN chapters/acknowledgement —

## Acknowledgements

This work was partially supported by Hong Kong RGC GRF No. 14206324, CUHK direct grant No. 4055209, and CUHK Knowledge Transfer Project Fund No. KPF23GWP20.
% — END chapters/acknowledgement —



% — BEGIN chapters/appendix —

## Appendix

### Details of Data Quality Assessment

Following dziri-etal-2019-evaluating [dziri-etal-2019-evaluating], we converted the dialogue data into a Natural Language Inference (NLI) format. In this format, the request and response from the previous dialogue round serve as the **premise**, and the current round's request serves as the **hypothesis**. We then use a trained BERT [reimers2019sentencebertsentenceembeddingsusing] to predict the relationship between the two, and we calculate the ratio of entailment predictions (EnR). A higher proportion indicates greater coherence between consecutive dialogue rounds. Additionally, we measure the semantic similarity (SS) between the premise and hypothesis. We extract sentence representations using BERT and compute their cosine similarity. A higher similarity score indicates a more coherent dialogue.

Regarding diversity, we calculate the text's Shannon entropy (*H*) based on the word frequency. 
We also compute the Distinct-N Score [li-etal-2016-diversity] for the dataset, with $N=3$ (*D-3*). Higher entropy or Distinct-N Score indicates that the dataset contains more information and has greater diversity. In addition, we sampled 200 dialogues in each set. We used GPT-4 to carefully evaluate each dialogue based on four dimensions: naturalness (*NAT*), coherence (*COH*), helpfulness (*HELP*), and accuracy (*ACC*). The prompt for GPT-4 evaluation is shown in Table §tab:eval_prompt.

Automatic evaluation indicates that the Planned-Generation strategy enhances conversation coherence. Both coherence metrics, SS and EnR, reflect this improvement. Intuitively, the plan is carefully designed by the model in advance, leading to more coherent dialogue. On the other hand, the Graph Sampling strategy can increase data diversity. This is because the strategy samples tools with strong associations, and the combination of these tools enhances data diversity.

GPT-4's evaluation indicates that the Planned-Generation strategy enhances the naturalness of dialogue. This metric assesses whether a dialogue could realistically occur in the real world. In data synthesized without a plan, most user requests are tool calls with little chitchat, which is uncommon in real-world scenarios, resulting in lower naturalness. GPT-4's coherence evaluation closely aligns with automatic assessments. In terms of helpfulness, both Graph Sampling and Planned-Generation strategies show improvement. The low scores in Helpfulness are mainly due to the assistant frequently asking follow-up questions about parameters, which consumes dialogue turns. These strategies help reduce such behavior. The accuracy of synthesized data is high across all four settings, likely due to pre-filtering with quality control tools.

### Reliability Analysis of GPT-4 as an Evaluator

To validate the reliability of GPT-4's evaluations, we conducted additional human evaluations. Specifically, we sampled 50 examples from the data previously evaluated by GPT-4 for human assessment. We recruited four Computer Science PhD students to rate these 50 samples using the same criteria as GPT-4: naturalness, coherence, helpfulness, and accuracy. We collected the human evaluation results and calculated both the Cohen's Kappa inter-rater agreement scores among evaluators and the Pearson correlation coefficients between human and GPT-4 evaluations.


The results in Table §tab:human_eval demonstrate high inter-rater agreement and strong correlations with GPT-4's evaluations, particularly for helpfulness and accuracy metrics. While the agreement and correlations for naturalness and coherence were relatively lower compared to the other two metrics, they still maintained a minimum correlation of 0.61, indicating a strong positive correlation between human and GPT-4 evaluations. We believe these results substantiate the reliability of GPT-4's evaluations.

### GPT-4 v.s. Open Source LLM for Data Synthesis

To evaluate the reliance of synthesis on closed-source, high-performance LLMs (e.g., GPT-4), we conducted comparative experiments using open-source alternatives. Following the same pipeline but replacing GPT-4 with LLaMA-3.1-8B-instruct, we conducted preliminary synthesis experiments. While still in the initial validation phase, we have synthesized 4,235 dialogue instances. We replicated experiments using this data to fine-tune LLaMA-3.1-8B-instruct, with BFCL test results shown in Table §tab:comp_gpt4_llama.



Here, "N/A" represents the baseline performance of LLaMA-3.1-8B-instruct (consistent with results reported in the main text). Notably, we leveraged existing results from Section 6's Correlation analysis, where we had downsampled to ten subsets of 4,000 dialogues each, comparable to our LLaMA-3.1-8B-instruct synthesized dataset size. The table reports the average results across these ten subsets. While data synthesized by LLaMA-3.1-8B-instruct indeed shows lower performance compared to GPT-4-synthesized data, the gap is not substantial. Moreover, compared to LLaMA-3.1-8B-instruct's initial performance, training on its self-synthesized data demonstrates improvements in tool-calling capabilities.

### Train with other base model with Table §tab:other_model_bfcl and §tab:other_model_api_bank display the results of two other base models fine-tuned with data generated with . Results demonstrate that *ToolFlow* enhances tool-calling performance across multiple LLMs, validating the generalizability of our synthetic data approach.

### Prompts and Demonstrations

See Table §tab:plan_prompt to §tab:tool_prompt for details.

**Table:** *Prompt for plan generation*

_(table content)_

**Table:** *Prompt for GPT-4 Data Evaluation*

_(table content)_

**Table:** *Prompt for User agent*

_(table content)_

**Table:** *Prompt for Assistant agent*

_(table content)_

**Table:** *Prompt for Tool agent*

_(table content)_

**Figure:** *Example tool in JSON format.* () _(image: figure)_

% — END chapters/appendix —