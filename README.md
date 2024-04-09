### I. Background, key concepts and comments:
- Large Language Models (LLM): 
	- [Transformer](https://arxiv.org/abs/1706.03762) architecture-based:
		- Decoder-only Transformers, e.g. [GPT](https://paperswithcode.com/paper/improving-language-understanding-by) and its extensions
		- Encoder-only Transformers, e.g., [BERT](https://arxiv.org/abs/1810.04805) and its extensions
		- Complete Transformers with both encoder and decoders, e.g., [BART](https://arxiv.org/abs/1910.13461), [T-5](https://arxiv.org/abs/1910.10683)
	- [Mamba](https://arxiv.org/abs/2312.00752)
	
Adopting next token prediction (so coined causal language models) as a simple learning task, base models of Decoder-only LLMs are usually pretrained on huge amount of  training data, where linguistic patterns and word/subword co-occurrence patterns can be learned. With the property of Transformer architecture, the patterns can be complex and dynamically contextualised, through which world knowledge is encoded. The training process can last for months or even more, depending on the data volume, consuming hundreds and thousands of GPUs or XPUs, which is not really practical for small groups.  Continual training, or fine tuning based on pretrained LLMs hence play a role in adapting LLMs for specific needs. 
- Fine tuning (unsupervised/self-supervised continual training):
	- In the narrowest sense, it refers to continual training LLMs on specific corpora of interest, so that those previously less exposed co-occurrence patterns of tokens (word/subword) can be captured by adapting the model parameters. The same unsupervised next token prediction can be the basic fine-tuning task. If so, the training result is, like the original pretrained model, also a base model. 
	- A pretrained or fine tuned __base model__ can be good at text generation, but still it can not well conduct instruct following tasks, e.g., instruct it to summarize, answer questions or chat. To enable LLMs in instruct following, there is a need for _instruct fine tuning_.    
- Instruct [fine] tuning (supervised):
	- In a more general sense, is to continue training LLM bases to obtain the ability of conducting different tasks described in task-specific meta information in forms of instructions. There is also motivation to generalise such ability to unseen instructions (i.e., tasks) after tuning. Example projects include:
		- [Super-NaturalInstructions](https://arxiv.org/pdf/2204.07705.pdf)
		- [Self-instruct GPT3](https://arxiv.org/abs/2212.10560)
		- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), 
	- In a more specific sense, instruct tuning may be used to only adapt the base LLMs to a single NLP task, e.g. question answering.
	- To train, generated tokens for answers or responses are compared with ground-truth for measuring losses and updating model parameters.   

- Alignment:
	- This step is normally conducted after fine-tunning, the goal is to align LLM's generations to human preferences. In practice, reinforcement learning from human feedback (RLHF) is the most popular general framework for inducing such preferences for LLMs (yes, here comes another learning procedure). The human preference scores for discourse are not direct labels predicted from LLMs, but rather, they can be viewed as long-term rewards, and hence naturally fit into the scope of a reinforcement learning framework. 
	- Practically, gradient-based reinforcement learning framework [Proximal Policy Optimization, PPO](https://arxiv.org/pdf/1707.06347.pdf) and more recent method [Direct Preference Optimization, DPO](https://arxiv.org/abs/2305.18290) are popular approaches.
	- Normally, this step requires a fixed trained reward model that can score LLM's outputs based on human preferences, LLM's parameters can then be adjusted with a chosen alignment framework to better match human preferences.  

Note, both plain fine tuning and instruct tuning may result in a LLM forgetting its previous captured token co-occurrence patterns (world knowledge). Plain fine-tuning may capture new patterns at the cost of losing old patterns, while instruct tuning may sacrifice world knowledge for obtaining instruct-following related token generation patterns.

> Comments
>- The strategy of the above practices is to inject all kinds of knowledge into the parametric models through training or fine tuning, which is very expensive (require substantial resources to be kept up-to-date, yet they struggle to capture long-tail knowledge), less flexible and manageable. 
>- Meanwhile, the different knowledge implicitly encoded in the parametric models can be roughly divided into: 
	>    1) Factual knowledge, their relations and common sense, and 
	>    2) those in different domains; 
	>    3) meta knowledge of linguistics or conventional usage patterns for generating human-understandable discourses; 
	>    4) meta knowledge of behaviors such as instruct-following, prompt understanding etc.
>- It can be noted that pretraining LLM bases is mainly for gaining knowledge in 1) & 3); domain-specific finetuning is mainly for 2); while instruct tuning is aiming at 4); Reinforcement learning from human feedback (RLHF) is for improving 3). In finetuning practice, different combinations of   2), 3) and 4) mayh happen.
>- __Questions__:
	>     - Is it necessary to encode all the knowledge in parametric form?
	>     - If not, can we decouple some knowledge component from the parameters? 
	>The most suitable candidate might be 2), especially those analogous to the knowledge we humans can query on the fly in dictionaries/encyclopedia.  As for 3) & 4), they are more like meta knowledge analogous to the language and cognitive skills that are more innate to a human, which are also more frequently triggered, and hence better contained in the model parameters. Finally, in corporation with 3) and 4), a certain degree of common sense knowledge should be learned to support acceptable  behaviors of LLMs, thus knowledge in 1) is better contained in the model parameters (fortunately, those pretrained LLMs have largely helped us in this respect) .

- [Retrieval-augmented generation (RAG)](https://arxiv.org/abs/2005.11401):
	![](https://media.licdn.com/dms/image/D4D12AQHY76w85U8W5g/article-cover_image-shrink_720_1280/0/1695787886133?e=1717027200&v=beta&t=IyS5v27mnHUC1C9DWDq7ddJ-hFzndAxOAGcyNiuxHG8)
	- [Recent survey](https://arxiv.org/abs/2312.10997) 
	- a solution to decoupling knowledge from model parameters, in the case of domain specific QA, knowledge under a RAG framework can be divided into two parts:
		- Fixed Model parameters: Knowledge 1), 3), 4)
		- External knowledge base (on the fly): Knowledge 2)
	-  Important components in a RAG framework:
		- LLM
		- Retriever (useful packages: [pyserini](https://github.com/castorini/pyserini))
			- Dense embedding-based ([Huggingface leaderboard](https://huggingface.co/spaces/mteb/leaderboard)):
				- Query encoder
				- Document encoder
				- Retrieval boils down to find the most similar document embeddings given a query embedding vector, [a pubmed fine-tuned retriever](https://github.com/ncbi/MedCPT)
			- Sparse embedding-based:
				- In contrast to dense embeddings, sparse embeddings normally work in the input token/sub-token level, where query and documents are represented in bags of tokens, with, optionally, the tokens weighted by, e.g., generative language models. Based on token overlapping, this way may be more conservative (higher precision & lower recall) than dense-vector based solution. 
				- A classical case is bag-of-token representation + BM25 (similarity computation algorithm), which is the backbone of traditional search engines.
		- Reranker
			- Different from retriever, modern rerankers usually take in both query and document as input to rerank the documents, thus more computationally expensive. Normally, they are applied to top-K candidate documents that are returned by a retriever, [example reranker](https://huggingface.co/BAAI/bge-reranker-base).   
		- Indexed resources
	- In a vanila RAG system, LLM is fixed, so the upper bound of the response quality is determined by the retrieval system and the external resources. 
		- At a small resource scale (e.g., < 10k documents), dense vector approach with vector databases (e.g., [qdrant](https://qdrant.tech/articles/sparse-vectors/), [faiss](https://faiss.ai/index.html)) can be a good solution. However, the vectors and indexes (e.g., by [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320)) are loaded to RAM, which may not be applicable to a larger scale.
		- Traditional  retrieval system such as [Lucene](https://lucene.apache.org/) based [Solr](https://solr.apache.org/) and  [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) can be a rescue in such scenarios. They are based on classic techniques such as BM25 and inverted index.
		- Update: The vector base Faiss has index stored in disc rather than in RAM, in such case larger-scale RAG using dense vectors seems possible.

> Comments
> With a competent LLM and good retrieval system accompanied by techniques like [Chain of Thoughts Prompting](https://arxiv.org/abs/2201.11903), RAG has the potential to perform comparably well or even better than fine-tuned LLMs in a new domain. It is also much more flexible to manipulate external knowledge for updating or filtering information. However, the potentially incurred new bottlenecks now are:
>      1) The "goodness" of a retriever and/or a reranker for a new domain:
>      2) After the retrieval systems finish its job, the "goodness" of a LLM's ability of selecting pertinent documents, ignoring distracting documents, and making sound responses.
>      For 1), in case of dense vector approach, expanding dictionary & finetunning top-performing encoders in a new domain would be a standard way: while in the case of sparse vector approach, more domain specific data engineering is the approach, e.g., better token-based index and Bag-Of-Token representations.
>      For 2), this ability can be categorised into some meta knowledge belonging to instruction following. Therefore, it might be helpful to compile an instruct-tuning task for LLMs to better capture such knowledge (There is increasing research interest in this direction). 

### II. Relevant recent research:
#### a. RAG vs Finetuning
- Ovadia, O., Brief, M., Mishaeli, M., & Elisha, O. (2023). [Fine-tuning or retrieval? comparing knowledge injection in llms](https://arxiv.org/abs/2312.05934). _arXiv preprint arXiv:2312.05934_.
> This paper compared RAG and unsupervised fine-tuning in new domain adaptation, both using the same new domain corpus. For evaluation, they use _LM-Evaluation-Harness_ (Which I used for evaluating Bloom, Llama2 and Mixture LLMs, too), the benefits are quoted below. The major results shown in Table 1 demonstrate _"while unsupervised fine-tuning offers some improvement, RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge."_ Base+RAG without finetuning is comparable to Base+Finetune+RAG in the benchmark. 
> 
> "_LM-Evaluation-Harness is a robust benchmarking tool that currently serves as the industry standard for model evaluation and is the basis of the HuggingFace leaderboard3 . Leveraging this platform ensured a standardized evaluation framework and allowed consistent comparison across models, methods, and datasets. More importantly, by using the industry standard for evaluation, we could avoid any differences stemming from prompt engineering and formatting issues and replicate the reported baseline results for each model._" 

- Soudani, H., Kanoulas, E., & Hasibi, F. (2024). [Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge](https://arxiv.org/pdf/2403.01432.pdf). _arXiv preprint arXiv:2403.01432_.
> _This paper explores and evaluates the impact of RAG and FT (supervised) on customizing LLMs in handling low-frequency entities on question answering task._ Their base model is a T5 variation, FlanT5. Most solid conclusion from Table 2 is that RAG can greatly (10X on FlanT5-base)  boost the performance of LLM without any finetuning, while finetuning alone can only achieve <2 X performance gain on FlanT5-base. RAG + finetuning achieves slightly better performance on FlanT5-base than RAG alone, showing the major contributor is RAG. 

- Gupta, A., Shirgaonkar, A., Balaguer, A. D. L., Silva, B., Holstein, D., Li, D., ... & Benara, V. (2024). [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406). _arXiv preprint arXiv:2401.08406_.
> Not sure how GPT-4 was fine-tuned, but from Tables 18, 19 & 20, by comparing the RAG+GPT4 and Finetune+GPT4 options, it is evident that the RAG is comparable to Finetuning in accuracy, and much better in _succinctness_ and  _Percent of answers that were fully correct_.

The above literature demonstrate the promising role of RAG in domain adaptation. 

#### b. Finetuning for RAG 

- Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., & Gonzalez, J. E. (2024). [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/pdf/2403.10131.pdf). _arXiv preprint arXiv:2403.10131_.[code](https://github.com/ShishirPatil/gorilla/tree/main/raft)
> This work proposes to finetune LLMs for better RAG, an approach that could be applied to addressing bottleneck 2) commented above (i.e. LLMs ability of selecting pertinent and ignoring distracting references). The key idea is, during finetuning, the input additionally include context documents that are mixed with unrelated candidates. Thus, the LLM has to change its behaviors during training to make best use of the right document for answering. The finetuning strategy also has a mechanism that occasionally turns off feeding context references into the input, so as to make the parametric LLM also learn new domain knowledge in its parameters.
> They claim using _Chain-of-Thought_ prompting strategy greatly boosted model performance.  

- Lin, X. V., Chen, X., Chen, M., Shi, W., Lomeli, M., James, R., ... & Yih, S. (2024). [Ra-dit: Retrieval-augmented dual instruction tuning](https://openreview.net/pdf?id=22OTbutug9). ICLR 2024.
> Both dense retriever and LLM are fine-tuned to improve RAG performance. In a sense it is addressing the two bottlenecks mentioned above, i.e., 1) improve “goodness” of a retriever in new domain, and 2) improve a LLM’s ability of selecting pertinent documents, ignoring distracting documents.
> For finetuning LLM, given a (prompt, response) pair, after retrieving C documents, each of them is prepended to prompt, creating C new datapoints of $\{[c_{i,j};x], y  \}$. These constitute the augmented training samples and losses are accumulated from all the datapoints: $L(D_L) = -\sum_{i}\sum_{j}logp_{LM}(y_i|[c_{ij};x])$. This adapts LLM to better utilize background knowledge in generating responses with or without relevant references.
> For finetuning retriever (only query encoder), they try to match LLM's score and retriever's relevance score with a KL divergence: 
> $L(D_r)=E_{(x,y)\in D_r} KL(p_r(c|x)||p_{LSR}(c|x,y))$
> where $p_R(c|x) = \frac{exp(s(x,c))}{\sum_{c' \in C'} exp(s(x, c'))}$ 
> $p_{LSR}(c|x,y) = \frac{exp(p_{LM}(y|[c;x])/\tau)}{\sum_{c' \in C'} exp(p_{LM}(y|[c';x])/\tau)}$
> $C'$ is the set of retrieved documents.
> Performance shown in Table 2 indicates a great improvement of RA-DIT over  non-finetuned raw model. The advantages over non-finetuned state-of-the-art RAG model are also demonstrated, although not that great.
- Ye, X., Sun, R., Arik, S. Ö., & Pfister, T. (2023). [Effective large language model adaptation for improved grounding](https://arxiv.org/abs/2311.09533). _arXiv preprint arXiv:2311.09533_.
> This work fine-tunes LLMs to improve the quality of citation. A fine-tuning training prompt consists of query, response (multi-sentence) and citations (multiple per-sentence, if any). First, they prompt LLM-base with query and reference as input, to synthesise responses. These triple elements serve as ground-truth to fine tune LLM based on the prompt above, in a sense the instruction following for citation is learned as a supervised task.
> After fine-tuning, the inference __has multiple rounds of retrieval and citation__ (_a trick in many RAG papers, see Adaptive-RAG paper_), with the aim to cite for previously unsupported answer sentences, until the iteration budget is exhausted.
> From Table 2,  answer qualities across benchmarks are comparable between fine-tuned and base LLMs, while the recall and precision of retrieved citations are better.  The inference strategy further boosts citation quality on some benchmarks.

- Lee, J., et al. (2024), [Gecko: Versatile Text Embeddings Distilled from Large Language Models](https://arxiv.org/abs/2403.20327). 
> This work tries to improve the embeddings models by distilling knowledge from LLM. Although it is not directly targeting RAG, it is definitely relevant and worth studying. _On the Massive Text Embedding Benchmark (MTEB), Gecko with 256 embedding dimensions outperforms all existing entries with 768 embedding size. Gecko with 768 embedding dimensions achieves an average score of 66.31, competing with 7x larger models and 5x higher dimensional embeddings._

#### c. RAG strategy
- Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403.pdf). _arXiv preprint arXiv:2403.14403_.
>Tailoring RAG strategies for queries at different complexity levels is this work's focus. The authors drew our attention to the complexity of user queries. They are motivated to categorise queries into different complexity levels, because simple questions (e.g., single hop ) can be addressed using one-step retrieval,  while more complex (e.g., multi-hop) queries may need iteratively retrieving references in multiple steps to reach the answers. Many queries, however, lies between simple and high complexity, which motivates this work to design an adaptive strategy to conduct RAG, _ranging from iterative, to single, to even no retrieval approaches_ .
> The first step towards adaptivity is to automatically determine the complexity of a given query. The authors hence trained a classifier, based on synthesised query-label pairs using three types of strategy: no RAG, simple-step RAG, multi-step RAG.
> Main results in Table 2. indicate Adaptive-RAG greatly boosted perfromance in both single-step and multi-step benchmarks. (RAG's advantage over non-RAG is also visible) 
> Comment: _In a sense it is like preparing a fine-grained router that can redirect the query to different handlers. It might be some lower level sub-router in a more sophisticated dialogue management architecture_ 

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). [Self-rag: Learning to retrieve, generate, and critique through self-reflection.] (https://arxiv.org/abs/2310.11511)_arXiv preprint arXiv:2310.11511_.
> LLMs are not trained to exploit facts from provided passages. Authors train a LM to learn to reflect on its own generation processing given a input, by generating both task output and intermediate special tokens (reflection tokens: retrieval and critique). "Retrieval" indicates retrieving documents on demand (input prompt + last generations -> retrieval or not ), while critique refers to evaluation (isRelevant; is Fully, partially, or not supported; is useful 5-1) of its own output after generating task outputs with retrieved passages and the input.
> Self-RAG needs to train both the generator M and the critic C. The critic is trained on prompted GPT-4 data initialized with Llama 2-7B. 

- (2024) [RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation](https://arxiv.org/abs/2404.00610)
> This paper reiterated the benefits of query rewriting. Imaging a complex query that entails multiple hops of reasoning, breaking it into separate simple queries would potentially retrieve better references. This work tries to learn a model to refine query for RAG. Their result is very exciting: A 7B model with query refining ability outperformed Chat-GPT3.5 on three multi-hop inferencing QA benchmarks.
> RAG here again indicated better performance than supervised fine-tuning (SFT).
>  
#### d. Evaluation
- Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling large language models to generate text with citations.  _arXiv preprint arXiv:2305.14627_.
> Becoming a standard RAG evaluation task. See github for how evaluation is conducted. 
- Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). [Benchmarking retrieval-augmented generation for medicine.](https://arxiv.org/abs/2402.13178) _arXiv preprint arXiv:2402.13178_.
> A new benchmark is created from five medical QA datasets.
> A toolkit MedRag is introduced. They reported two observations 1) a _log-linear scaling relationship between model performance and the number of of retrieved snippets_ 2) _lost-in-the-middle phenomenon between model performance and the position of the ground-truth snippet_
> ![Main result](https://teddy-xionggz.github.io/benchmark-medical-rag/figs/result_llm.png) 

#### e. Combining Knowledge Graph and LLM by RAG
- Yang, R., Liu, H., Zeng, Q., Ke, Y. H., Li, W., Cheng, L., ... & Li, I. (2024). [KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques.](https://arxiv.org/abs/2403.05881) _arXiv preprint arXiv:2403.05881_.
> Pure RAG, no training.
> _KG-Rank, a framework that integrates a structured medical knowledge graph, into existing pre-trained LLMs to achieve more accurate medical question-answering (QA)_
> Implementation details: UMLS as medical KG. Medical NER Prompt to identify medical entities; 1-hop relation extraction from KG between entities; Embed query and knowledge triple into embeddings using UmlsBERT (Must be a BERT trained/fine-tuned on UMLS); ranking based on 1) embedding vector similarity; 2) similarity between LLM-answer-expanded query embedding and knowledge triple; 3) Maximal  Marginal Relevance (MMR) similarity; reranking using a medical cross-encoder model (MedCPT*, Cohere).
> They compared 7b models LLaMa2 and [Baize-healthcare](https://huggingface.co/project-baize/baize-healthcare-lora-7B)*
- Soman, K., Rose, P. W., Morris, J. H., Akbas, R. E., Smith, B., Peetoom, B., ... & Baranzini, S. E. (2023). [Biomedical knowledge graph-enhanced prompt generation for large language models.](https://arxiv.org/abs/2311.17330)  _arXiv preprint arXiv:2311.17330_. [KG-RAG](https://github.com/BaranziniLab/KG_RAG)
> Pure RAG, no training.
> KG used: Scalable Precision Medicine Open Knowledge Engine, (KG SPOKE)
> Steps: 1) entity recognition from user prompt, 2) biomedical concept extraction from KG (one/two hops), vector similarity based on embedding model (MiniLM & PubMedBert*). 3) prompt-aware context generation, conversion to language, prompt assembly, and 4) answer retrieval.
> Entity grounding (step 2) is implemented via vector similarity between extracted input entities and pre-embedded disease names (nodes in SPOKE) stored in 'Chroma' vector database. edges and nodes from one/two hops of neighbours are then retrieved. A further filtering step is applied to the triples to retain 25% of the KG pieces that > 0.5 cosine similarity.
> Performance boost is huge with the KG-RAG approach.
  -Delile, J., Mukherjee, S., Van Pamel, A., & Zhukov, L. (2024). [Graph-Based Retriever Captures the Long Tail of Biomedical Knowledge](). _arXiv preprint arXiv:2402.12352_.
#### f. Survey, Position articles
- Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). [Retrieval-augmented generation for large language models: A survey](https://arxiv.org/abs/2312.10997). _arXiv preprint arXiv:2312.10997_. 
> Simply a pertinent albeit somewhat boring survey.
- Asai, A., Zhong, Z., Chen, D., Koh, P. W., Zettlemoyer, L., Hajishirzi, H., & Yih, W. T. (2024). [Reliable, Adaptable, and Attributable Language Models with Retrieval](https://arxiv.org/pdf/2403.03187.pdf). _arXiv preprint arXiv:2403.03187_.
> Some directions for improving RAG-based LLM generation
- Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). [Unifying large language models and knowledge graphs: A roadmap](https://arxiv.org/pdf/2306.08302.pdf). _IEEE Transactions on Knowledge and Data Engineering_.
> How to integrate Knowledge Graphs (KG) and LLMs, or enhance one with the other.

#### g. Others, Miscellaneous
- Wang, B., Ping, W., Xu, P., McAfee, L., Liu, Z., Shoeybi, M., ... & Catanzaro, B. (2023). [Shall we pretrain autoregressive language models with retrieval? a comprehensive study](https://arxiv.org/abs/2304.06762). _arXiv preprint arXiv:2304.06762_.
> This work attempts to augment LLM with retrieved context during [continued] training time.

- Jiang, Z., Sun, Z., Shi, W., Rodriguez, P., Zhou, C., Neubig, G., ... & Iyer, S. (2024). [Instruction-tuned Language Models are Better Knowledge Learners](https://arxiv.org/pdf/2402.12847.pdf). _arXiv preprint arXiv:2402.12847_.

- Chevalier, A., Geng, J., Wettig, A., Chen, H., Mizera, S., Annala, T., ... & Chen, D. (2024). [Language Models as Science Tutors](https://arxiv.org/pdf/2402.11111.pdf). _arXiv preprint arXiv:2402.11111_.

- Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). [Gorilla: Large language model connected with massive apis](https://arxiv.org/abs/2305.15334). _arXiv preprint arXiv:2305.15334_. [code](https://github.com/ShishirPatil/gorilla)
> _a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls._

- Jin, Q., Yang, Y., Chen, Q., & Lu, Z. (2024). [Genegpt: Augmenting large language models with domain tools for improved access to biomedical information](https://arxiv.org/pdf/2304.09667.pdf). _Bioinformatics_, _40_(2), btae075.
> An Agent-like application that routes input queries to pubmed url tools, which allows search position of variant in the genome, meta information about snps, genes, etc.

##### Tuning LLM for Tables
Zhang, T., Yue, X., Li, Y., & Sun, H. (2023). [Tablellama: Towards open large generalist models for tables.](https://osu-nlp-group.github.io/TableLlama/)  _arXiv preprint arXiv:2311.09206_.


Li, P., He, Y., Yashar, D., Cui, W., Ge, S., Zhang, H., ... & Chaudhuri, S. (2023). [Table-gpt: Table-tuned gpt for diverse table tasks](https://arxiv.org/pdf/2310.09263.pdf). _arXiv preprint arXiv:2310.09263_.

Lu, W., Zhang, J., Zhang, J., & Chen, Y. (2024). [Large Language Model for Table Processing: A Survey](https://arxiv.org/pdf/2402.05121.pdf). _arXiv preprint arXiv:2402.05121_.

[Defog SQLCoder](https://github.com/defog-ai/sqlcoder)



<!--stackedit_data:
eyJoaXN0b3J5IjpbMzAyMjA1ODk4LC03NDE1MjExODQsLTE2MD
czNzM1MjgsLTExMDkzMzg1NywzMTk2ODk5NzksMTM3OTg2MTUw
OCwyODQ5NjA3ODYsMTQxMTU1NDEwNSwtMTAwODkxOTk1MiwtMT
c5MjU3NzQ2MiwyMDQwODYzODgxLDMzMTY1MDI1NywzMjgzMzk5
MCwtMTYwMDgyODI3NiwtNDEyMjYzNDE5LDgwMzE0OTEwNCwtMT
gwNDQwMTI0NSwxMjM2MjgxNjcyLC05OTc4ODYwMzUsMTY2NjA0
NzM5M119
-->