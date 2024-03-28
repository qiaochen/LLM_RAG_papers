#### Background, key concepts and comments:
- Large Language Models (LLM): 
	- [Transformer](https://arxiv.org/abs/1706.03762) architecture-based:
		- Decoder-only Transformers, e.g. [GPT](https://paperswithcode.com/paper/improving-language-understanding-by) and its extensions
		- Encoder-only Transformers, e.g., [BERT](https://arxiv.org/abs/1810.04805) and its extensions
		- Complete Transformers with both encoder and decoders, e.g., [BART](https://arxiv.org/abs/1910.13461), [T-5](https://arxiv.org/abs/1910.10683)
	- [Mamba](https://arxiv.org/abs/2312.00752)
	
Adopting next token prediction (so coined causal language models) as a simple learning task, base models of Decoder-only LLMs are usually pretrained on huge amount of  training data, where linguistic patterns and word/subword co-occurrence patterns can be learned. With the property of Transformer architecture, the patterns can be complex and dynamically contextualised, through which world knowledge is encoded. The training process can last for months or even more, depending on the data volume, consuming hundreds and thousands of GPUs or XPUs, which is not really practical for small groups.  Continual training, or fine tuning based on pretrained LLMs hence play a role in adapting LLMs for specific needs. 
- Fine tuning:
	- In the narrowest sense, it refers to continual training LLMs on specific corpora of interest, so that those previously less exposed co-occurrence patterns of tokens (word/subword) can be captured by adapting the model parameters. The same next token prediction can be the basic fine-tuning task. If so, the training result is, like the original pretrained model, also a base model. 
	- A pretrained or fine tuned __base model__ can be good at text generation, but still it can not well conduct instruct following tasks, e.g., instruct it to summarize, answer questions or chat. To enable LLMs in instruct following, there is a need for _instruct fine tuning_.    
- Instruct [fine] tuning:
	- In a more general sense, is to continue training LLM bases to obtain the ability of conducting different tasks described in task-specific meta information in forms of instructions. There is also motivation to generalise such ability to unseen instructions (i.e., tasks) after tuning. Example projects include:
		- [Super-NaturalInstructions](https://arxiv.org/pdf/2204.07705.pdf)
		- [Self-instruct GPT3](https://arxiv.org/abs/2212.10560)
		- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), 
	- In a more specific sense, instruct tuning may be used to only adapt the base LLMs to a single NLP task, e.g. question answering.
	- To train, generated tokens for answers or responses are compared with ground-truth for measuring losses and updating model parameters.   

Note, both plain fine tuning and instruct tuning may result in a LLM forgetting its previous captured token co-occurrence patterns (world knowledge). Plain fine-tuning may capture new patterns at the cost of losing old patterns, while instruct tuning may sacrifice world knowledge for obtaining instruct-following related token generation patterns.

> Comments
>- The strategy of the above practices is to inject all kinds of knowledge into the parametric models through training or fine tuning, which is very expensive (require substantial resources to be kept up-to-date, yet they struggle to capture long-tail knowledge), less flexible and manageable. 
>- Meanwhile, the different knowledge implicitly encoded in the parametric models can be roughly divided into: 
	>    1) Factual knowledge, their relations and common sense, and 
	>    2) different domains; 
	>    3) meta knowledge of linguistics or conventional usage patterns for generating human-understandable discourses; 
	>    4) meta knowledge of behaviors such as instruct-following, prompt understanding etc.
>- It can be noted that pretraining LLM bases is mainly for gaining knowledge in 1) & 3); domain-specific finetuning is mainly for 2); while instruct tuning is aimed at 4); Reinforcement learning from human feedback (RLHF) is for improving 3);
>- __Questions__:
	>     - Is it necessary to encode all the knowledge in parametric form?
	>     - If not, can we decouple some knowledge component from the parameters? 
	>The most suitable candidate might be 2), especially those analogous to the knowledge we humans can query on the fly in dictionaries/encyclopedia.  As for 3) & 4), they are more like meta knowledge analogous to the language and cognitive skills that are more innate to a human, which are also more frequently triggered, and hence better contained in the model parameters. Finally, in corporation with 3) and 4), a certain degree of common sense knowledge should be learned to support acceptable  behaviors of LLMs, thus knowledge in 1) is better contained in the model parameters .

- [Retrieval-augmented generation (RAG)](https://arxiv.org/abs/2005.11401):
	![](https://media.licdn.com/dms/image/D4D12AQHY76w85U8W5g/article-cover_image-shrink_720_1280/0/1695787886133?e=1717027200&v=beta&t=IyS5v27mnHUC1C9DWDq7ddJ-hFzndAxOAGcyNiuxHG8)
	- [Recent survey](https://arxiv.org/abs/2312.10997) 
	- a solution to decoupling knowledge from model parameters, in the case of domain specific QA:
		- Fixed Model parameters: Knowledge 1), 3), 4)
		- External knowledge base (on the fly): Knowledge 2)
	-  Important components in RAG:
		- LLM
		- Retriever 
			- Dense embedding-based ([Huggingface leaderboard](https://huggingface.co/spaces/mteb/leaderboard)):
				- Query encoder
				- Document encoder
				- Retrieval boils down to find the most similar document embeddings given a query embedding vector, [a pubmed fine-tuned retriever](https://github.com/ncbi/MedCPT)
			- Sparse embedding-based:
				- In contrast to dense embeddings, sparse embeddings normally work in the input token/subtoken level, where query and documents are represented in bags of tokens, with optionally, the tokens weighted by output of some language models. 
				- A special case is bag-of-token representation + BM25 (similarity computation algorithm), which based on token overlap.
		- Reranker
			- Different from retriever, modern rerankers usually take in both query and document as input to rerank the documents, thus more computationally expensive. Normally, they are applied to top-K candidate documents that are returned by a retriever, [example reranker](https://huggingface.co/BAAI/bge-reranker-base).   
		- Indexed resources
	- In a vanila RAG system, LLM is fixed, so the upper bound of the response quality is determined by the retrieval system and the external resources. 
		- At a small resource scale (e.g., < 10k documents), dense vector approach with vector databases (e.g., [qdrant](https://qdrant.tech/articles/sparse-vectors/), [faiss](https://faiss.ai/index.html)) can be a good solution. However, the vectors and indexes (e.g., by [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320)) are loaded to RAM, which may not be applicable to a larger scale.
		- Traditional  retrieval system such as [Lucene](https://lucene.apache.org/) based [Solr](https://solr.apache.org/) and  [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) can be a rescue in such scenarios. They are based on classic techniques such as BM25 and inverted index.

> Comments
> With a competent LLM and good retrieval system accompanied by techniques like [Chain of Thoughts Prompting](https://arxiv.org/abs/2201.11903), RAG has the potential to perform comparably well or even better than fine-tuned LLMs in a new domain. It is also much more flexible to manipulate external knowledge for updating or filtering information. However, the potentially incurred new bottlenecks now are:
>      1) A "good" retriever and/or a reranker:
>      2) LLM's ability of 
>      Question:
>      Would the ability of something best included in the "meta knowledge of behaviors such as instruct-following, prompt understanding etc" ?
>      If so, then it might be helpful to explicitly fine-tune models to better capture such knowledge (There is increasing research interest in this direction). 
>      

A technique gaining increasing attention:




<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ1NzExOTE4MSwtMTczNjMwNTE4LDEyNT
MxNDIwMzcsLTYxMTY0NDc2NywyMDI0MDk5NzA1LC00NTY1ODU1
OTEsMTAyMjAwNzcyOCwtNjQ4MzYyMjI1LC0xMDY1ODc0OTgzLC
01NDcwMzkwODEsLTYxODkyOTk0MSwyMDc3NzUyMDIxLDU1NjAx
NTUzMywtNzUwNTE0OTQ1LDcyNzQ5ODA4MywtNTM3MDY1NTc3LC
00MDIxMDgxNywtMTYyNzQyODcsMTg3ODAxNTc1NiwtMjA4ODc0
NjYxMl19
-->