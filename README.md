#### Keywords:
- Large Language Models (LLM): 
	- [Transformer](https://arxiv.org/abs/1706.03762) architecture-based:
		- Decoder-only Transformers, e.g. [GPT](https://paperswithcode.com/paper/improving-language-understanding-by) and its extensions
		- Encoder-only Transformers, e.g., [BERT](https://arxiv.org/abs/1810.04805) and its extensions
		- Complete Transformers with both encoder and decoders, e.g., [BART](https://arxiv.org/abs/1910.13461), [T-5](https://arxiv.org/abs/1910.10683)
	- [Mamba](https://arxiv.org/abs/2312.00752)
	
Adopting next token prediction (so coined causal language models) as a simple learning task, base models of Decoder-only LLMs are usually pretrained on huge amount of  training data for months, consuming hundreds and thousands of GPUs or XPUs, which is not really practical for small groups.   Continual training, or fine tuning with pretrained LLMs hence play a role in adapting LLMs for many downstream tasks. 
- Fine tuning:
	- In the narrowest sense, it refers to 
- Instruct fine tuning
- Retrieval-augmented generation (RAG):
	- 


<!--stackedit_data:
eyJoaXN0b3J5IjpbNzQ4NDQxNTM0LDE4NzgwMTU3NTYsLTIwOD
g3NDY2MTJdfQ==
-->