#### Keywords:
- Large Language Models (LLM): 
	- [Transformer](https://arxiv.org/abs/1706.03762) architecture-based:
		- Decoder-only Transformers, e.g. [GPT](https://paperswithcode.com/paper/improving-language-understanding-by) and its extensions
		- Encoder-only Transformers, e.g., [BERT](https://arxiv.org/abs/1810.04805) and its extensions
		- Complete Transformers with both encoder and decoders, e.g., [BART](https://arxiv.org/abs/1910.13461), [T-5](https://arxiv.org/abs/1910.10683)
	- [Mamba](https://arxiv.org/abs/2312.00752)
	
Adopting next token prediction (so coined causal language models) as a simple learning task, base models of Decoder-only LLMs are usually pretrained on huge amount of  training data for months, consuming hundreds and thousands of GPUs or XPUs, which is not really practical for small groups.   Continual training, or fine tuning with pretrained LLMs hence play a role in adapting LLMs for specific interests. 
- Fine tuning:
	- In the narrowest sense, it refers to continual training LLMs on specific corpora of interest, so that those previously less underscored specific co-occurrence patterns of tokens (word/subword) can be captured by the model parameters. The same next token prediction can be the basic fine-tuning task. If so, the training result is, like the original pretrained model, also a base model. 
	- A pretrained or fine tuned __base model__ can be good at text generation, but still can not be well used to conduct instruct following tasks, e.g., instruct it to summarize, answer questions or chat. To enable the instruct following ability, there is a need for _instruct fine tuning_.    
- Instruct [fine] tuning:
	- In a more general sense, for tuning LLM bases to conduct different tasks according to the task-specific meta information described in the instruction. There is also motivation to generalise such ability to unseen instructions (e.g., tasks) after tuning. Example projects include:
		- [Super-NaturalInstructions](https://arxiv.org/pdf/2204.07705.pdf)
		- [Self-instruct GPT3](https://arxiv.org/abs/2212.10560)
		- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), 
	- In a more specific sense, instruct tuning may be used to only adapt the base LLMs to a single NLP task, e.g. question answering.
	- Generated tokens for answers are compared with ground-truth for measuring losses and updating model parameters.   

Note, both plain fine tuning and instruct tuning may result in a LLM forgetting its previous captured token co-occurrence patterns (which encodes world knowledge). Plain fine-tuning break the pa 
 
- Retrieval-augmented generation (RAG):
	- 


<!--stackedit_data:
eyJoaXN0b3J5IjpbNzY2NzUzMjY2LDcyNzQ5ODA4MywtNTM3MD
Y1NTc3LC00MDIxMDgxNywtMTYyNzQyODcsMTg3ODAxNTc1Niwt
MjA4ODc0NjYxMl19
-->