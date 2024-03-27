#### Key concepts:
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


>- The strategy of the above practices is to inject all kinds of knowledge into the parametric models through training or fine tuning, which is very expensive (require substantial resources to be kept up-to-date, yet they struggle to capture long-tail knowledge), less flexible and manageable. 
>- Meanwhile, the different knowledge implicitly encoded in the parametric models can be roughly divided into: 1) Factual knowledge, their relations and common sense, and 2) different domains; 3) meta knowledge of linguistics or conventional usage patterns for generating human-understandable discourses; 4) meta  knowledge of behaviors such as instruct-following, prompting understanding etc.
>- It can be noted that pretraining LLM bases is mainly for gaining knowledge in 1) & 3); domain-specific finetuning is mainly for 2); while instruct tuning is aimed at 4); Reinforcement learning from human feedback (RLHF) is for improving 3);
>- __Questions__:
	>-- Is it necessary to encode all the knowledge in parametric form?
	>-- If not, can we decouple some knowledge component from the parameters

- Retrieval-augmented generation (RAG):
	- 


<!--stackedit_data:
eyJoaXN0b3J5IjpbODQ5NTg4OTY3LC01NDcwMzkwODEsLTYxOD
kyOTk0MSwyMDc3NzUyMDIxLDU1NjAxNTUzMywtNzUwNTE0OTQ1
LDcyNzQ5ODA4MywtNTM3MDY1NTc3LC00MDIxMDgxNywtMTYyNz
QyODcsMTg3ODAxNTc1NiwtMjA4ODc0NjYxMl19
-->