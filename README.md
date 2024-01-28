# Aditya

This repository aims to develop a chatbot application that can introduce me to others and was developed as part of my technical assessment for SAP. It is repurposed from the excellent `llm-search` package by Denis Lapchev (https://github.com/snexus/llm-search). It is designed primarily to act as a question-answering system (using Retrieval-Augmented Generation) and leverages my CV and personal bio as the knowledge base. 

## Existing features of LLM-Search

* Better document parsing, hybrid search, HyDE enabled search, re-ranking and the ability to customize embeddings.
   * General configurations can be viewed and modified in `src/llmsearch/config.yaml`.
     
* The package is also designed to work with custom Large Language Models (LLMs) – whether from OpenAI or installed locally.
   * LLM-specific configuration can be viewed and modified in `src/llmsearch/openai.yaml`.
   * To switch to local LLMs like Llama cpp or HuggingFace models, please refer to the steps outlined in the `llm-search`'s [Quickstart](https://github.com/snexus/llm-search/tree/main?tab=readme-ov-file#quickstart)

* Generates dense embeddings from documents and stores them in a vector database (ChromaDB) using Sentence-transformers' `intfloat/e5-large-v2` model.

* Generates sparse embeddings using SPLADE (https://github.com/naver/splade) to enable hybrid search (sparse + dense).

* Supports the "Retrieve and Re-rank" strategy for semantic search, see - https://www.sbert.net/examples/applications/retrieve_rerank/README.html.
    * Besides the originally `ms-marco-MiniLM` cross-encoder, more modern `bge-reranker` is supported.

* Supports HyDE (Hypothetical Document Embeddings) - https://arxiv.org/pdf/2212.10496.pdf

* Support for multi-querying, inspired by `RAG Fusion` - https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
    * When multi-querying is turned on, the original query will be replaced by 3 variants of the same query, allowing to bridge the gap in the terminology and "offer different angles or perspectives" according to the article.
 
## Components Developed for Chatbot

To repurpose `llm-search` into a chatbot, the following modifications were made

* Prompt engineering `GPT-3.5 Turbo` for the specific task.
    * **Prompt**:
      ```
      I want you to act as a person called Aditya. I will provide you with an individual looking to know about Aditya, and your task is to introduce Aditya and answer questions about Aditya. Always use first person to answer questions. Context information is provided below. Given only the context and not prior knowledge, provide concise answer to the question. If context does not provide enough details, answer it without hallucinating.
        
       ### Context:
       ---------------------
       {context}
       ---------------------
  
       ### Question: {question}
      ```
    * The prompt addresses the requirement of LLM to act as Aditya, use first-person when interacting and utilize the fetched context to answer the question.
    * In addition, a key requirement is that the chatbot should be able to answer questions for which knowledge is not present in the documents.
        * The last sentence in the prompt handles this.
        * Chain-of-Thought prompting can handle this better but has not been incorporated in the current implementation. 
* Revamping the UI using [Steamlit Chat](https://github.com/AI-Yash/st-chat) to suit a chat-style interface.


## Demo

<img width="1512" alt="image" src="https://github.com/adtygan/Aditya/assets/51450254/2ad6ec2c-ef7c-4cdf-9425-6f9e296c5461">


## Prerequisites

* Tested on Mac (M1 Pro), but should work on Linux and Windows as well (please follow respective installation procedures below which are based on `llm-search`)
* Python 3.8+
* OpenAI API key to interact with OpenAI models
    * Personal cost for developing and testing the OpenAI model was ≈ $0.2 at the time of writing this
    * Testing the chatbot would require a much lesser cost


## Manual virtualenv based installation (personally tested on Mac)

```bash
git clone https://github.com/adtygan/Aditya.git
cd Aditya

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

# Set variables for llama-cpp to compile with CUDA.

# Assuming Nvidia CUDA Toolkit is installed and pointing to `usr/local/cuda` on Ubuntu

source ./setvars.sh 

# Install newest stable torch for CUDA 11.x
pip3 install torch torchvision

# Install the package using Metal (MPS) for mac (required for llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" pip install .
```

## Automatic virtualenv based installation on Linux

```bash
git clone https://github.com/adtygan/Aditya.git
cd Aditya

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

./install_linux.sh
```

# Quickstart

## 1) Generate document embeddings

```bash
llmsearch index create -c src/llmsearch/config.yaml
```

Using the configuration YAML file in `src/llmsearch/config.yaml`, this command will scan the PDF files contained in document path (`src/llmsearch/docs`) and generate a dense embeddings database in the `src/llmsearch/embeddings` directory. Additionally, a local cache folder (`src/llmsearch/cache`) will be utilized to store embedding models, LLM models, and tokenizers.

The default vector database for dense is ChromaDB, and default embedding model is `intfloat/e5-large-v2`, which is known for its high performance. You can find more information about this and other embedding models at [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

In addition to dense embeddings, sparse embedding will be generated in `src/llmsearch/embeddings/splade` using SPLADE algorithm. Both dense and sparse embeddings will be used for context search.

## 2) Interact with the documents

First, add your OpenAI API key in `.env` file by replacing <<YOUR_API_KEY>>> with your key.

> 📌 Note
> 
> Please ensure OpenAI API key has been added to the `.env` file before proceeding

To interact with the documents using web app interface, run the below command:

```bash
llmsearch interact webapp -c src/llmsearch -m src/llmsearch/openai.yaml
```

Based on the configuration YAML files provided, the following actions will take place:

- Based on the model config, GPT-3.5 Turbo will be accessed and model-specific parameters mentioned in `model_kwargs` from the `llm->params` section will be passed to the model.
- The system will query the embeddings database using hybrid search algorithm using sparse and dense embeddings. It will provide the most relevant context from different documents, up to a maximum context size of 4096 characters (`max_char_size` in `semantic_search`).
- It will then use this context to answer the user query.
