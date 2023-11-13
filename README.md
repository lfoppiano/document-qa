---
title: Scientific Document Insights Q/A
emoji: 📝
colorFrom: yellow
colorTo: pink
sdk: streamlit
sdk_version: 1.27.2
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# DocumentIQA: Scientific Document Insights Q/A

**Work in progress** :construction_worker: 

## Introduction

Question/Answering on scientific documents using LLMs (OpenAI, Mistral, ~~LLama2,~~ etc..).
This application is the frontend for testing the RAG (Retrieval Augmented Generation) on scientific documents, that we are developing at NIMS.
Differently to most of the project, we focus on scientific articles. We target only the full-text using [Grobid](https://github.com/kermitt2/grobid) that provide and cleaner results than the raw PDF2Text converter (which is comparable with most of other solutions).

**NER in LLM response**: The responses from the LLMs are post-processed to extract <span stype="color:yellow">physical quantities, measurements</span> (with [grobid-quantities](https://github.com/kermitt2/grobid-quantities)) and <span stype="color:blue">materials</span> mentions (with [grobid-superconductors](https://github.com/lfoppiano/grobid-superconductors)).

**Demos**: 
 - (on HuggingFace spaces): https://lfoppiano-document-qa.hf.space/
 - (on the Streamlit cloud): https://document-insights.streamlit.app/

## Getting started

- Select the model+embedding combination you want ot use 
- Enter your API Key ([Open AI](https://platform.openai.com/account/api-keys) or [Huggingface](https://huggingface.co/docs/hub/security-tokens)). 
- Upload a scientific article as PDF document. You will see a spinner or loading indicator while the processing is in progress. 
- Once the spinner stops, you can proceed to ask your questions

 ![screenshot2.png](docs%2Fimages%2Fscreenshot2.png)

## Documentation

### Context size
Allow to change the number of blocks from the original document that are considered for responding. 
The default size of each block is 250 tokens (which can be changed before uploading the first document). 
With default settings, each question uses around 1000 tokens.

**NOTE**: if the chat answers something like "the information is not provided in the given context", **changing the context size will likely help**. 

### Chunks size
When uploaded, each document is split into blocks of a determined size (250 tokens by default). 
This setting allow users to modify the size of such blocks. 
Smaller blocks will result in smaller context, yielding more precise sections of the document. 
Larger blocks will result in larger context less constrained around the question.

### Query mode
Indicates whether sending a question to the LLM (Language Model) or to the vector storage. 
 - LLM (default) enables question/answering related to the document content.
 - Embeddings: the response will consist of the raw text from the document related to the question (based on the embeddings). This mode helps to test why sometimes the answers are not satisfying or incomplete.

### NER (Named Entities Recognition)

This feature is specifically crafted for people working with scientific documents in materials science. 
It enables to run NER on the response from the LLM, to identify materials mentions and properties (quantities, masurements).
This feature leverages both [grobid-quantities](https://github.com/kermitt2/grobid-quanities) and [grobid-superconductors](https://github.com/lfoppiano/grobid-superconductors) external services. 


## Development notes

To release a new version: 

- `bump-my-version bump patch` 
- `git push --tags`

To use docker: 

- docker run `lfoppiano/document-insights-qa:latest`

To install the library with Pypi: 

- `pip install document-qa-engine` 


## Acknolwedgement 

This project is developed at the [National Institute for Materials Science](https://www.nims.go.jp) (NIMS) in Japan in collaboration with the [Lambard-ML-Team](https://github.com/Lambard-ML-Team).



