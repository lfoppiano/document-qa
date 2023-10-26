---
title: DocumentIQA
emoji: 🚀
colorFrom: yellow
colorTo: pink
sdk: streamlit
sdk_version: 1.27.2
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# DocumentIQA: Scientific Document Insight QA

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

- Select the model+embedding combination you want ot use ~~(for LLama2 you must acknowledge their licence both on meta.com and on huggingface. See [here](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf))~~(Llama2 was removed due to API limitations). 
- Enter your API Key ([Open AI](https://platform.openai.com/account/api-keys) or [Huggingface](https://huggingface.co/docs/hub/security-tokens)). 
- Upload a scientific article as PDF document. You will see a spinner or loading indicator while the processing is in progress. 
- Once the spinner stops, you can proceed to ask your questions

 ![screenshot2.png](docs%2Fimages%2Fscreenshot2.png)

### Options
#### Context size
Allow to change the number of embedding chunks that are considered for responding. The text chunk are around 250 tokens, which uses around 1000 tokens for each question.

#### Query mode
By default, the mode is set to LLM (Language Model) which enables question/answering. You can directly ask questions related to the document content, and the system will answer the question using content from the document.
If you switch the mode to "Embedding," the system will return specific chunks from the document that are semantically related to your query. This mode helps to test why sometimes the answers are not satisfying or incomplete.

## Development notes

To release a new version: 

- `bump-my-version bump patch` 
- `git push --tags

To use docker: 

- docker run `lfoppiano/document-insights-qa:latest`

To install the library with Pypi: 

- `pip install document-qa-engine` 



## Acknolwedgement 

This project is developed at the [National Institute for Materials Science](https://www.nims.go.jp) (NIMS) in Japan in collaboration with the [Lambard-ML-Team](https://github.com/Lambard-ML-Team). 



