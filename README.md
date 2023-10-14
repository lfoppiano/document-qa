# DocumentIQA: Scientific Document Insight QA

## Introduction

Question/Answering on scientific documents using LLMs (OpenAI, Mistral, LLama2).
This application is the frontend for testing the RAG (Retrieval Augmented Generation) on scientific documents, that we are developing at NIMS.
Differently to most of the project, we focus on scientific articles and we are using [Grobid](https://github.com/kermitt2/grobid) for text extraction instead of the raw PDF2Text converter allow to extract only full-text.

**Work in progress**

- Select the model+embedding combination you want ot use. 
- Enter your API Key (Open AI or Huggingface). 
- Upload a scientific article as PDF document. You will see a spinner or loading indicator while the processing is in progress. 
- Once the spinner stops, you can proceed to ask your questions

 ![screenshot1.png](docs%2Fimages%2Fscreenshot1.png)

### Query mode (LLm vs Embeddings)
By default, the mode is set to LLM (Language Model) which enables question/answering. You can directly ask questions related to the document content, and the system will answer the question using content from the document.
If you switch the mode to "Embedding," the system will return specific chunks from the document that are semantically related to your query. This mode helps to test why sometimes the answers are not satisfying or incomplete.

## Demo
The demo is deployed with streamlit and, depending on the model used, requires either OpenAI or HuggingFace **API KEYs**.

https://document-insights.streamlit.app/


### Acknolwedgement 

This project is developed at the [National Institute for Materials Science](https://www.nims.go.jp) (NIMS) in Japan. 



