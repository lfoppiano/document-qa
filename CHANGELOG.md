# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.1] - 2024-08-23

### Added
+ Scroll to the first relevant context passage, if the most relevant context passage is at the end, it will scroll to the end of the document
+ Added Mistral NEMO as default model 

### Changed
+ Rearranged the interface to get more space
+ Updated libraries to the latest versions

### Fixed
+ Fixed the chat messages sequence that were buggy
+ Updated the PDF viewer to the latest version

## [0.4.0] - 2024-06-24

### Added 
+ Add selection of embedding functions 
+ Add selection of text from the pdf viewer (provided by https://github.com/lfoppiano/streamlit-pdf-viewer) 
+ Added an experimental feature for calculating the coefficient that relate the question and the embedding database 
+ Added the data availability statement in the searchable text

### Changed
+ Removed obsolete and non-working models zephyr and mistral v0.1
+ The underlying library was refactored to make it easier to maintain 
+ Removed the native PDF viewer
+ Updated langchain and streamlit to the latest versions
+ Removed conversational memory which was causing more problems than bringing benefits
+ Rearranged the interface to get more space

### Fixed
+ Updated and removed models that were not working 
+ Fixed problems with langchain and other libraries

## [0.3.4] - 2023-12-26

### Added

+ Add gpt4 and gpt4-turbo

### Changed

+ improved UI: replace combo boxes with dropdown box

### Fixed

+ Fixed dependencies when installing as library

## [0.3.3] - 2023-12-14

### Added

+ Add experimental PDF rendering in the page

### Fixed

+ Fix GrobidProcessors API implementation

## [0.3.2] - 2023-12-01

### Fixed

+ Remove memory when using Zephyr-7b-beta, that easily hallucinate

## [0.3.1] - 2023-11-22

### Added

+ Include biblio in embeddings by @lfoppiano in #21

### Fixed

+ Fix conversational memory by @lfoppiano in #20

## [0.3.0] - 2023-11-18

### Added

+ add zephyr-7b by @lfoppiano in #15
+ add conversational memory in #18

## [0.2.1] - 2023-11-01

### Fixed

+ fix env variables by @lfoppiano in #9

## [0.2.0] – 2023-10-31

### Added

+ Selection of chunk size on which embeddings are created upon
+ Mistral model to be used freely via the Huggingface free API

### Changed

+ Improved documentation, adding privacy statement
+ Moved settings on the sidebar
+ Disable NER extraction by default, and allow user to activate it
+ Read API KEY from the environment variables and if present, avoid asking the user
+ Avoid changing model after update

## [0.1.3] – 2023-10-30

### Fixed

+ ChromaDb accumulating information even when new papers were uploaded

## [0.1.2] – 2023-10-26

### Fixed

+ docker build

## [0.1.1] – 2023-10-26

### Fixed

+ Github action build
+ dependencies of langchain and chromadb

## [0.1.0] – 2023-10-26

### Added

+ pypi package
+ docker package release

## [0.0.1] – 2023-10-26

### Added

+ Kick off application
+ Support for GPT-3.5
+ Support for Mistral + SentenceTransformer
+ Streamlit application
+ Docker image
+ pypi package

<!-- markdownlint-disable-file MD024 MD033 -->
