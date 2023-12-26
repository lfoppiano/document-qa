# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.5] - 2023-12-26

### Fixed

+ Fixed dependencies when installing as library

## [0.3.4] - 2023-12-16

### Added

+ Add gpt4 and gpt4-turbo

### Changed

+ improved UI: replace combo boxes with dropdown box

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
