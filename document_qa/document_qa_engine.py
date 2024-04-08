import copy
import os
from pathlib import Path
from typing import Union, Any, Optional, List, Dict, Tuple, ClassVar, Collection

import tiktoken
from langchain.chains import create_extraction_chain
from langchain.chains.question_answering import load_qa_chain, stuff_prompt, refine_prompts, map_reduce_prompt, \
    map_rerank_prompt
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma, DEFAULT_K
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from tqdm import tqdm

from document_qa.grobid_processors import GrobidProcessor


def _results_to_docs_scores_and_embeddings(results: Any) -> List[Tuple[Document, float, List[float]]]:
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2], result[3])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["embeddings"][0],
        )
    ]


class TextMerger:
    """
    This class tries to replicate the RecursiveTextSplitter from LangChain, to preserve and merge the
    coordinate information from the PDF document.
    """

    def __init__(self, model_name=None, encoding_name="gpt2"):
        if model_name is not None:
            self.enc = tiktoken.encoding_for_model(model_name)
        else:
            self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text, allowed_special=set(), disallowed_special="all"):
        return self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def merge_passages(self, passages, chunk_size, tolerance=0.2):
        new_passages = []
        new_coordinates = []
        current_texts = []
        current_coordinates = []
        for idx, passage in enumerate(passages):
            text = passage['text']
            coordinates = passage['coordinates']
            current_texts.append(text)
            current_coordinates.append(coordinates)

            accumulated_text = " ".join(current_texts)

            encoded_accumulated_text = self.encode(accumulated_text)

            if len(encoded_accumulated_text) > chunk_size + chunk_size * tolerance:
                if len(current_texts) > 1:
                    new_passages.append(current_texts[:-1])
                    new_coordinates.append(current_coordinates[:-1])
                    current_texts = [current_texts[-1]]
                    current_coordinates = [current_coordinates[-1]]
                else:
                    new_passages.append(current_texts)
                    new_coordinates.append(current_coordinates)
                    current_texts = []
                    current_coordinates = []

            elif chunk_size <= len(encoded_accumulated_text) < chunk_size + chunk_size * tolerance:
                new_passages.append(current_texts)
                new_coordinates.append(current_coordinates)
                current_texts = []
                current_coordinates = []

        if len(current_texts) > 0:
            new_passages.append(current_texts)
            new_coordinates.append(current_coordinates)

        new_passages_struct = []
        for i, passages in enumerate(new_passages):
            text = " ".join(passages)
            coordinates = ";".join(new_coordinates[i])

            new_passages_struct.append(
                {
                    "text": text,
                    "coordinates": coordinates,
                    "type": "aggregated chunks",
                    "section": "mixed",
                    "subSection": "mixed"
                }
            )

        return new_passages_struct


class BaseRetrieval:

    def __init__(
            self,
            persist_directory: Path,
            embedding_function
    ):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory


class AdvancedVectorStoreRetriever(VectorStoreRetriever):
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
        "similarity_with_embeddings"
    )

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, similarity in docs_and_similarities:
                if '__similarity' not in doc.metadata.keys():
                    doc.metadata['__similarity'] = similarity

            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_with_embeddings":
            docs_scores_and_embeddings = (
                self.vectorstore.advanced_similarity_search(
                    query, **self.search_kwargs
                )
            )

            for doc, score, embeddings in docs_scores_and_embeddings:
                if '__embeddings' not in doc.metadata.keys():
                    doc.metadata['__embeddings'] = embeddings
                if '__similarity' not in doc.metadata.keys():
                    doc.metadata['__similarity'] = score

            docs = [doc for doc, _, _ in docs_scores_and_embeddings]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


class AdvancedVectorStore(VectorStore):
    def as_retriever(self, **kwargs: Any) -> AdvancedVectorStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AdvancedVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)


class ChromaAdvancedRetrieval(Chroma, AdvancedVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
            self,
            query_texts: Optional[List[str]] = None,
            query_embeddings: Optional[List[List[float]]] = None,
            n_results: int = 4,
            where: Optional[Dict[str, str]] = None,
            where_document: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    def advanced_similarity_search(
            self,
            query: str,
            k: int = DEFAULT_K,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> [List[Document], float, List[float]]:
        docs_scores_and_embeddings = self.similarity_search_with_scores_and_embeddings(query, k, filter=filter)
        return docs_scores_and_embeddings

    def similarity_search_with_scores_and_embeddings(
            self,
            query: str,
            k: int = DEFAULT_K,
            filter: Optional[Dict[str, str]] = None,
            where_document: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float, List[float]]]:

        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=['metadatas', 'documents', 'embeddings', 'distances']
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=['metadatas', 'documents', 'embeddings', 'distances']
            )

        return _results_to_docs_scores_and_embeddings(results)


class FAISSAdvancedRetrieval(FAISS):
    pass


class NER_Retrival(VectorStore):
    """
    This class implement a retrieval based on NER models.
    This is an alternative retrieval to embeddings that relies on extracted entities.
    """
    pass


engines = {
    'chroma': ChromaAdvancedRetrieval,
    'faiss': FAISSAdvancedRetrieval,
    'ner': NER_Retrival
}


class DataStorage:
    embeddings_dict = {}
    embeddings_map_from_md5 = {}
    embeddings_map_to_md5 = {}

    def __init__(
            self,
            embedding_function,
            root_path: Path = None,
            engine=ChromaAdvancedRetrieval,
    ) -> None:
        self.root_path = root_path
        self.engine = engine
        self.embedding_function = embedding_function

        if root_path is not None:
            self.embeddings_root_path = root_path
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            else:
                self.load_embeddings(self.embeddings_root_path)

    def load_embeddings(self, embeddings_root_path: Union[str, Path]) -> None:
        """
        Load the vector storage assuming they are all persisted and stored in a single directory.
        The root path of the embeddings containing one data store for each document in each subdirectory
        """

        embeddings_directories = [f for f in os.scandir(embeddings_root_path) if f.is_dir()]

        if len(embeddings_directories) == 0:
            print("No available embeddings")
            return

        for embedding_document_dir in embeddings_directories:
            self.embeddings_dict[embedding_document_dir.name] = self.engine(
                persist_directory=embedding_document_dir.path,
                embedding_function=self.embedding_function
            )

            filename_list = list(Path(embedding_document_dir).glob('*.storage_filename'))
            if filename_list:
                filenam = filename_list[0].name.replace(".storage_filename", "")
                self.embeddings_map_from_md5[embedding_document_dir.name] = filenam
                self.embeddings_map_to_md5[filenam] = embedding_document_dir.name

        print("Embedding loaded: ", len(self.embeddings_dict.keys()))

    def get_loaded_embeddings_ids(self):
        return list(self.embeddings_dict.keys())

    def get_md5_from_filename(self, filename):
        return self.embeddings_map_to_md5[filename]

    def get_filename_from_md5(self, md5):
        return self.embeddings_map_from_md5[md5]

    def embed_document(self, doc_id, texts, metadatas):
        if doc_id not in self.embeddings_dict.keys():
            self.embeddings_dict[doc_id] = self.engine.from_texts(texts,
                                                                  embedding=self.embedding_function,
                                                                  metadatas=metadatas,
                                                                  collection_name=doc_id)
        else:
            # Workaround Chroma (?) breaking change
            self.embeddings_dict[doc_id].delete_collection()
            self.embeddings_dict[doc_id] = self.engine.from_texts(texts,
                                                                  embedding=self.embedding_function,
                                                                  metadatas=metadatas,
                                                                  collection_name=doc_id)

        self.embeddings_root_path = None


class DocumentQAEngine:
    llm = None
    qa_chain_type = None

    default_prompts = {
        'stuff': stuff_prompt,
        'refine': refine_prompts,
        "map_reduce": map_reduce_prompt,
        "map_rerank": map_rerank_prompt
    }

    def __init__(self,
                 llm,
                 data_storage: DataStorage,
                 qa_chain_type="stuff",
                 grobid_url=None,
                 memory=None
                 ):

        self.llm = llm
        self.memory = memory
        self.chain = load_qa_chain(llm, chain_type=qa_chain_type)
        self.text_merger = TextMerger()
        self.data_storage = data_storage

        if grobid_url:
            self.grobid_processor = GrobidProcessor(grobid_url)

    def query_document(
            self,
            query: str,
            doc_id,
            output_parser=None,
            context_size=4,
            extraction_schema=None,
            verbose=False
    ) -> (Any, str):
        # self.load_embeddings(self.embeddings_root_path)

        if verbose:
            print(query)

        response, coordinates = self._run_query(doc_id, query, context_size=context_size)
        response = response['output_text'] if 'output_text' in response else response

        if verbose:
            print(doc_id, "->", response)

        if output_parser:
            try:
                return self._parse_json(response, output_parser), response
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response, coordinates
        elif extraction_schema:
            try:
                chain = create_extraction_chain(extraction_schema, self.llm)
                parsed = chain.run(response)
                return parsed, response, coordinates
            except Exception as oe:
                print("Failing to parse the response", oe)
                return None, response, coordinates
        else:
            return None, response, coordinates

    def query_storage(self, query: str, doc_id, context_size=4) -> (List[Document], list):
        """
        Returns the context related to a given query
        """
        documents, coordinates = self._get_context(doc_id, query, context_size)

        context_as_text = [doc.page_content for doc in documents]
        return context_as_text, coordinates

    def query_storage_and_embeddings(self, query: str, doc_id, context_size=4):
        """
        Returns both the context and the embedding information from a given query
        """
        db = self.data_storage.embeddings_dict[doc_id]
        retriever = db.as_retriever(search_kwargs={"k": context_size}, search_type="similarity_with_embeddings")
        relevant_documents = retriever.get_relevant_documents(query)

        context_as_text = [doc.page_content for doc in relevant_documents]
        return context_as_text

        # chroma_collection.get(include=['embeddings'])['embeddings']

    def _parse_json(self, response, output_parser):
        system_message = "You are an useful assistant expert in materials science, physics, and chemistry " \
                         "that can process text and transform it to JSON."
        human_message = """Transform the text between three double quotes in JSON.\n\n\n\n
        {format_instructions}\n\nText: \"\"\"{text}\"\"\""""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

        prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        results = self.llm(
            prompt_template.format_prompt(
                text=response,
                format_instructions=output_parser.get_format_instructions()
            ).to_messages()
        )
        parsed_output = output_parser.parse(results.content)

        return parsed_output

    def _run_query(self, doc_id, query, context_size=4) -> (List[Document], list):
        relevant_documents = self._get_context(doc_id, query, context_size)
        relevant_document_coordinates = [doc.metadata['coordinates'].split(";") if 'coordinates' in doc.metadata else []
                                         for doc in
                                         relevant_documents]
        response = self.chain.run(input_documents=relevant_documents,
                                  question=query)

        if self.memory:
            self.memory.save_context({"input": query}, {"output": response})
        return response, relevant_document_coordinates

    def _get_context(self, doc_id, query, context_size=4) -> (List[Document], list):
        db = self.data_storage.embeddings_dict[doc_id]
        retriever = db.as_retriever(search_kwargs={"k": context_size})
        relevant_documents = retriever.get_relevant_documents(query)
        relevant_document_coordinates = [doc.metadata['coordinates'].split(";") if 'coordinates' in doc.metadata else []
                                         for doc in
                                         relevant_documents]
        if self.memory and len(self.memory.buffer_as_messages) > 0:
            relevant_documents.append(
                Document(
                    page_content="""Following, the previous question and answers. Use these information only when in the question there are unspecified references:\n{}\n\n""".format(
                        self.memory.buffer_as_str))
            )
        return relevant_documents, relevant_document_coordinates

    def get_full_context_by_document(self, doc_id):
        """
        Return the full context from the document
        """
        db = self.data_storage.embeddings_dict[doc_id]
        docs = db.get()
        return docs['documents']

    def _get_context_multiquery(self, doc_id, query, context_size=4):
        db = self.data_storage.embeddings_dict[doc_id].as_retriever(search_kwargs={"k": context_size})
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=db, llm=self.llm)
        relevant_documents = multi_query_retriever.get_relevant_documents(query)
        return relevant_documents

    def get_text_from_document(self, pdf_file_path, chunk_size=-1, perc_overlap=0.1, verbose=False):
        """
        Extract text from documents using Grobid.
        - if chunk_size is < 0, keeps each paragraph separately
        - if chunk_size > 0, aggregate all paragraphs and split them again using an approximate chunk size
        """
        if verbose:
            print("File", pdf_file_path)
        filename = Path(pdf_file_path).stem
        coordinates = True  # if chunk_size == -1 else False
        structure = self.grobid_processor.process(pdf_file_path, coordinates=coordinates)

        biblio = structure['biblio']
        biblio['filename'] = filename.replace(" ", "_")

        if verbose:
            print("Generating embeddings for:", hash, ", filename: ", filename)

        texts = []
        metadatas = []
        ids = []

        if chunk_size > 0:
            new_passages = self.text_merger.merge_passages(structure['passages'], chunk_size=chunk_size)
        else:
            new_passages = structure['passages']

        for passage in new_passages:
            biblio_copy = copy.copy(biblio)
            if len(str.strip(passage['text'])) > 0:
                texts.append(passage['text'])

                biblio_copy['type'] = passage['type']
                biblio_copy['section'] = passage['section']
                biblio_copy['subSection'] = passage['subSection']
                biblio_copy['coordinates'] = passage['coordinates']
                metadatas.append(biblio_copy)

                # ids.append(passage['passage_id'])

            ids = [id for id, t in enumerate(new_passages)]

        return texts, metadatas, ids

    def create_memory_embeddings(
            self,
            pdf_path,
            doc_id=None,
            chunk_size=500,
            perc_overlap=0.1
    ):
        texts, metadata, ids = self.get_text_from_document(
            pdf_path,
            chunk_size=chunk_size,
            perc_overlap=perc_overlap)
        if doc_id:
            hash = doc_id
        else:
            hash = metadata[0]['hash']

        self.data_storage.embed_document(hash, texts, metadata)

        return hash

    def create_embeddings(
            self,
            pdfs_dir_path: Path,
            chunk_size=500,
            perc_overlap=0.1,
            include_biblio=False
    ):
        input_files = []
        for root, dirs, files in os.walk(pdfs_dir_path, followlinks=False):
            for file_ in files:
                if not (file_.lower().endswith(".pdf")):
                    continue
                input_files.append(os.path.join(root, file_))

        for input_file in tqdm(input_files, total=len(input_files), unit='document',
                               desc="Grobid + embeddings processing"):

            md5 = self.calculate_md5(input_file)
            data_path = os.path.join(self.data_storage.embeddings_root_path, md5)

            if os.path.exists(data_path):
                print(data_path, "exists. Skipping it ")
                continue
            # include = ["biblio"] if include_biblio else []
            texts, metadata, ids = self.get_text_from_document(
                input_file,
                chunk_size=chunk_size,
                perc_overlap=perc_overlap)
            filename = metadata[0]['filename']

            vector_db_document = Chroma.from_texts(texts,
                                                   metadatas=metadata,
                                                   embedding=self.embedding_function,
                                                   persist_directory=data_path)
            vector_db_document.persist()

            with open(os.path.join(data_path, filename + ".storage_filename"), 'w') as fo:
                fo.write("")

    @staticmethod
    def calculate_md5(input_file: Union[Path, str]):
        import hashlib
        md5_hash = hashlib.md5()
        with open(input_file, 'rb') as fi:
            md5_hash.update(fi.read())
        return md5_hash.hexdigest().upper()
