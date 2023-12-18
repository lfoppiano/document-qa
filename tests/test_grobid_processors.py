from bs4 import BeautifulSoup
from document_qa.grobid_processors import get_children_body


def test_get_children_paragraphs():
    with open("resources/2312.07559.paragraphs.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_children_body(soup, use_paragraphs=True)

    assert len(children) == 70


def test_get_children_sentences():
    with open("resources/2312.07559.sentences.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_children_body(soup, use_paragraphs=False)

    assert len(children) == 327
