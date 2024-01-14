from bs4 import BeautifulSoup
from document_qa.grobid_processors import get_xml_nodes_body, get_xml_nodes_figures, get_xml_nodes_header


def test_get_xml_nodes_body_paragraphs():
    with open("resources/2312.07559.paragraphs.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    nodes = get_xml_nodes_body(soup, use_paragraphs=True)

    assert len(nodes) == 70


def test_get_xml_nodes_body_sentences():
    with open("resources/2312.07559.sentences.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_body(soup, use_paragraphs=False)

    assert len(children) == 327


def test_get_xml_nodes_figures():
    with open("resources/2312.07559.paragraphs.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_figures(soup)

    assert len(children) == 13


def test_get_xml_nodes_header_paragraphs():
    with open("resources/2312.07559.paragraphs.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup)

    assert len(children) == 8

def test_get_xml_nodes_header_sentences():
    with open("resources/2312.07559.sentences.tei.xml", 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup, use_paragraphs=False)

    assert len(children) == 15
