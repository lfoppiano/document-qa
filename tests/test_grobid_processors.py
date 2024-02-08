import os

from bs4 import BeautifulSoup
from document_qa.grobid_processors import get_xml_nodes_body, get_xml_nodes_figures, get_xml_nodes_header
from tests.resources import TEST_DATA_PATH


def test_get_xml_nodes_body_paragraphs():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    nodes = get_xml_nodes_body(soup, use_paragraphs=True)

    assert len(nodes) == 70


def test_get_xml_nodes_body_sentences():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.sentences.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_body(soup, use_paragraphs=False)

    assert len(children) == 327


def test_get_xml_nodes_figures():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_figures(soup)

    assert len(children) == 13


def test_get_xml_nodes_header_paragraphs():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup)

    assert sum([len(child) for k, child in children.items()]) == 8


def test_get_xml_nodes_header_sentences():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.sentences.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup, use_paragraphs=False)

    assert sum([len(child) for k, child in children.items()]) == 15
