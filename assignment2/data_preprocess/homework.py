import argparse
import re
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string
from bs4 import BeautifulSoup

def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", lambda m: re.sub(r"\d", "X", m.group(0)), text)
    text = re.sub(r"\+1\d{10}\b", lambda m: "+" + ("X" * (len(m.group(0)) - 1)), text)
    return text
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    cleaned_paragraphs = []
    for paragraph in text.split("\n"):
        if re.search(r"[A-Za-z0-9]{101,}", paragraph):
            continue
        if not any(ch in string.punctuation for ch in paragraph):
            continue
        cleaned_paragraphs.append(paragraph)
    return "\n".join(cleaned_paragraphs)


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    if not text or not text.strip():
        return False

    lowered = text.lower()
    bad_words = retrieve_bad_words()
    if any(bad_word in lowered for bad_word in bad_words):
        return False

    if not any(ch in string.punctuation for ch in text):
        return False

    allowed_chars = sum(
        ch.isalnum() or ch in string.punctuation or ch.isspace()
        for ch in text
    )
    return (allowed_chars / len(text)) >= 0.8


def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    stripped = text.strip()
    if not stripped:
        return False

    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if not alpha_chars:
        return False

    ascii_alpha = sum(ch.isascii() for ch in alpha_chars)
    non_whitespace = [ch for ch in stripped if not ch.isspace()]
    ascii_printable = sum(
        ch.isascii() and (ch.isalnum() or ch in string.punctuation)
        for ch in non_whitespace
    )

    return (ascii_alpha / len(alpha_chars)) >= 0.8 and (ascii_printable / len(non_whitespace)) >= 0.85
    

def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """
    deduplicated = []
    seen_normalized = set()
    seen_token_sets = []

    for text in texts:
        normalized = " ".join(text.lower().split())
        if not normalized or normalized in seen_normalized:
            continue

        token_set = set(re.findall(r"\w+", normalized))
        is_duplicate = False
        for existing_set in seen_token_sets:
            union = token_set | existing_set
            similarity = len(token_set & existing_set) / len(union) if union else 1.0
            if similarity >= 0.8:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(text)
            seen_normalized.add(normalized)
            seen_token_sets.append(token_set)

    return deduplicated


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")