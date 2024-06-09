import json
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
import click
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import tiktoken
from tqdm import tqdm
import spacy

load_dotenv()

class NodeAdder:
    def __init__(self, graph_file):
        self.graph_file = graph_file
        self.api_key = os.environ.get("OPEN_AI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        return re.sub(r'[^\w\s]', '', text.replace("\n", " ")).strip()

    def get_embedding(self, texts):
        embeddings = self.client.embeddings.create(input=texts, model="text-embedding-3-large")
        return [data.embedding for data in embeddings.data]

    def batch_and_embed(self, texts):
        # Calculate token counts and batch texts
        token_counts = [len(self.tokenizer.encode(text)) for text in texts]
        batch_texts = []
        current_batch = []
        current_token_count = 0
        max_tokens = 8191

        for text, count in zip(texts, token_counts):
            if current_token_count + count > max_tokens:
                if current_batch:
                    batch_texts.append(current_batch)
                current_batch = [text]
                current_token_count = count
            else:
                current_batch.append(text)
                current_token_count += count

        if current_batch:
            batch_texts.append(current_batch)

        # Embed batches and collect embeddings
        all_embeddings = []
        for batch in batch_texts:
            batch_embeddings = self.get_embedding(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def split_content(self, content):
        doc = self.nlp(content)
        paragraphs = [self.clean_text(p.text) for p in doc.sents]
        sentences = [self.clean_text(sent.text) for sent in doc.sents]
        return paragraphs, sentences

    def add_node(self, new_node_json):
        data = torch.load(self.graph_file)
        G = to_networkx(data)

        # Parse new node JSON data
        node_data = new_node_json
        main_content = self.clean_text(node_data.get('content', ''))
        summary = self.clean_text(node_data.get('summary', ''))

        # Split content into paragraphs and sentences using spaCy
        paragraphs, sentences = self.split_content(main_content)
        texts_to_embed = [main_content, summary] + paragraphs + sentences

        # Get embeddings
        embeddings = self.batch_and_embed(texts_to_embed)
        default_attributes = {
                'date': node_data.get('date'),
                'summary': node_data.get('summary'),
                'authors': node_data.get('authors'),
                'content': main_content,
                'category': node_data.get('category'),
                'topic': node_data.get('topic'),
                'redactability_index': node_data.get('redactability_index'),
                'topic_index': node_data.get('topic_index'),
                'quarter_index': node_data.get('quarter_index'),
                'parent': None,
                'relation': None,
                'embedding': embeddings[0],
                'summary_embedding': embeddings[1],
                'question_embedding': None, # Placeholder for question embedding - whenever we answer a question using the correct nodes we can update this (a simple thumbs up/down in the chat UI after the query could be used to define if the question goes here)
            }

        # Create main node
        node_idx = len(G.nodes())
        G.add_node(node_idx, **default_attributes)

        print("New node attributes:")
        print(G.nodes[node_idx])

        # Adding paragraph nodes
        for i, paragraph in enumerate(paragraphs):
            paragraph_attributes = {**default_attributes, 'content': paragraph, 'embedding': embeddings[2 + i], 'parent': node_idx}
            paragraph_id = f"{node_idx}_p{i}"
            G.add_node(paragraph_id, **paragraph_attributes)
            G.add_edge(node_idx, paragraph_id, relation='contains_paragraph')

        # Adding sentence nodes
        offset = 2 + len(paragraphs)
        for i, sentence in enumerate(sentences):
            sentence_attributes = {**default_attributes, 'content': sentence, 'embedding': embeddings[offset + i], 'parent': node_idx}
            sentence_id = f"{node_idx}_s{i}"
            G.add_node(sentence_id, **sentence_attributes)
            G.add_edge(node_idx, sentence_id, relation='contains_sentence')

        print("Existing node attributes:")
        for node in G.nodes(data=True):
            print(node[1])

        # Save updated graph
        updated_data = from_networkx(G)
        torch.save(updated_data, self.graph_file)

@click.command()
@click.option('--graph_file', type=click.Path(exists=True), required=True, help='Path to the graph file')
@click.option('--new_node_json', type=str, required=True, help='New node data in JSON format')
def main(graph_file, new_node_json):
    adder = NodeAdder(graph_file)
    adder.add_node(new_node_json)

if __name__ == "__main__":
    main()