import json
import torch
from torch_geometric.data import Data
import click
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import tiktoken
from tqdm import tqdm
import spacy
from pydantic import BaseModel

load_dotenv()

class NodeFeatures(BaseModel):
    date: str = None
    summary: str = None
    authors: list = None
    content: str = None
    category: str = None
    topic: str = None
    redactability_index: float = None
    topic_index: int = None
    quarter_index: int = None
    embedding: list = None
    summary_embedding: list = None
    question_embedding: list = []

class ParagraphFeatures(BaseModel):
    index: int
    content: str
    embedding: list

class SentenceFeatures(BaseModel):
    index: int
    content: str
    embedding: list

class CombinedGraphCreator:
    def __init__(self, json_file, output_file):
        self.json_file = json_file
        self.output_file = output_file
        self.api_key = os.environ.get("OPEN_AI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load the JSON data
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

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

    def create_and_embed_graph(self):
        # Define node features, paragraph features, and sentence features
        node_features = []
        paragraph_features = []
        sentence_features = []

        # Define edge index and edge types
        edge_index = []
        edge_types = []

        # PyTorch Geometric uses a single integer to represent each node in the graph
        # We keep track of the current node index
        current_node_index = 0

        for node_data in tqdm(self.data, desc="Processing nodes", total=len(self.data)):
            # Split content into paragraphs and sentences
            main_content = self.clean_text(node_data.get('content', ''))
            node_summary = self.clean_text(node_data.get('summary', ''))
            paragraphs, sentences = self.split_content(main_content)

            # Gather all text elements for batching
            texts_to_embed = [node_summary, main_content] + paragraphs + sentences

            # Get embeddings
            node_embeddings = self.batch_and_embed(texts_to_embed)

            # Create NodeFeatures object
            node_feature = NodeFeatures(
                date=node_data.get('date'),
                summary=node_data.get('summary'),
                authors=node_data.get('authors'),
                content=main_content,
                category=node_data.get('category', None),
                topic=node_data.get('topic', None),
                redactability_index=node_data.get('redactability_index', None),
                topic_index=node_data.get('topic_index', None),
                quarter_index=node_data.get('quarter_index'),
                embedding=node_embeddings[1],
                summary_embedding=node_embeddings[0],
                question_embedding=[]
            )
            node_features.append(node_feature)

            # Create ParagraphFeatures objects
            for i, paragraph in enumerate(paragraphs):
                paragraph_feature = ParagraphFeatures(
                    index=i,
                    content=paragraph,
                    embedding=node_embeddings[2 + i]
                )
                paragraph_features.append(paragraph_feature)

                # Add edges connecting the node to its paragraph
                edge_index.append([current_node_index, len(node_features) + len(paragraph_features) - 1])
                edge_types.append(0)  # 0 represents node-to-paragraph edge

            # Create SentenceFeatures objects
            for i, sentence in enumerate(sentences):
                sentence_feature = SentenceFeatures(
                    index=i,
                    content=sentence,
                    embedding=node_embeddings[2 + len(paragraphs) + i]
                )
                sentence_features.append(sentence_feature)

                # Add edges connecting the node to its sentence
                edge_index.append([current_node_index, len(node_features) + len(paragraph_features) + len(sentence_features) - 1])
                edge_types.append(1)  # 1 represents node-to-sentence edge

            current_node_index += 1

        # Create the Data object
        graph_data = Data(
            x=node_features,  # Node features
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),  # Edge index
            edge_attr=torch.tensor(edge_types, dtype=torch.long),  # Edge types
            paragraph_features=paragraph_features,  # Paragraph features
            sentence_features=sentence_features  # Sentence features
        )

        torch.save(graph_data, self.output_file)

    def add_node(self):
        data = torch.load(self.output_file)

        # Parse new node JSON data
        node_data = self.data
        main_content = self.clean_text(node_data.get('content', ''))
        summary = self.clean_text(node_data.get('summary', ''))

        # Split content into paragraphs and sentences using spaCy
        paragraphs, sentences = self.split_content(main_content)
        texts_to_embed = [summary, main_content] + paragraphs + sentences

        # Get embeddings
        embeddings = self.batch_and_embed(texts_to_embed)

        # Create NodeFeatures object for the new node
        new_node_feature = NodeFeatures(
            date=node_data.get('date'),
            summary=node_data.get('summary', None),
            authors=node_data.get('authors'),
            content=main_content,
            category=node_data.get('category', None),
            topic=node_data.get('topic', None),
            redactability_index=node_data.get('redactability_index', None),
            topic_index=node_data.get('topic_index'),
            quarter_index=node_data.get('quarter_index', None),
            embedding=embeddings[1],
            summary_embedding=embeddings[0],
            question_embedding=[]  # Placeholder for question embedding
        )

        # Create ParagraphFeatures objects for the new node
        new_paragraph_features = []
        for i, paragraph in enumerate(paragraphs):
            paragraph_feature = ParagraphFeatures(
                index=i,
                content=paragraph,
                embedding=embeddings[2 + i]
            )
            new_paragraph_features.append(paragraph_feature)

        # Create SentenceFeatures objects for the new node
        new_sentence_features = []
        for i, sentence in enumerate(sentences):
            sentence_feature = SentenceFeatures(
                index=i,
                content=sentence,
                embedding=embeddings[2 + len(paragraphs) + i]
            )
            new_sentence_features.append(sentence_feature)

        # Update the graph data
        data.x.append(new_node_feature)
        data.paragraph_features.extend(new_paragraph_features)
        data.sentence_features.extend(new_sentence_features)

        # Update the edge index and edge types
        new_node_index = data.num_nodes - 1
        for i in range(len(new_paragraph_features)):
            data.edge_index = torch.cat((data.edge_index, torch.tensor([[new_node_index, data.num_nodes + i]]).t()), dim=1)
            data.edge_attr = torch.cat((data.edge_attr, torch.tensor([0])))  # 0 represents node-to-paragraph edge
        for i in range(len(new_sentence_features)):
            data.edge_index = torch.cat((data.edge_index, torch.tensor([[new_node_index, data.num_nodes + len(new_paragraph_features) + i]]).t()), dim=1)
            data.edge_attr = torch.cat((data.edge_attr, torch.tensor([1])))  # 1 represents node-to-sentence edge

        # Save the updated graph
        torch.save(data, self.output_file)

@click.command()
@click.option('--json_file', type=click.Path(exists=True), required=True, help='Path to the JSON file')
@click.option('--output_file', type=click.Path(), required=True, help='Path to save the graph')
@click.option('--add_node', type=click.Path(exists=True), help='Path to the JSON file containing the new node data')
def main(json_file, output_file, add_node):
    creator = CombinedGraphCreator(json_file, output_file)
    
    if add_node:
        # Load the new node data from the JSON file
        with open(add_node, 'r') as f:
            new_node_data = json.load(f)
        
        # Set the new node data in the creator object
        creator.data = new_node_data
        
        # Add the new node to the graph
        creator.add_node()
    else:
        # Create and embed the graph from the input JSON file
        creator.create_and_embed_graph()

if __name__ == "__main__":
    main()