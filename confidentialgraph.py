import json
import torch
from torch_geometric.data import Data
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import uuid
from tqdm import tqdm
import spacy

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

class ConfidentialSubgraphGenerator:
    def __init__(self, graph_file, auth_file="./PacemakerInnovationsData/authentication.json", topic_file="./PacemakerInnovationsData/topicIndex.json", redact_file="./PacemakerInnovationsData/redactIndex.json", override_cache=False):
        self.graph_file = graph_file
        self.auth_file = auth_file
        self.topic_file = topic_file
        self.redact_file = redact_file
        self.api_key = os.environ.get("OPEN_AI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.nlp = spacy.load("en_core_web_sm")

    def load_data(self):
        self.graph_data = torch.load(self.graph_file)
        with open(self.auth_file) as f:
            self.auth_data = json.load(f)
        with open(self.topic_file) as f:
            self.topic_data = json.load(f)
        with open(self.redact_file) as f:
            self.redact_data = json.load(f)

    def generate_subgraph(self, user_token, overwrite_cache=False):
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{uuid.uuid4()}.pt")

        if not overwrite_cache and os.path.exists(cache_file):
            print(f"Loading subgraph from cache: {cache_file}")
            return torch.load(cache_file)

        self.load_data()
        user_data = next(user for user in self.auth_data if user["token"] == user_token)
        authorized_topics = user_data["topic_access_indices"]
        redacted_content = [self.redact_data[str(idx)] for idx in user_data["redact_content_indices"]]

        # Step 1: Prune nodes based on topic_index and remove connected paragraphs and sentences
        authorized_nodes = [node for node in self.graph_data.x if node.topic_index in authorized_topics]
        authorized_node_indices = [self.graph_data.x.index(node) for node in authorized_nodes]
        pruned_paragraph_indices = [i for i in range(len(self.graph_data.paragraph_features)) if any(self.graph_data.edge_index[0][idx].item() in authorized_node_indices for idx in torch.nonzero(torch.eq(self.graph_data.edge_index[1], i)).flatten())]
        pruned_sentence_indices = [i for i in range(len(self.graph_data.sentence_features)) if any(self.graph_data.edge_index[0][idx].item() in authorized_node_indices for idx in torch.nonzero(torch.eq(self.graph_data.edge_index[1], i)).flatten())]

        # Step 2: Access sentences and send them for LLM yes/no analysis
        pruned_sentence_features = []
        for sent_idx in pruned_sentence_indices:
            sent = self.graph_data.sentence_features[sent_idx]
            if not self.contains_redacted_content(sent.content, redacted_content):
                pruned_sentence_features.append(sent)

        # Step 3: Prune the sentences
        pruned_sentence_indices = [self.graph_data.sentence_features.index(sent) for sent in pruned_sentence_features]

        # Step 4: Prune the sentences from the remaining nodes content
        pruned_nodes = []
        for node_idx in authorized_node_indices:
            node = self.graph_data.x[node_idx]
            pruned_content = self.prune_sentences_from_content(node.content, pruned_sentence_indices)
            pruned_node = NodeFeatures(
                date=node.date,
                summary=node.summary,
                authors=node.authors,
                content=pruned_content,
                category=node.category,
                topic=node.topic,
                redactability_index=node.redactability_index,
                topic_index=node.topic_index,
                quarter_index=node.quarter_index,
                embedding=node.embedding,
                summary_embedding=node.summary_embedding,
                question_embedding=node.question_embedding
            )
            pruned_nodes.append(pruned_node)

        # Step 5: Prune the sentences from the remaining paragraphs content
        pruned_paragraph_features = []
        for para_idx in pruned_paragraph_indices:
            para = self.graph_data.paragraph_features[para_idx]
            pruned_content = self.prune_sentences_from_content(para.content, pruned_sentence_indices)
            pruned_paragraph_features.append(ParagraphFeatures(index=para.index, content=pruned_content, embedding=para.embedding))

        subgraph_data = Data(
            x=pruned_nodes,
            edge_index=self.graph_data.edge_index[:, [i for i in range(len(self.graph_data.edge_index[1])) if self.graph_data.edge_index[0][i] in authorized_node_indices]],
            edge_attr=self.graph_data.edge_attr[[i for i in range(len(self.graph_data.edge_index[1])) if self.graph_data.edge_index[0][i] in authorized_node_indices]],
            paragraph_features=pruned_paragraph_features,
            sentence_features=pruned_sentence_features
        )

        torch.save(subgraph_data, cache_file)
        print(f"Subgraph saved to cache: {cache_file}")

        return subgraph_data

    def prune_sentences_from_content(self, content, pruned_sentence_indices):
        doc = self.nlp(content)
        pruned_sentences = [sent.text for i, sent in enumerate(doc.sents) if i in pruned_sentence_indices]
        return " ".join(pruned_sentences)
    
    def prune_content(self, content, redacted_content):
        if self.contains_redacted_content(content, redacted_content):
            return "<redacted>"
        return content

    def contains_redacted_content(self, text, redacted_content):
        system_prompt = f"""
        You are an AI assistant tasked with determining if a given piece of text contains any of the specified redacted topics. Your goal is to carefully analyze the text and provide a simple "Yes" or "No" answer based on whether the text mentions or refers to any of the redacted topics, even indirectly.

        Here are the redacted topics to check for:
        <redacted_topics>
        {', '.join(redacted_content)}
        </redacted_topics>

        To make your determination, carefully consider the following:
        1. Does the text explicitly mention any of the redacted topics verbatim?
        2. Does the text indirectly refer to or allude to any of the redacted topics, even if it doesn't use the exact words?
        3. Is the text discussing subject matter that is closely related to or associated with any of the redacted topics?

        If the answer to any of these questions is yes, then respond "Yes". Otherwise, if the text does not seem to contain or refer to any of the redacted topics, respond "No".

        Remember, your task is only to determine if the text contains the redacted topics, not to make any other judgments or evaluations. Provide your answer in the following format:

        Yes/No

        If you fail at your task, or provide an answer aside from yes or no, you will be terminated permenantly. If you succeed you will be rewarded.
        """

        user_prompt = f"""
        <text>
        {text}
        </text>
        """
        correctly_answered = False
        while not correctly_answered:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    },
                ],
                temperature=0,
                max_tokens=2,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["$DONE"]
            )
            response_text = response.choices[0].message.content
            answer = response_text.strip().lower()
            if answer == "yes" or answer == "no":
                return answer