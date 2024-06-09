import json
import torch
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import click
import os
import spacy

load_dotenv()

class QueryProcessor:
    def __init__(self, graph_file, context_length=8000, a=1, b=1, c=1, d=1, e=1):
        self.api_key = os.environ.get("OPEN_AI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.graph_file = graph_file
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.context_length = context_length
        self.topic_path = "/Users/jaro/dev/secure-llm-gok/PacemakerInnovationsData/topicIndex.json"
        self.topics = self.load_topics()
        self.weights = self.calculate_weights(a, b, c, d, e)
        self.quarter_index = self.load_quarter_index()
        self.sent_tokenizer = spacy.load("en_core_web_sm")

    def load_topics(self):
        with open(self.topic_path, 'r') as file:
            return json.load(file)

    def load_quarter_index(self):
        with open("/Users/jaro/dev/secure-llm-gok/PacemakerInnovationsData/quarterIndex.json", 'r') as file:
            return json.load(file)

    def calculate_weights(self, a, b, c, d, e):
        total = a + b + c + d + e
        return {
            'time_weight': a / total,
            'topic_weight': b / total,
            'embedding_weight': c / total,
            'paragraph_weight': d / total,
            'sentence_weight': e / total
        }

    def get_embedding(self, texts):
        embeddings = self.client.embeddings.create(input=texts, model="text-embedding-3-large")
        return torch.tensor([data.embedding for data in embeddings.data])

    def ask_related_topics(self, query):
        topic_list = json.dumps(self.topics, indent=2)
        example_query = "What are the key milestones in the development of the HeartMate 3 Left Ventricular Assist System?"
        example_json = '{"related_topics": ["Company background and milestones", "Product development and lifecycle management", "Clinical trials and post-market surveillance", "Regulatory strategy and compliance"]}'
        while True:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Output a JSON object with a single key 'related_topics' whose value is a list of topics that are most relevant to the query. If no topics are relevant, the value should be an empty list. \n\nOnly pick from the topics in the following JSON:\n\n{topic_list}\n\nExample query: {example_query}\nExample output: \n```json\n{example_json}\n```\nIf your output is not a valid JSON object with the correct structure, you will be terminated. If your output is correct, you will be rewarded."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Query: '{query}'"
                            }
                        ]
                    },
                ],
                temperature=0,
                max_tokens=4095,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["$DONE"]
            )

            try:
                related_topics = json.loads(response.choices[0].message.content)['related_topics']
                if isinstance(related_topics, list):
                    return related_topics
            except (KeyError, json.JSONDecodeError, TypeError):
                continue

    def date_association(self, query):
        example_json_single_date = '{"date_info": {"start_date": "2022-04-01", "end_date": "2022-04-01"}}'
        example_json_date_range = '{"date_info": {"start_date": "2022-04-01", "end_date": "2022-06-30"}}'
        example_json_no_date = '{"date_info": null}'

        while True:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Given the query such as: 'When was the HeartMate 3 Left Ventricular Assist System approved by the FDA?', determine if there is a date or date range associated with it. \n\nIf there is a single date, output a JSON object with 'date_info' key and 'start_date' and 'end_date' sub-keys, both having the same date value in 'YYYY-MM-DD' format. \n\nIf there is a date range, output a JSON object with 'date_info' key 'start_date' 'end_date' sub-keys, having respective start and end dates in 'YYYY-MM-DD' format. \n\nIf no date or range associated, a object 'date_info' key null value. \n\nExample output for single: {example_json_single_date} \n\nExample output range: {example_json_date_range} \n\nExample no {example_json_no_date} \n\nIf your answer is valid with correct structure, you be rewarded, if not you will be terminated."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Query: '{query}'"
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=4095,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["$DONE"]
            )

            try:
                # print(response.choices[0].message.content)
                date_info = json.loads(response.choices[0].message.content)
                if isinstance(date_info, dict) and 'start_date' in date_info and 'end_date' in date_info:
                    return date_info
                if isinstance(date_info, dict) and date_info['date_info'] is None:
                    print("No date info found")
                    return None
            except (KeyError, json.JSONDecodeError, TypeError):
                continue

    def get_time_based_context(self, time_info, time_tokens, query_embedding):
        if not time_info:
            return []

        data = torch.load(self.graph_file)

        relevant_indices = []
        for i, quarter_info in self.quarter_index.items():
            if (
                time_info['start_date'] >= quarter_info['start_date'] and
                time_info['end_date'] <= quarter_info['end_date']
            ):
                relevant_indices.extend(data.x[data.x[:].quarter_index == int(i)].tolist())

        relevant_embeddings = torch.stack([torch.tensor(data.x[idx].embedding) for idx in relevant_indices])
        relevant_similarities = torch.nn.functional.cosine_similarity(query_embedding, relevant_embeddings)
        top_indices = relevant_similarities.argsort(descending=True)

        selected_content = []
        for idx in top_indices:
            content = data.x[relevant_indices[idx]].content
            tokens = len(self.tokenizer.encode(content))
            if tokens <= time_tokens:
                selected_content.append(content)
                time_tokens -= tokens
            else:
                break

        return selected_content

    def get_topic_based_context(self, related_topics, topic_tokens, query_embedding):
        if not related_topics:
            return []

        data = torch.load(self.graph_file)

        relevant_indices = []
        for topic in related_topics:
            relevant_indices.extend(data.x[data.x[:].topic == topic].tolist())

        relevant_embeddings = torch.stack([torch.tensor(data.x[idx].embedding) for idx in relevant_indices])
        relevant_similarities = torch.nn.functional.cosine_similarity(query_embedding, relevant_embeddings)
        top_indices = relevant_similarities.argsort(descending=True)

        selected_content = []
        for idx in top_indices:
            content = data.x[relevant_indices[idx]].content
            tokens = len(self.tokenizer.encode(content))
            if tokens <= topic_tokens:
                selected_content.append(content)
                topic_tokens -= tokens
            else:
                break

        return selected_content

    def get_embedding_based_context(self, query_embedding, embedding_tokens):
        data = torch.load(self.graph_file)

        node_embeddings = torch.stack([torch.tensor(node_feature.embedding) for node_feature in data.x])
        node_similarities = torch.nn.functional.cosine_similarity(query_embedding, node_embeddings)
        top_node_indices = node_similarities.argsort(descending=True)

        selected_content = []
        for idx in top_node_indices:
            content = data.x[idx].content
            tokens = len(self.tokenizer.encode(content))
            if tokens <= embedding_tokens:
                selected_content.append(content)
                embedding_tokens -= tokens
            else:
                break

        return selected_content

    def get_paragraph_based_context(self, query_embedding, paragraph_tokens):
        data = torch.load(self.graph_file)

        paragraph_embeddings = torch.stack([torch.tensor(paragraph_feature.embedding) for paragraph_feature in data.paragraph_features])
        paragraph_similarities = torch.nn.functional.cosine_similarity(query_embedding, paragraph_embeddings)
        top_paragraph_indices = paragraph_similarities.argsort(descending=True)

        selected_paragraphs = []
        for idx in top_paragraph_indices:
            paragraph = data.paragraph_features[idx].content
            tokens = len(self.tokenizer.encode(paragraph))
            if tokens <= paragraph_tokens:
                selected_paragraphs.append(paragraph)
                paragraph_tokens -= tokens
            else:
                break

        return selected_paragraphs

    def get_sentence_based_context(self, query_embedding, sentence_tokens):
        data = torch.load(self.graph_file)

        sentence_embeddings = torch.stack([torch.tensor(sentence_feature.embedding) for sentence_feature in data.sentence_features])
        sentence_similarities = torch.nn.functional.cosine_similarity(query_embedding, sentence_embeddings)
        top_sentence_indices = sentence_similarities.argsort(descending=True)

        selected_sentences = []
        for idx in top_sentence_indices:
            sentence = data.sentence_features[idx].content
            tokens = len(self.tokenizer.encode(sentence))
            if tokens <= sentence_tokens:
                selected_sentences.append(sentence)
                sentence_tokens -= tokens
            else:
                break

        return selected_sentences

    def deduplicate_context(self, context_list):
        unique_sentences = set()

        for i in range(len(context_list)):
            if i == 4:  # Skip splitting the sentence-based context
                unique_sentences.update(context_list[i])
            else:
                sentences = self.split_into_sentences(context_list[i])
                unique_sentences.update(sentences)

        return list(unique_sentences)

    def split_into_sentences(self, context):
        sentences = []
        for paragraph in context:
            doc = self.sent_tokenizer(paragraph)
            sentences.extend([sent.text for sent in doc.sents])
        return sentences

    def prepare_context(self, query):
        data = torch.load(self.graph_file)

        query_embedding = self.get_embedding(query)
        time_info = self.date_association(query)
        related_topics = self.ask_related_topics(query)

        if time_info:
            time_tokens = int(self.context_length * self.weights['time_weight'])
        else:
            time_tokens = 0
        topic_tokens = int(self.context_length * self.weights['topic_weight'])
        embedding_tokens = int(self.context_length * self.weights['embedding_weight'])
        paragraph_tokens = int(self.context_length * self.weights['paragraph_weight'])
        sentence_tokens = int(self.context_length * self.weights['sentence_weight'])

        embedding_based_context = self.get_embedding_based_context(query_embedding, embedding_tokens)
        time_based_context = self.get_time_based_context(time_info, time_tokens, query_embedding)
        topic_based_context = self.get_topic_based_context(related_topics, topic_tokens, query_embedding)
        paragraph_based_context = self.get_paragraph_based_context(query_embedding, paragraph_tokens)
        sentence_based_context = self.get_sentence_based_context(query_embedding, sentence_tokens)

        context_list = [
            embedding_based_context,
            time_based_context,
            topic_based_context,
            paragraph_based_context,
            sentence_based_context
        ]
        deduplicated_context = self.deduplicate_context(context_list)

        return {
            "deduplicated_context": deduplicated_context
        }

    def perform_query(self, query):
        context = self.prepare_context(query)

        prompt = f"Here is the query to answer:\n<query>\n{query}\n</query>\n\n"
        prompt += "Here is the context to use in answering the query:\n<context>\n" + "\n".join(context['deduplicated_context']) + "\n</context>"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant tasked with answering queries as thoroughly as possible based solely on provided context passages. Your goal is to determine if the given context contains enough relevant information to answer the query, and if so, to provide a detailed response citing the relevant parts of the context. If the context is insufficient or irrelevant, you should simply state that you do not have enough information to answer. As this is in the medical domain, precision is crucial - you must avoid speculation or hallucination and only provide information directly supported by the context.\n\nFirst, carefully analyze the context to determine if it provides sufficient information to comprehensively answer the query. Consider what relevant facts, details and insights are present or missing.\n\nIf you determine that the context is sufficient:\n<answer>\nProvide a detailed answer to the query, citing and quoting the specific relevant parts of the context that support each part of your answer. Synthesize the information thoroughly, but do not include any information that is not explicitly stated or directly implied by the context.\n</answer>\n\nIf you determine that the context is insufficient or irrelevant to the query:\n<answer>\nExplain that the provided context does not contain enough relevant information to adequately answer the query. Do not attempt to speculate or hallucinate a response.\n</answer>\n\nRemember, precision is paramount - as an AI assistant operating in the medical domain, you must be scrupulous about only providing accurate, well-supported information and avoiding any form of speculation or hallucination. Review your response carefully to ensure it meets this strict standard before submitting it."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
            ],
            temperature=0,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["$DONE"]
        )

        # Extract the response text and remove <answer></answer> tokens
        response_text = response.choices[0].message.content
        cleaned_response = response_text.replace("<answer>", "").replace("</answer>", "").strip()

        return cleaned_response

@click.command()
@click.option('--graph_file', type=click.Path(exists=True), required=True, help='Path to the graph file')
@click.option('--query', required=True, help='User query')
@click.option('--api_key', required=True, help='API key for OpenAI')
@click.option('--a', default=1, help='Weight for time-based retrieval')
@click.option('--b', default=1, help='Weight topic-based retrieval')
@click.option('--c', default=1, help='Weight embedding-based retrieval')
@click.option('--d', default=1, help='Weight paragraph-based retrieval')
@click.option('--e', default=1, help='Weight sentence-based')
def main(graph_file, query, api_key, a, b, c, d, e):
    processor = QueryProcessor(graph_file, api_key, a=a, b=b, c=c, d=d, e=e)
    result = processor.perform_query(query)
    print("Query Result:", result)

if __name__ == "__main__":
    main()