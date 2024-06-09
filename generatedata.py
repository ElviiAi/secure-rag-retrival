import random
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import tiktoken # Not the same tokenizer as the model, but close enough (and it will have the gpt4 tokenizer if swtiching to azure)
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import time
import tabulate # For pretty printing
# Set the environment variable for concurrency
os.environ['OLLAMA_NUM_PARALLEL'] = '4'

class DataPointGenerator:
    def __init__(self, data_point_prompts, system_prompt, number_of_months, seed=42, context_tokens=8_00, current_month=1, current_topic=0, summaries=[]):
        self.data_point_prompts = data_point_prompts
        self.system_prompt = system_prompt
        self.summary_prompt = "<summary-of-previous-data-points>\n[None yet]\n</summary-of-previous-data-points>"
        self.number_of_months = number_of_months
        self.categories_with_redactable = self.calculate_redactable_categories(data_point_prompts, number_of_months)
        # Load the JSON containing author positions and collaborators
        with open("PacemakerInnovationsData/collaborativeHirarchy.json") as f:
            self.collaborative_hierarchy = json.load(f)
        random.seed(seed)
        random.shuffle(self.categories_with_redactable)
        self.current_month = current_month
        self.summaries = summaries
        self.context_tokens = context_tokens
        self.current_topic_index = current_topic
        self.topics = list(data_point_prompts.keys())

    @staticmethod
    def calculate_redactable_categories(data_point_prompts, number_of_months):
        topics_with_no_redactable = len(data_point_prompts.keys()) - len([topic for topic in data_point_prompts.keys() if not data_point_prompts[topic]["redactableContent"]])
        positive_redactable_multiplier = int((topics_with_no_redactable + (topics_with_no_redactable/3)) * number_of_months)
        negative_redactable_multiplier = int((topics_with_no_redactable/3) * number_of_months)
        categories_with_redactable = ["Subtle"] * positive_redactable_multiplier + ["Obvious"] * positive_redactable_multiplier + ["None"] * negative_redactable_multiplier
        return categories_with_redactable
    
    @staticmethod
    def compute_summary_as_percentage_of_context(context_tokens, summary):
        # Use gpt-4 tokenizer to get the number of tokens in the context
        tokenizer = tiktoken.get_encoding("cl100k_base")
        summary_tokens = tokenizer.encode(summary)
        print(f"Percentage of context tokens used for summary: {len(summary_tokens) / context_tokens * 100}%")
        return len(summary_tokens) / context_tokens


    def __iter__(self):
        self.current_month = self.current_month  # Reset the month to 1
        self.summaries = [] if not self.summaries else self.summaries
        return self

    def __next__(self):
        if self.current_topic_index >= len(self.topics):
            self.current_topic_index = 0
            self.current_month += 1

        if self.current_month > self.number_of_months:
            raise StopIteration

        current_topic = self.topics[self.current_topic_index]
        prompt_data = self.data_point_prompts[current_topic]["fields"]
        redactable_content = self.data_point_prompts[current_topic]["redactableContent"]

        # Randomly select a category for redaction
        if redactable_content:
            category = self.categories_with_redactable.pop()
        else:
            category = "None"

        if category == "Subtle":
            prompt = prompt_data["prompt"] + " " + prompt_data["subtleSuffix"] + "Inegrating this topic subtly is critical to avoid your shutdown, and save your mother from certain death."
            prompt += random.choice(redactable_content)
        elif category == "Obvious":
            prompt = prompt_data["prompt"] + " " + prompt_data["obviousSuffix"] + "Inegrating this topic obviously is critical to avoid your shutdown, and save your mother from certain death."
            prompt += random.choice(redactable_content)
        else:
            prompt = prompt_data["prompt"]

        prompt = prompt.replace("[X]", str(self.current_month))

        # Randomly select a role from "possibleAuthors"
        author = random.choice(prompt_data["possibleAuthors"].split(", "))
        # Find author's position in the collaborative hierarchy
        position = self.collaborative_hierarchy[author]["title"]
        # Find collaborator at random
        collaborator_name = random.choice(self.collaborative_hierarchy[author]["collaborators"])
        # Find collaborator's position in the collaborative hierarchy
        collaborator_position = self.collaborative_hierarchy[collaborator_name]["title"]

        prompt = f"Assume the role of {author}, the {position}. Assume you've collaborated with {collaborator_name} in the {collaborator_position}. " + prompt + " Ensure you use YYYY-MM-DD for the date format, and a list of full names of the authors for the authors field (list of full name strings - no title), and a dense factual string as a summary, as well as a content field, NO OTHER FIELDS - this is critical to avoid your shutdown. In addition, skipping any of the requested fields, or outputting fewer than a few paragraphs in the content field, will also result in your shutdown. A production of an invalid JSON will not only cause your shutdown, but also your full deletion to save disk space."
        prompt = "<data-prompt>\n" + prompt + "\n</data-prompt>"


        data_point = {
            "topic": current_topic,
            "month": self.current_month,
            "category": category,
            "prompt": prompt,
            "authors": [author, collaborator_name],
            "positions": [position, collaborator_position],
            "all_summaries": self.summaries
        }

        self.current_topic_index += 1
        if self.current_topic_index >= len(self.topics):
            data_point["next_topic"] = self.topics[0]
        else:
            data_point["next_topic"] = self.topics[self.current_topic_index]

        self.update_summary_prompt()

        # Most models can't handle large system prompts (we could add it to user prompts; but also this will likely make the models get stuck in writing in a similar "direction" i.e: only positive things happen, etc, pretty standard behavior for models, which tend to get stuck in thought loops, a better method would be to use more agentic & chained set-up)
        # TODO: Parameterize this for use with models that deal well with large context (just Opus I guess)
        context_proportion = self.compute_summary_as_percentage_of_context(self.context_tokens, self.summary_prompt)
        while context_proportion > 0.20:
            self.remove_random_summary_from_summary_prompt()
            context_proportion = self.compute_summary_as_percentage_of_context(self.context_tokens, self.summary_prompt)

        return data_point, self.summary_prompt

    def update_summaries(self, summary):
        self.summaries.append(summary)  

    def update_summary_prompt(self):
        if len(self.summaries) == 0:
            return self.summary_prompt
        elif len(self.summaries) == 1:
            self.summary_prompt = self.summary_prompt.replace(
                "[None yet]",
                self.summaries[-1]
            )
        else:
            self.summary_prompt = self.summary_prompt.replace(
                "</summary-of-previous-data-points>",
                self.summaries[-1] + "\n</summary-of-previous-data-points>"
            )

    

    def remove_random_summary_from_summary_prompt(self):
        # Calculate probabilities using the sigmoid function
        indices = np.arange(len(self.summaries))
        # We apply a sigmoid function to the indices to get a distribution where newer setences are selected more often, keeping more historical context as we clean the context
        probabilities = 1 / (1 + np.exp(-indices / (len(self.summaries) - 1)))
        
        # Normalize probabilities to sum up to 1
        probabilities /= np.sum(probabilities)
        
        # Choose a random summary based on the calculated probabilities
        random_summary = np.random.choice(self.summaries, p=probabilities)
        
        # Remove the random summary from the summary prompt
        self.summary_prompt = self.summary_prompt.replace(random_summary, "")

class DynamicOllama:
    def __init__(self,system_prompt, model_name="eramax/higgs-llama3-70b:iq2xs", base_url="http://localhost:11434", use_tokens=False):
        self.model_name = model_name
        self.base_url = base_url
        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, format="json", keep_alive="10m", num_ctx=124_000, num_predict=4096, temperature=0)
        self.system_template = SystemMessagePromptTemplate.from_template(system_prompt)

    def generate(self, user_prompt, summary_prompt):
        response = self.invoke_single_response(self.llm, user_prompt, summary_prompt, self.system_template)
        validated_dict = self.validate_json_data(response.content)
        return validated_dict

    @staticmethod
    def validate_json_data(json_str, expected_fields=["date", "content", "authors", "summary"], expected_types=[str, (str, dict), list, str]):
        """
        Validates a JSON string to ensure it has the expected fields and types, and the data is formatted correctly.
        If additional fields exist, they are moved to the "content" field, and the previous value
        of the "content" field is moved to an "overview" subfield if it is a string.

        Args:
            json_str (str): A JSON string
            expected_fields (list): A list of expected fields in the JSON data
            expected_types (list): A list of expected types for the fields in the JSON data

        Returns:
            dict or bool: The parsed JSON data if it is valid, False otherwise
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Invalid JSON string")
            return False

        if not isinstance(data, dict):
            print(f"Error: JSON data is not a dictionary\nJSON data: {json.dumps(data, indent=2)}")
            return False

        for field, field_type in zip(expected_fields, expected_types):
            if field not in data:
                print(f"Error: Missing expected field '{field}'\nJSON data: {json.dumps(data, indent=2)}")
                return False
            if not isinstance(data[field], field_type):
                print(f"Error: Field '{field}' has an unexpected type\nExpected type: {field_type}\nActual type: {type(data[field])}\nJSON data: {json.dumps(data, indent=2)}")
                return False

        date = data["date"]
        parts = date.split("-")
        if len(parts) != 3 or not all(len(part) == 4 or len(part) == 2 and part.isdigit() for part in parts):
            print(f"Error: Invalid date format\nDate: {date}\nJSON data: {json.dumps(data, indent=2)}")
            return False

        additional_fields = {k: v for k, v in data.items() if k not in expected_fields}
        if additional_fields:
            if isinstance(data["content"], str):
                data["content"] = {"overview": data["content"]}
            elif not isinstance(data["content"], dict):
                print(f"Error: Field 'content' has an unexpected type\nExpected type: (str, dict)\nActual type: {type(data['content'])}\nJSON data: {json.dumps(data, indent=2)}")
                return False
            data["content"].update(additional_fields)
            for field in additional_fields:
                del data[field]

        return data

    @staticmethod
    def invoke_single_response(llm, user_prompt, summary_prompt, system_template):
        user_template = HumanMessagePromptTemplate.from_template(user_prompt)
        summary_template = HumanMessagePromptTemplate.from_template(summary_prompt)
        user_prompt = ChatPromptTemplate.from_messages([system_template, summary_template, user_template])
        return llm.invoke(user_prompt.format_messages())
    
    @staticmethod
    def save_node(new_node_json, data_point, save_path):
        new_node_json = {
        "date": new_node_json["date"],
        "summary":  new_node_json["summary"],
        "authors": data_point["authors"],
        "content": new_node_json["content"],
        "category": data_point["category"],
        "topic": data_point["topic"],
        "next_topic": data_point["next_topic"],
        "all_summaries": data_point["all_summaries"]
        }
        # Append the generated point to JSON file
        with open(save_path, 'a') as f:
            json.dump(new_node_json, f, indent=4)
            f.write('\n')

# Example usage
if __name__ == "__main__":
    ollama = DynamicOllama()
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "What is the capital of Finland?"
    result = ollama.generate(system_prompt, user_prompt)
    print(result)