# Secure LLM Graph of Knowledge (GoK)

This project demonstrates the implementation of a secure LLM (Large Language Model) Graph of Knowledge (GoK) system. The system allows users to query a knowledge graph based on their access level and topic permissions. It also provides an API endpoint for adding new knowledge to the graph.

## Project Structure

The project is structured as follows:

```
├── Dockerfile
├── PacemakerInnovationsData
│   ├── accessLevel.json
│   ├── authentication.json
│   ├── categoryIndex.json
│   ├── collaborativeHirarchy.json
│   ├── dataPrompts.json
│   ├── generated_data_points_v0.json
│   ├── generated_data_points_v1.json
│   ├── generated_data_points_v2.json
│   ├── graph.pt
│   ├── quarterIndex.json
│   ├── redactIndex.json
│   ├── systemPrompt.json
│   ├── test_node_ingestion.json
│   └── topicIndex.json
├── add_node.py
├── app.py
├── confidentialgraph.py
├── environment.yaml
├── generate_graph.py
├── generateanswer.py
├── generatedata.py
├── main.py
├── pdf_to_markdown.py
├── supervisord.conf
└── work-book.ipynb
```

- `Dockerfile`: Contains the instructions to build the Docker image for the application.
- `PacemakerInnovationsData`: Directory containing JSON files with data for the knowledge graph.
- `add_node.py`: Script to add a new node to the knowledge graph.
- `app.py`: The main FastAPI application file.
- `confidentialgraph.py`: Script to generate confidential subgraphs based on user access level.
- `environment.yaml`: Conda environment configuration file.
- `generate_graph.py`: Script to generate the knowledge graph from JSON data.
- `generateanswer.py`: Script to generate answers based on user queries.
- `generatedata.py`: Script to generate data points for the knowledge graph.
- `main.py`: Entry point for the FastAPI application.
- `pdf_to_markdown.py`: Script to convert PDF files to Markdown format.
- `supervisord.conf`: Supervisor configuration file for running the application.
- `work-book.ipynb`: Jupyter Notebook containing the project documentation and workflow.

## Getting Started

To run the application, follow these steps:

1. Build the Docker image:
   ```bash
   docker build -t secure-llm-gok .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 -p 8000:8000 secure-llm-gok
   ```

   This will start the FastAPI application and make it accessible on `http://localhost:8000`.

## API Endpoints

The application provides the following API endpoints:

### Query Endpoint

- URL: `/query`
- Method: POST
- Request Body:
  - `user_token` (string): User authentication token.
  - `query` (string): User query.
- Response:
  - `answer` (string): Generated answer based on the user query and access level.

This endpoint allows users to query the knowledge graph based on their access level and topic permissions. It generates a confidential subgraph for the user and uses the `QueryProcessor` to generate an answer.

### Add Knowledge Endpoint

- URL: `/add_knowledge`
- Method: POST
- Request Body:
  - `user_token` (string): User authentication token.
  - `knowledge` (JSON): Knowledge data to be added to the graph.
- Response:
  - `message` (string): Success message indicating the knowledge has been added.

This endpoint allows users to add new knowledge to the graph. The knowledge data is filtered based on the topics that the user has access to.

## Deployment

The application is containerized using Docker for easy deployment. The provided `Dockerfile` contains the necessary instructions to build the Docker image.

To run the application, use the following commands:

```bash
docker build -t secure-llm-gok .
docker run -p 8501:8501 -p 8000:8000 secure-llm-gok
```

This will build the Docker image and run the container, making the application accessible on `http://localhost:8000`.

For more details on the project workflow and implementation, please refer to the `work-book.ipynb` Jupyter Notebook.