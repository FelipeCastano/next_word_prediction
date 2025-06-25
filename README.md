# Next Word Prediction API

This project demonstrates a deployment of a Next Word Prediction model based on n-grams, developed during a Coursera NLP course workshop. The main goal is to showcase deployment best practices including containerization with Docker, orchestration with Docker Compose, and API development with FastAPI.

## Overview

- **Model:** N-grams based Next Word Prediction  
- **Languages Supported:** English and Spanish (custom extension)  
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)  
- **Frontend UI:** [Streamlit](https://streamlit.io/)  
- **Containerization:** [Docker](https://www.docker.com/)  
- **Orchestration:** [Docker Compose](https://docs.docker.com/compose/)  

The API provides a simple interface to predict the next word given an input text, optionally filtered by a starting prefix. Spanish support has been added as a custom extension.

## Features

- Fully containerized for easy setup and deployment  
- FastAPI-powered RESTful API  
- Lightweight frontend interface built with Streamlit  
- Supports English and Spanish next word prediction  
- Ideal for educational and demonstration purposes  

## Getting Started

### Prerequisites

Make sure Docker and Docker Compose are installed on your machine:

- [Install Docker](https://docs.docker.com/get-docker/)  
- [Install Docker Compose](https://docs.docker.com/compose/install/)  

### Build and Run the Project

To build and start the services locally, run:

```bash
# Build the containers
docker compose -f container/docker-compose.yml build

# Start the API in foreground
docker compose -f container/docker-compose.yml up api

# Alternatively, start all services in detached mode
docker compose -f container/docker-compose.yml up -d
```

Access the API
API Base URL: [http://localhost:8000]

Interactive API docs: [http://localhost:8000/docs]

Access the Frontend
A simple user interface is available via Streamlit:

Frontend URL: [http://localhost:8501]

Example API Request
To get the next word prediction, send a POST request to /get_next_word with the following JSON payload:
```json
{
    "text": "input_text",
    "lang": 1,          # 1 for English, 2 for Spanish
    "starts_with": ""   # optional string to filter suggestions by prefix
}
```
Example API Response
The API returns a JSON with three predicted next words, each with its probability:

```json
{
  "1": {"word": "example1", "prob": 0.35},
  "2": {"word": "example2", "prob": 0.25},
  "3": {"word": "example3", "prob": 0.15}
}
```

Important Note on Training Data Size
In the file src/back/main.py at lines 24 and 45, there is a line:

```python
train_data = load_data(train_path, 800000)
```
The second parameter controls the number of training samples used for predictions. You can reduce this number if your hardware has limited resources.