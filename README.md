Next word prediction

python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

### Build and Run the Project

To build and start the services locally, run the following commands:

```bash
# Build the containers
docker compose -f container/docker-compose.yml build

# Start the API
docker compose -f container/docker-compose.yml up api

# Alternatively, start all services in detached mode
docker compose -f container/docker-compose.yml up -d
```