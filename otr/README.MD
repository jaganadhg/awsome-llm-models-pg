# OTR: On-The-Topic Rate

OTR is a small Python utility to compute the On-The-Topic Rate (OTR) of ranked search results and to score individual documents for relevance using OpenAI models.
The code is based on the paper Semantic Search Evaluation https://arxiv.org/abs/2410.21549v1 

## Features

- Compute overall OTR: ratio of relevant documents to total.
- Compute OTR@K: ratio of relevant docs in the top-K positions.
- Query an OpenAI model to label documents as relevant (1) or not (0).

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/your-org/otr.git
   cd otr

   Configure the Poetry environment

    ```bash
    poetry config virtualenvs.in-project true
    poetry install
    ```
   Create a .env in src/ with your OpenAI key:

    ```bash
    OPENAI_API_KEY=sk-…
    ```

    ```bash
    poetry run python otr.py
    ```

    Python example

    ```python
    from loguru import logger
    from otr import on_topic_rate_at_k, get_relevance_for_items

    items = [
        {"qid": "1", "query": "What is the capital of France?", "docuement": "The capital of France is Paris."},
        {"qid": "2", "query": "What is the capital of Germany?", "docuement": "Berlin is the capital of Germany."},
        {"qid": "3", "query": "What is the capital of Italy?", "docuement": "Rome is the capital of Italy."},
        {"qid": "4", "query": "What is the capital of Spain?", "docuement": "Madrid is the capital of Spain."},
        {"qid": "5", "query": "What is the capital of Portugal?", "docuement": "Lisbon is the capital of Portugal."},
        {"qid": "6", "query": "What is the capital of Netherlands?", "docuement": "Amsterdam is the capital of Netherlands."},
        {"qid": "7", "query": "What is the capital of Belgium?", "docuement": "Brussels is the capital of Belgium."},
        {"qid": "8", "query": "What is the capital of Switzerland?", "docuement": "Bern is the capital of Switzerland."},
        {"qid": "9", "query": "What is the capital of Austria?", "docuement": "Vienna is the capital of Austria."},
        {"qid": "10", "query": "What is the capital of Greece?", "docuement": "Athens is the capital of Greece."}
    ]
    results = get_relevance_for_items(items)
    for result in results:
        logger.info(f"Query: {result['query']}, Document: {result['docuement']}, Relevance: {result['relevance']}")
    topic_r = on_topic_rate_at_k([item['relevance'] for item in results], 5)
    logger.info(f"On Topic Rate at K=5: {topic_r}")
    ```

### Citation
If you use OTR in your research or projects, please cite:

    APA

    Gopinadhan, J. (2025). OTR: On-The-Topic Rate [Computer software]. GitHub. https://github.com/jaganadhg/awsome-llm-models-pg/tree/main/otr 

    BibTex

    ```
    @misc{gopinadhan2025otr,
    author       = {Gopinadhan, Jaganadh},
    title        = {{OTR: On\,-The\,-Topic Rate}},
    year         = {2025},
    howpublished = {\url{https://github.com/jaganadhg/awsome-llm-models-pg/tree/main/otr }},
    }
    ```