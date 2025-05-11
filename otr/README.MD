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

    ```md
    OPENAI_API_KEY=sk-…
    ```

    ```bash
    poetry run python otr.py
    ```

    Python example

    ```
    from otr import on_topic_rate_at_k, get_relevance_for_items

    flags = [1,0,1,1,0]
    print(on_topic_rate_at_k(flags, 3))  # 0.666…

    items = [
    {"qid":"1","query":"Capitals of France?","docuement":"Paris is…"}
    ]
    results = get_relevance_for_items(items, model="o4-mini")
    print(results)
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