#!/usr/bin/env python

from dotenv import load_dotenv
import os
import openai


load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


def on_topic_rate(rel_count: int, total_docs: int) -> float:
    """
    Calculate the On-The-Topic Rate (OTR) for a given set of documents.

    Args:
    rel_count (int): The number of relevant documents.
    total_docs (int): The total number of documents.

    Returns:
    float: The OTR value, which is the ratio of relevant documents to total documents.
    """
    if total_docs == 0:
        return 0.0
    return rel_count / total_docs


def on_topic_rate_at_k(relevance_flags: list[int], k: int) -> float:
    """
    Calculate On Topic Rate at cutoff K (OTR@K) using integer flags.

    OTR@K = (sum of flags in top-K) / K

    Args:
        relevance_flags (list[int]): List of 0 (not relevant) or 1 (relevant).
        k (int): Cutoff rank K.

    Returns:
        float: OTR@K (0.0 if k <= 0).
    """
    if k <= 0:
        return 0.0

    top_k = relevance_flags[:k]
    relevant_count = sum(top_k) 
    return relevant_count / k


def get_relevance_for_items(
    items: list[dict],
    model: str = "gpt-4"
) -> list[dict]:
    """
    For each item in items (with keys "qid", "query", "docuement"),
    call OpenAI to get {"relevance":0 or 1} and return a new list
    of dicts with that field added.
    """
    results = []
    for item in items:
        prompt = (
            f"""
            Task: Evaluate Relevance of Search Result 
            Definition of On-Topic Rate: The on-topic rate measures the percentage of search results that are relevant to the intended topic. A document is considered "on-topic" if it is primarily about the query or strongly relevant to the query. 
            The question is \"{item['query']}\"
            Instructions:
            1. Focus on Semantic Matching: Do not rely solely on keyword matching. Carefully consider the user's intent behind the query and whether the document truly addresses that intent.
            2. Thoroughly Analyze Document Content: Take into account all provided information about the document, including the title, body, and any additional context. 
            3. Provide Detailed Reasoning: Explain your reasoning for the relevance decision, highlighting specific aspects of the document that support your judgment.
            Query: \"{item['query']}\"
            Document: \"{item['docuement']}\"
            Response Format:
            respond with a JSON object 
            {{"relevance": <0 or 1>}} where 1 means relevant and 0 not relevant.
            """
        )
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            relevance = int(data.get("relevance", 0))
        except (ValueError, json.JSONDecodeError):
            relevance = 0
        results.append({**item, "relevance": relevance})
    return results