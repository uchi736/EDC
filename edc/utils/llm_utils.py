import os
import openai
from openai import AzureOpenAI, OpenAI
import time
import ast
from typing import List
import logging
from pathlib import Path

# Load .env file
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    # Try to find .env file in current directory or parent directories
    current_dir = Path(__file__).parent
    for _ in range(5):  # Search up to 5 levels
        env_path = current_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
        current_dir = current_dir.parent

    # Also try working directory
    if Path(".env").exists():
        load_dotenv(".env")

# Load .env on module import
load_env()

logger = logging.getLogger(__name__)

# Azure OpenAI client cache
_azure_client = None
_openai_client = None


def get_azure_client():
    """Get or create Azure OpenAI client"""
    global _azure_client
    if _azure_client is None:
        _azure_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
    return _azure_client


def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    return _openai_client


def free_model(model=None, tokenizer=None):
    """Free model from memory (only for HuggingFace models)"""
    try:
        import gc
        import torch
        if model is not None:
            model.cpu()
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    """Get embedding using E5-Mistral model (requires HuggingFace)"""
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model, text: str, prompt_name=None, prompt=None):
    """Get embedding using SentenceTransformer"""
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    """Check if model is OpenAI or Azure OpenAI"""
    return "gpt" in model_name or is_model_azure(model_name)


def is_model_azure(model_name):
    """Check if model should use Azure OpenAI"""
    return model_name.startswith("azure/") or model_name == "azure"


def get_azure_chat_deployment_name():
    """Get Azure chat deployment name from environment"""
    return os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4")


def get_azure_embedding_deployment_name():
    """Get Azure embedding deployment name from environment"""
    return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")


def get_azure_deployment_name(model_name):
    """Extract deployment name from azure/deployment_name format or use env default"""
    if model_name == "azure":
        # Use default from .env
        return get_azure_chat_deployment_name()
    if model_name.startswith("azure/"):
        return model_name[6:]  # Remove "azure/" prefix
    return model_name


def get_azure_embedding(texts: List[str]) -> List[List[float]]:
    """Get embeddings using Azure OpenAI embedding model"""
    client = get_azure_client()
    deployment_name = get_azure_embedding_deployment_name()

    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(
        model=deployment_name,
        input=texts
    )

    return [item.embedding for item in response.data]


class AzureEmbedder:
    """
    Azure OpenAI Embedding wrapper that mimics SentenceTransformer interface.
    Use this as a drop-in replacement for SentenceTransformer when using Azure.
    """

    def __init__(self):
        self.prompts = {}  # For compatibility with SentenceTransformer

    def encode(self, text: str, prompt_name: str = None, prompt: str = None) -> List[float]:
        """
        Encode text to embedding vector.
        prompt_name and prompt are ignored (for SentenceTransformer compatibility)
        """
        embeddings = get_azure_embedding(text)
        return embeddings[0] if isinstance(text, str) else embeddings


def is_embedder_azure(embedder_name: str) -> bool:
    """Check if embedder should use Azure OpenAI"""
    return embedder_name == "azure" or embedder_name.startswith("azure/")


def get_embedder(embedder_name: str):
    """
    Get embedder instance based on name.
    Returns AzureEmbedder for "azure" or SentenceTransformer for others.
    """
    if is_embedder_azure(embedder_name):
        logger.info("Using Azure OpenAI Embedder")
        return AzureEmbedder()
    else:
        # Lazy import SentenceTransformer only when needed
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {embedder_name}")
        return SentenceTransformer(embedder_name, trust_remote_code=True)


def generate_completion_transformers(
    input: list,
    model,
    tokenizer,
    max_new_token=256,
    answer_prepend="",
):
    """Generate completion using HuggingFace transformers (requires local model)"""
    # Lazy import
    from transformers import GenerationConfig

    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    """
    Chat completion supporting both OpenAI and Azure OpenAI.

    For Azure OpenAI, use model name format: "azure/your-deployment-name"
    """
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    while response is None:
        try:
            if is_model_azure(model):
                # Azure OpenAI
                client = get_azure_client()
                deployment_name = get_azure_deployment_name(model)
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Standard OpenAI
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        except Exception as e:
            logger.warning(f"API call failed: {e}, retrying in 5 seconds...")
            time.sleep(5)

    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
