from datasets import load_dataset, concatenate_datasets, Features, Sequence, Value, DatasetDict
import pandas as pd
import re
from transformers import AutoTokenizer
import json
import numpy as np
import os
import shutil
import warnings
import argparse
from typing import Dict, List, Optional, Union
from datetime import datetime
import multiprocessing as mp
from functools import partial
import gc
import psutil
import tempfile
from contextlib import contextmanager
from datasets import Dataset

# Global tokenizer for multiprocessing (kept for compatibility but not used in Colab mode)
_tokenizer = None


def init_worker(tokenizer_name: str):
    """Initialize tokenizer in worker process (for non-Colab environments)."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def detect_environment():
    """Detect if running in Colab or similar environment."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass

    # Check for other notebook environments
    try:
        import ipykernel
        return "jupyter"
    except ImportError:
        pass

    return "standard"


@contextmanager
def temporary_directory(use_hf_cache=True):
    """Context manager for temporary directory that gets cleaned up."""
    if use_hf_cache:
        # Use Hugging Face cache directory
        try:
            # Try multiple ways to get the HF cache directory
            hf_cache_dir = None

            # Method 1: Use HF_DATASETS_CACHE environment variable
            if 'HF_DATASETS_CACHE' in os.environ:
                hf_cache_dir = os.environ['HF_DATASETS_CACHE']

            # Method 2: Use the datasets config
            if hf_cache_dir is None:
                try:
                    from datasets.config import HF_DATASETS_CACHE
                    hf_cache_dir = HF_DATASETS_CACHE
                except ImportError:
                    pass

            # Method 3: Use huggingface_hub cache directory
            if hf_cache_dir is None:
                try:
                    from huggingface_hub import HF_HUB_CACHE
                    hf_cache_dir = os.path.dirname(HF_HUB_CACHE)  # Go one level up from hub cache
                except ImportError:
                    pass

            # Method 4: Default HF cache location
            if hf_cache_dir is None:
                home_dir = os.path.expanduser("~")
                hf_cache_dir = os.path.join(home_dir, ".cache", "huggingface", "datasets")

            # Create the temp directory within HF cache
            temp_dir = os.path.join(hf_cache_dir, f"dataset_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"Using HF cache directory: {hf_cache_dir}")

        except Exception as e:
            print(f"Failed to use HF cache directory: {e}, falling back to system temp")
            temp_dir = tempfile.mkdtemp(prefix="dataset_merge_")
    else:
        temp_dir = tempfile.mkdtemp(prefix="dataset_merge_")

    try:
        yield temp_dir
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            warnings.warn(f"Failed to clean up temporary directory {temp_dir}: {e}")


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_limit(max_memory_gb=32):
    """Check if memory usage exceeds limit and force garbage collection if needed."""
    current_memory_gb = get_memory_usage() / 1024
    if current_memory_gb > max_memory_gb * 0.8:  # Warn at 80% of limit
        print(f"Warning: Memory usage ({current_memory_gb:.1f}GB) approaching limit ({max_memory_gb}GB)")
        gc.collect()
        current_memory_gb = get_memory_usage() / 1024

    if current_memory_gb > max_memory_gb:
        print(f"Error: Memory usage ({current_memory_gb:.1f}GB) exceeds limit ({max_memory_gb}GB)")
        raise MemoryError(f"Memory usage exceeded {max_memory_gb}GB limit")

    return current_memory_gb


def log_memory(message: str, max_memory_gb=32):
    """Log memory usage with message and check limits."""
    current_memory_gb = check_memory_limit(max_memory_gb)
    print(f"{message} - Memory usage: {current_memory_gb:.1f}GB / {max_memory_gb}GB limit")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_and_fix_messages(messages):
    """
    Validate and fix message format to ensure it's a proper list of dicts.
    Returns a list of valid message dictionaries.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        print(f"Warning: messages field is not a list: {type(messages)}")
        return []

    validated_messages = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            print(f"Warning: message {i} is not a dict: {type(msg)}")
            continue

        if "role" not in msg or "content" not in msg:
            print(f"Warning: message {i} missing required fields: {msg}")
            continue

        # Ensure role and content are strings
        validated_messages.append({
            "role": str(msg["role"]).strip(),
            "content": str(msg["content"]).strip()
        })

    return validated_messages


def process_example_single(example, tokenizer, min_tokens: Optional[int] = None, max_tokens: Optional[int] = None):
    """
    Process a single example with tokenization and filtering.
    Returns None if filtered out, otherwise returns processed example.
    """
    chat_messages = example.get("message", [])

    # Validate and fix message format
    chat_messages = validate_and_fix_messages(chat_messages)

    try:
        if not chat_messages:
            formatted_text = ""
            encoded_text = []
        else:
            formatted_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            encoded_text = tokenizer.encode(formatted_text)
    except Exception as e:
        print(f"Error processing example from {example.get('source_dataset', 'unknown')}: {e}")
        formatted_text = ""
        encoded_text = []

    token_count = len(encoded_text)

    # Apply token filtering in one go
    if min_tokens is not None and token_count < min_tokens:
        return None
    if max_tokens is not None and token_count > max_tokens:
        return None

    processed_example = {
        "task": str(example.get("task", "")),
        "input": str(example.get("input", "")),
        "output": str(example.get("output", "")),
        "message": chat_messages,  # This is now guaranteed to be a list of dicts
        "formatted_text": formatted_text,
        "encoded_text": encoded_text,
        "total_token_count": token_count,
        "message_length": len(formatted_text),
        "num_turns": len(chat_messages),
        "source_dataset": str(example.get("source_dataset", "unknown")),
        "has_existing_messages": bool(example.get("has_existing_messages", False))
    }

    return processed_example


def process_examples_threaded(examples, tokenizer, min_tokens: Optional[int] = None, max_tokens: Optional[int] = None,
                              num_workers: int = 4):
    """
    Process examples using threading instead of multiprocessing to avoid Colab issues.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = []
    filtered_count = 0

    def process_chunk(chunk):
        chunk_results = []
        chunk_filtered = 0
        for example in chunk:
            result = process_example_single(example, tokenizer, min_tokens, max_tokens)
            if result is not None:
                chunk_results.append(result)
            else:
                chunk_filtered += 1
        return chunk_results, chunk_filtered

    # Split into chunks for better progress tracking
    chunk_size = max(len(examples) // (num_workers * 4), 100)
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]

    print(f"Processing {len(examples)} examples in {len(chunks)} chunks using {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

        # Process completed chunks
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results, chunk_filtered = future.result()
                results.extend(chunk_results)
                filtered_count += chunk_filtered

                if (chunk_idx + 1) % 10 == 0 or chunk_idx == len(chunks) - 1:
                    print(
                        f"Processed {chunk_idx + 1}/{len(chunks)} chunks, {len(results)} examples kept, {filtered_count} filtered")

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")

    return results, filtered_count


def process_example_batch_mp(examples, tokenizer_name: str, min_tokens: Optional[int] = None,
                             max_tokens: Optional[int] = None):
    """
    Process a batch of examples with tokenization and filtering for multiprocessing.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    results = []

    for example in examples:
        result = process_example_single(example, _tokenizer, min_tokens, max_tokens)
        if result is not None:
            results.append(result)

    return results


def save_processed_dataset_to_cache(processed_examples: List[Dict], dataset_id: str, cache_dir: str):
    """Save processed dataset to cache directory as a Hugging Face dataset."""
    if not processed_examples:
        print(f"Warning: No processed examples for {dataset_id}")
        return None

    # Create features schema
    features = Features({
        'task': Value('string'),
        'input': Value('string'),
        'output': Value('string'),
        'message': Sequence({
            'role': Value('string'),
            'content': Value('string')
        }),
        'formatted_text': Value('string'),
        'encoded_text': Sequence(Value('int32')),
        'total_token_count': Value('int64'),
        'message_length': Value('int64'),
        'num_turns': Value('int64'),
        'source_dataset': Value('string'),
        'has_existing_messages': Value('bool')
    })

    # Create dataset from processed examples
    dataset = Dataset.from_list(processed_examples, features=features)

    # Save to cache directory with safe filename
    safe_dataset_name = dataset_id.replace('/', '_').replace('-', '_').replace('.', '_')
    cache_path = os.path.join(cache_dir, f"processed_{safe_dataset_name}")

    print(f"Saving {len(processed_examples)} processed examples to cache: {cache_path}")
    dataset.save_to_disk(cache_path)

    return cache_path


def load_processed_dataset_from_cache(cache_path: str):
    """Load processed dataset from cache directory."""
    if not os.path.exists(cache_path):
        print(f"Warning: Cache path does not exist: {cache_path}")
        return None

    print(f"Loading processed dataset from cache: {cache_path}")
    dataset = Dataset.load_from_disk(cache_path)
    print(f"Loaded {len(dataset)} examples from cache")
    return dataset


def process_single_dataset(dataset_id: str, config: Dict, tokenizer, tokenizer_name: str,
                           cache_dir: str, max_memory_gb: int = 32):
    """
    Process a single dataset completely and save to cache.
    Returns the cache path or None if processing failed.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_id}")
    print(f"{'=' * 60}")

    log_memory(f"Starting {dataset_id}", max_memory_gb)

    # Extract parameters from config
    split = config["datasets"]["split"]
    field_mapping = config["datasets"]["field_mapping"]
    task_mapping = config["datasets"]["task_mapping"]
    datasets_with_messages = config["datasets"].get("datasets_with_messages", [])
    messages_field_name = config["datasets"].get("messages_field_name", "messages")
    preserve_existing_fields = config["datasets"].get("preserve_existing_fields", True)

    # Processing parameters
    max_records_per_dataset = config["processing"]["max_records_per_dataset"]
    max_tokens_per_record = config["processing"]["max_tokens_per_record"]
    min_tokens_per_record = config["processing"]["min_tokens_per_record"]
    seed = config["processing"]["seed"]

    # Performance parameters
    batch_size = config["processing"].get("batch_size", 1000)
    num_workers = config["processing"].get("num_workers", min(mp.cpu_count(), 8))
    force_threading = config["processing"].get("force_threading", False)

    # Adjust parameters based on memory limit
    if max_memory_gb <= 16:
        batch_size = min(batch_size, 500)
        num_workers = min(num_workers, 2)
    elif max_memory_gb <= 32:
        batch_size = min(batch_size, 1000)
        num_workers = min(num_workers, 4)

    # Detect environment and adjust processing method
    env = detect_environment()
    use_threading = force_threading or env in ["colab", "jupyter"]

    if use_threading:
        num_workers = min(num_workers, 4)

    try:
        # Load the dataset
        log_memory(f"Loading {dataset_id}", max_memory_gb)
        ds = load_dataset(dataset_id, split=split)
        print(f"Loaded {dataset_id} with {len(ds)} examples")
        print(f"  Original fields: {list(ds.features.keys())}")

        # Shuffle before limiting (to get a random sample)
        ds = ds.shuffle(seed=seed)

        # Limit number of records if specified
        if max_records_per_dataset is not None and max_records_per_dataset < len(ds):
            ds = ds.select(range(max_records_per_dataset))
            print(f"  Limited to {max_records_per_dataset} examples")

        # Apply field mapping if provided
        if dataset_id in field_mapping:
            mapping = field_mapping[dataset_id]
            safe_mapping = {}
            existing_columns = set(ds.features.keys())

            for old, new in mapping.items():
                if old not in ds.features:
                    continue

                if new == "task" and preserve_existing_fields and "task" in existing_columns:
                    print(f"  Preserving existing 'task' field, skipping mapping from '{old}'")
                    continue
                elif new == "source_dataset" and preserve_existing_fields and "source_dataset" in existing_columns:
                    print(f"  Preserving existing 'source_dataset' field, skipping mapping from '{old}'")
                    continue
                elif new in existing_columns and old != new:
                    temp_name = f"{new}_{dataset_id.replace('/', '_').replace('-', '_')}"
                    safe_mapping[old] = temp_name
                    warnings.warn(f"Column '{new}' already exists in {dataset_id}, renaming to '{temp_name}'")
                else:
                    safe_mapping[old] = new

            if safe_mapping:
                ds = ds.rename_columns(safe_mapping)
                print(f"  Applied field mapping: {safe_mapping}")

        # Add task description
        if "task" not in ds.features:
            task_desc = task_mapping.get(dataset_id, "Generic task")
            ds = ds.add_column("task", [task_desc] * len(ds))
            print(f"  Added task column with value: '{task_desc}'")
        elif preserve_existing_fields:
            print(f"  Preserving existing 'task' field")
        else:
            task_desc = task_mapping.get(dataset_id, "Generic task")
            ds = ds.remove_columns(["task"])
            ds = ds.add_column("task", [task_desc] * len(ds))
            print(f"  Replaced task column with value: '{task_desc}'")

        # Handle source_dataset field
        if "source_dataset" not in ds.features:
            ds = ds.add_column("source_dataset", [dataset_id] * len(ds))
            print(f"  Added source_dataset column")
        elif preserve_existing_fields:
            print(f"  Preserving existing 'source_dataset' field")
        else:
            ds = ds.remove_columns(["source_dataset"])
            ds = ds.add_column("source_dataset", [dataset_id] * len(ds))
            print(f"  Replaced source_dataset column")

        # Check if this dataset already has messages field
        has_messages = dataset_id in datasets_with_messages

        if has_messages:
            if messages_field_name not in ds.features:
                warnings.warn(
                    f"Dataset {dataset_id} is marked as having messages field '{messages_field_name}', but field not found. Treating as regular dataset.")
                has_messages = False
            else:
                print(f"  Dataset {dataset_id} already has messages field: {messages_field_name}")
                if messages_field_name != "message":
                    ds = ds.rename_column(messages_field_name, "message")
                if "input" not in ds.features:
                    ds = ds.add_column("input", [""] * len(ds))
                if "output" not in ds.features:
                    ds = ds.add_column("output", [""] * len(ds))
        else:
            required_fields = {"input", "output"}
            available_fields = set(ds.features.keys())
            missing_fields = required_fields - available_fields
            if missing_fields:
                warnings.warn(f"Dataset {dataset_id} is missing required fields: {missing_fields}")
                return None

        # Add flag for pre-existing messages
        ds = ds.add_column("has_existing_messages", [has_messages] * len(ds))

        # Add empty message column if not present
        if "message" not in ds.features:
            print(f"Adding empty message column to dataset")
            ds = ds.add_column("message", [[] for _ in range(len(ds))])

        print(f"  Final fields: {list(ds.features.keys())}")

        log_memory(f"After loading and preprocessing {dataset_id}", max_memory_gb)

        # Convert to unified chat format
        def convert_to_unified_chat_format(example):
            if example.get("has_existing_messages", False) and "message" in example:
                messages = example["message"]
                # Validate and fix message format
                messages = validate_and_fix_messages(messages)
            else:
                messages = []

                task_content = example.get("task", "").strip()
                if task_content and task_content != "Generic task":
                    messages.append({
                        "role": "system",
                        "content": task_content
                    })

                if "input" in example and example["input"]:
                    messages.append({
                        "role": "user",
                        "content": str(example["input"])
                    })

                if "output" in example and example["output"]:
                    messages.append({
                        "role": "assistant",
                        "content": str(example["output"])
                    })

            return {
                "task": str(example.get("task", "")),
                "input": str(example.get("input", "")),
                "output": str(example.get("output", "")),
                "message": messages,
                "source_dataset": str(example.get("source_dataset", "unknown")),
                "has_existing_messages": bool(example.get("has_existing_messages", False))
            }

        log_memory(f"Before chat format conversion {dataset_id}", max_memory_gb)

        # Apply the conversion with batching to manage memory
        conversion_batch_size = min(batch_size, 500) if max_memory_gb <= 32 else batch_size

        ds = ds.map(
            convert_to_unified_chat_format,
            remove_columns=ds.column_names,
            batch_size=conversion_batch_size,
            num_proc=min(num_workers, 2) if max_memory_gb <= 32 else num_workers
        )

        log_memory(f"After chat format conversion {dataset_id}", max_memory_gb)

        # Convert to list for processing
        examples_list = list(ds)
        original_count = len(examples_list)
        del ds  # Free memory
        gc.collect()

        log_memory(f"Before tokenization and filtering {dataset_id}", max_memory_gb)

        # Process with appropriate method based on environment
        if use_threading:
            print(f"Processing examples with {num_workers} threads (environment-optimized)...")
            processed_examples, total_filtered = process_examples_threaded(
                examples_list,
                tokenizer,
                min_tokens_per_record,
                max_tokens_per_record,
                num_workers
            )
        else:
            print(f"Processing examples with {num_workers} processes...")

            # Split examples into chunks for parallel processing
            chunk_size = max(batch_size // num_workers, 1)
            chunks = [examples_list[i:i + chunk_size] for i in range(0, len(examples_list), chunk_size)]

            # Process chunks in parallel with multiprocessing
            with mp.Pool(num_workers, initializer=init_worker, initargs=(tokenizer_name,)) as pool:
                process_func = partial(
                    process_example_batch_mp,
                    tokenizer_name=tokenizer_name,
                    min_tokens=min_tokens_per_record,
                    max_tokens=max_tokens_per_record
                )

                processed_chunks = pool.map(process_func, chunks)

            # Flatten results
            processed_examples = []
            for chunk_results in processed_chunks:
                processed_examples.extend(chunk_results)

            total_filtered = original_count - len(processed_examples)

        del examples_list  # Free memory
        gc.collect()

        print(f"Processed {dataset_id}: kept {len(processed_examples)} examples, "
              f"filtered {total_filtered} ({total_filtered / original_count:.2%})")

        log_memory(f"After processing {dataset_id}", max_memory_gb)

        # Save processed dataset to cache
        cache_path = save_processed_dataset_to_cache(processed_examples, dataset_id, cache_dir)

        # Clear processed examples from memory
        del processed_examples
        gc.collect()

        log_memory(f"After caching {dataset_id}", max_memory_gb)

        return cache_path

    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {e}")
        return None


def merge_cached_datasets(cache_paths: List[str], max_memory_gb: int = 32):
    """
    Load and merge all cached processed datasets.
    """
    print(f"\n{'=' * 60}")
    print("Merging cached processed datasets")
    print(f"{'=' * 60}")

    log_memory("Starting merge from cache", max_memory_gb)

    cached_datasets = []

    for cache_path in cache_paths:
        if cache_path and os.path.exists(cache_path):
            dataset = load_processed_dataset_from_cache(cache_path)
            if dataset is not None:
                cached_datasets.append(dataset)
                log_memory(f"Loaded dataset from {os.path.basename(cache_path)}", max_memory_gb)
        else:
            print(f"Warning: Cache path not found or invalid: {cache_path}")

    if not cached_datasets:
        raise ValueError("No valid cached datasets found")

    print(f"Merging {len(cached_datasets)} datasets...")

    # Concatenate all cached datasets
    merged_dataset = concatenate_datasets(cached_datasets)

    # Clear cached datasets from memory
    del cached_datasets
    gc.collect()

    log_memory("After merging cached datasets", max_memory_gb)

    print(f"Merged dataset has {len(merged_dataset)} examples")

    return merged_dataset


# def create_dataset_card(merged_dataset, config: Dict, output_path: str):
#     """Create and save a dataset card with metadata."""
#     # Get dataset statistics
#     token_counts = np.array(merged_dataset["total_token_count"])
#     source_datasets = set(merged_dataset["source_dataset"])
#
#     # Count by source dataset
#     dataset_counts = {}
#     for dataset in source_datasets:
#         dataset_counts[dataset] = sum(1 for ex in merged_dataset if ex["source_dataset"] == dataset)
#
#     # Create dataset card content
#     card_content = f"""---
# dataset_info:
#   features:
#   - name: task
#     dtype: string
#   - name: input
#     dtype: string
#   - name: output
#     dtype: string
#   - name: message
#     sequence:
#     - name: role
#       dtype: string
#     - name: content
#       dtype: string
#   - name: formatted_text
#     dtype: string
#   - name: encoded_text
#     sequence: int32
#   - name: message_length
#     dtype: int64
#   - name: total_token_count
#     dtype: int64
#   - name: source_dataset
#     dtype: string
#   - name: has_existing_messages
#     dtype: bool
#   - name: num_turns
#     dtype: int64
#   configs:
#   - config_name: default
#     data_files:
#     - split: train
#       path: data/train-*
#     - split: test
#       path: data/test-*
# license: mit
# task_categories:
# - text-generation
# - conversational
# language:
# - en
# tags:
# - synthetic
# - chat
# - instruction-following
# size_categories:
# - {get_size_category(len(merged_dataset))}
# ---
#
# # Merged Chat Dataset
#
# ## Dataset Description
#
# This dataset is a merged collection of multiple instruction-following and conversational datasets, formatted for supervised fine-tuning (SFT) of language models.
#
# **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# ## Dataset Statistics
#
# - **Total Examples:** {len(merged_dataset):,}
# - **Total Tokens:** {int(np.sum(token_counts))}
#
# - **Token Count Statistics:**
#   - Min: {int(token_counts.min())}
#   - Max: {int(token_counts.max())}
#   - Mean: {int(token_counts.mean())}
#   - Median: {int(np.median(token_counts))}
#
# ## Source Datasets
#
# This merged dataset includes examples from the following sources:
#
# """
#
#     for dataset, count in sorted(dataset_counts.items()):
#         percentage = (count / len(merged_dataset)) * 100
#         card_content += f"- **{dataset}**: {count:,} examples ({percentage:.1f}%)\n"
#
#     card_content += f"""
# ## Token Distribution
#
# """
#
#     # Add token threshold statistics
#     if "analysis" in config and "token_thresholds" in config["analysis"]:
#         for threshold in config["analysis"]["token_thresholds"]:
#             count = (token_counts > threshold).sum()
#             percentage = (count / len(token_counts)) * 100
#             card_content += f"- Examples > {threshold} tokens: {count:,} ({percentage:.1f}%)\n"
#
#     card_content += f"""
# ## Data Format
#
# Each example contains:
# - `task`: Task description or category
# - `input`: User input/question
# - `output`: Expected response
# - `message`: List of chat messages with roles (system/user/assistant)
# - `formatted_text`: Tokenizer-formatted text
# - `total_token_count`: Number of tokens in the formatted text
# - `source_dataset`: Original dataset identifier
# - `has_existing_messages`: Whether the example had pre-existing chat format
#
# ## Usage
#
# ```python
# from datasets import load_dataset
#
# dataset = load_dataset("path/to/this/dataset")
# train_data = dataset["train"]
# test_data = dataset["test"]
#
# # Access chat messages
# for example in train_data:
#     messages = example["message"]  # This is a list
#     for message in messages:
#         print(f"{{message['role']}}: {{message['content']}}")
# ```
#
# ## Configuration Used
#
# The dataset was created with the following configuration:
#
# ```json
# {json.dumps(config, indent=2)}
# ```
#
# ## Citation
#
# If you use this dataset, please cite the original source datasets appropriately.
# """
#
#     # Save the dataset card
#     card_path = os.path.join(output_path, "README.md")
#     # Create output directory if it doesn't exist
#     os.makedirs(output_path, exist_ok=True)
#
#     with open(card_path, 'w', encoding='utf-8') as f:
#         f.write(card_content)
#
#     print(f"Dataset card saved to {card_path}")
#     return card_content
#

def create_dataset_card(merged_dataset, config: Dict, output_path: str):
    """Create and save a dataset card with metadata."""
    from collections import Counter
    import numpy as np

    # Single pass through dataset to collect all statistics
    dataset_counts = Counter()
    token_counts = []

    for example in merged_dataset:
        dataset_counts[example["source_dataset"]] += 1
        token_counts.append(example["total_token_count"])

    # Convert to numpy array only once
    token_counts = np.array(token_counts)

    # Pre-calculate token statistics
    total_examples = len(merged_dataset)
    total_tokens = int(np.sum(token_counts))
    min_tokens = int(token_counts.min())
    max_tokens = int(token_counts.max())
    mean_tokens = int(token_counts.mean())
    median_tokens = int(np.median(token_counts))

    # Pre-calculate threshold statistics if needed
    threshold_stats = []
    if "analysis" in config and "token_thresholds" in config["analysis"]:
        for threshold in config["analysis"]["token_thresholds"]:
            count = (token_counts > threshold).sum()
            percentage = (count / len(token_counts)) * 100
            threshold_stats.append((threshold, count, percentage))

    # Build source dataset section
    source_section = []
    for dataset, count in sorted(dataset_counts.items()):
        percentage = (count / total_examples) * 100
        source_section.append(f"- **{dataset}**: {count:,} examples ({percentage:.1f}%)")

    # Build threshold section
    threshold_section = []
    for threshold, count, percentage in threshold_stats:
        threshold_section.append(f"- Examples > {threshold} tokens: {count:,} ({percentage:.1f}%)")

    # Create dataset card content using string formatting
    card_content = f"""---
dataset_info:
  features:
  - name: task
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: message
    sequence:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: formatted_text
    dtype: string
  - name: encoded_text
    sequence: int32
  - name: message_length
    dtype: int64
  - name: total_token_count
    dtype: int64
  - name: source_dataset
    dtype: string
  - name: has_existing_messages
    dtype: bool
  - name: num_turns
    dtype: int64
  configs:
  - config_name: default
    data_files:
    - split: train
      path: data/train-*
    - split: test
      path: data/test-*
license: mit
task_categories:
- text-generation
- conversational
language:
- en
tags:
- synthetic
- chat
- instruction-following
size_categories:
- {get_size_category(total_examples)}
---

# Merged Chat Dataset

## Dataset Description

This dataset is a merged collection of multiple instruction-following and conversational datasets, formatted for supervised fine-tuning (SFT) of language models.

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics

- **Total Examples:** {total_examples:,}
- **Total Tokens:** {total_tokens:,}

- **Token Count Statistics:**
  - Min: {min_tokens:,}
  - Max: {max_tokens:,}
  - Mean: {mean_tokens:,}
  - Median: {median_tokens:,}

## Source Datasets

This merged dataset includes examples from the following sources:

{chr(10).join(source_section)}

## Token Distribution

{chr(10).join(threshold_section) if threshold_section else ""}

## Data Format

Each example contains:
- `task`: Task description or category
- `input`: User input/question
- `output`: Expected response
- `message`: List of chat messages with roles (system/user/assistant)
- `formatted_text`: Tokenizer-formatted text
- `total_token_count`: Number of tokens in the formatted text
- `source_dataset`: Original dataset identifier
- `has_existing_messages`: Whether the example had pre-existing chat format

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("path/to/this/dataset")
train_data = dataset["train"]
test_data = dataset["test"]

# Access chat messages
for example in train_data:
    messages = example["message"]  # This is a list
    for message in messages:
        print(f"{{message['role']}}: {{message['content']}}")
```

## Configuration Used

The dataset was created with the following configuration:

```json
{json.dumps(config, indent=2)}
```

## Citation

If you use this dataset, please cite the original source datasets appropriately.
"""

    # Save the dataset card
    card_path = os.path.join(output_path, "README.md")
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(card_content)

    print(f"Dataset card saved to {card_path}")
    return card_content

def get_size_category(size: int) -> str:
    """Get size category for dataset card."""
    if size < 1000:
        return "n<1K"
    elif size < 10000:
        return "1K<n<10K"
    elif size < 100000:
        return "10K<n<100K"
    elif size < 1000000:
        return "100K<n<1M"
    else:
        return "1M<n<10M"


def merge_hf_datasets(config: Dict):
    """
    Merge multiple Hugging Face datasets into a common chat format suitable for SFT.
    Memory-optimized version that processes each dataset individually.
    """
    # Get memory limit from config
    max_memory_gb = config["processing"].get("max_memory_gb", 32)
    print(f"Memory limit set to: {max_memory_gb}GB")

    log_memory("Starting dataset merge", max_memory_gb)

    # Extract parameters from config
    dataset_ids = config["datasets"]["dataset_ids"]

    # Tokenizer configuration
    tokenizer_config = config["tokenizer"]
    primary_tokenizer = tokenizer_config["primary"]
    fallback_tokenizer = tokenizer_config["fallback"]

    # Load tokenizer for main thread
    try:
        tokenizer = AutoTokenizer.from_pretrained(primary_tokenizer, trust_remote_code=True)
        tokenizer_name = primary_tokenizer
        print(f"Successfully loaded {primary_tokenizer} tokenizer")
    except Exception as e:
        print(f"Error loading {primary_tokenizer} tokenizer: {e}")
        print(f"Falling back to {fallback_tokenizer} tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer, trust_remote_code=True)
        tokenizer_name = fallback_tokenizer

    # Use HF cache directory for temporary files
    with temporary_directory(use_hf_cache=True) as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Process each dataset individually and cache results
        cache_paths = []

        for i, dataset_id in enumerate(dataset_ids):
            print(f"\nProcessing dataset {i + 1}/{len(dataset_ids)}: {dataset_id}")

            # Process single dataset and get cache path
            cache_path = process_single_dataset(
                dataset_id=dataset_id,
                config=config,
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                cache_dir=temp_dir,
                max_memory_gb=max_memory_gb
            )

            if cache_path:
                cache_paths.append(cache_path)
                print(f"Successfully processed and cached {dataset_id}")
            else:
                print(f"Failed to process {dataset_id}, skipping...")

            # Force garbage collection between datasets
            gc.collect()
            log_memory(f"Completed dataset {i + 1}/{len(dataset_ids)}", max_memory_gb)

        if not cache_paths:
            raise ValueError("No datasets were successfully processed")

        print(f"\nSuccessfully processed {len(cache_paths)} out of {len(dataset_ids)} datasets")

        # Load and merge all cached datasets
        merged_dataset = merge_cached_datasets(cache_paths, max_memory_gb)

        log_memory("Final merged dataset created", max_memory_gb)

        return merged_dataset


def save_as_jsonl(dataset, filename):
    """Save dataset as JSONL format with memory optimization."""
    print(f"Saving {len(dataset)} examples to {filename}")

    with open(filename, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if i % 10000 == 0:
                print(f"  Saved {i} examples...")

            example_dict = {
                "task": example["task"],
                "input": example["input"],
                "output": example["output"],
                "message": example["message"],
                "formatted_text": example["formatted_text"],
                "encoded_text": example["encoded_text"],
                "message_length": example["message_length"],
                "total_token_count": example["total_token_count"],
                "num_turns": example["num_turns"],
                "source_dataset": example["source_dataset"],
                "has_existing_messages": example["has_existing_messages"]
            }
            f.write(json.dumps(example_dict, ensure_ascii=False) + '\n')

    print(f"Finished saving {filename}")


# def print_statistics(merged_data, config):
#     """Print dataset statistics and token analysis."""
#     token_counts = np.array(merged_data["total_token_count"])
#
#     token_stats = {
#         "min": float(token_counts.min()),
#         "max": float(token_counts.max()),
#         "mean": float(token_counts.mean()),
#         "median": float(np.median(token_counts)),
#         "total": float(np.sum(token_counts)),
#     }
#     print(f"Token count statistics: {token_stats}")
#
#     if "analysis" in config and "token_thresholds" in config["analysis"]:
#         token_thresholds = config["analysis"]["token_thresholds"]
#         for threshold in token_thresholds:
#             count = (token_counts > threshold).sum()
#             percentage = (count / len(token_counts)) * 100
#             print(f"Examples with more than {threshold} tokens: {count} ({percentage:.2f}%)")
#
#     print("\nDataset breakdown:")
#     source_datasets = set(merged_data["source_dataset"])
#     for dataset in source_datasets:
#         dataset_examples = [ex for ex in merged_data if ex["source_dataset"] == dataset]
#         has_existing_msgs = any(ex["has_existing_messages"] for ex in dataset_examples)
#         print(
#             f"  {dataset}: {len(dataset_examples)} examples {'(with existing messages)' if has_existing_msgs else '(converted from input/output)'}")

def print_statistics(merged_data, config):
    """Print dataset statistics and token analysis."""
    from collections import Counter, defaultdict

    # Single pass through data to collect all statistics
    dataset_counts = Counter()
    dataset_has_existing_msgs = defaultdict(bool)
    token_counts = []

    for example in merged_data:
        source = example["source_dataset"]
        dataset_counts[source] += 1

        # Track if ANY example from this dataset has existing messages
        if example["has_existing_messages"]:
            dataset_has_existing_msgs[source] = True

        token_counts.append(example["total_token_count"])

    # Convert to numpy array only once
    token_counts = np.array(token_counts)

    # Calculate token statistics
    token_stats = {
        "min": float(token_counts.min()),
        "max": float(token_counts.max()),
        "mean": float(token_counts.mean()),
        "median": float(np.median(token_counts)),
        "total": float(np.sum(token_counts)),
    }
    print(f"Token count statistics: {token_stats}")

    # Token threshold analysis
    if "analysis" in config and "token_thresholds" in config["analysis"]:
        token_thresholds = config["analysis"]["token_thresholds"]
        total_examples = len(token_counts)

        for threshold in token_thresholds:
            count = (token_counts > threshold).sum()
            percentage = (count / total_examples) * 100
            print(f"Examples with more than {threshold} tokens: {count} ({percentage:.2f}%)")

    # Dataset breakdown - now using pre-calculated counts
    print("\nDataset breakdown:")
    for dataset in sorted(dataset_counts.keys()):
        count = dataset_counts[dataset]
        has_existing_msgs = dataset_has_existing_msgs[dataset]
        msg_type = "(with existing messages)" if has_existing_msgs else "(converted from input/output)"
        print(f"  {dataset}: {count} examples {msg_type}")

def main():
    parser = argparse.ArgumentParser(description='Merge Hugging Face datasets with JSON configuration')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to JSON configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    log_memory("Script started", config.get("processing", {}).get("max_memory_gb", 32))

    # Merge datasets using memory-optimized approach
    merged_data = merge_hf_datasets(config)

    # Preview the merged dataset
    print("\nSample record:")
    print(merged_data[0])

    # Print statistics
    print_statistics(merged_data, config)

    # Save outputs based on configuration
    output_config = config["output"]
    max_memory_gb = config.get("processing", {}).get("max_memory_gb", 32)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_config.get("full_dataset_filename", "output"))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save full dataset
    if output_config["save_full_dataset"]:
        full_filename = output_config["full_dataset_filename"]
        save_as_jsonl(merged_data, full_filename)
        print(f"Full dataset saved as {full_filename}")

    # Create and save train/test splits
    dataset_dict = None
    if output_config["create_splits"]:
        test_size = output_config["test_size"]
        seed = config["processing"]["seed"]

        log_memory("Creating train/test split", max_memory_gb)
        train_test_dataset = merged_data.train_test_split(test_size=test_size, seed=seed)
        dataset_dict = train_test_dataset

        train_filename = output_config["train_filename"]
        val_filename = output_config["val_filename"]

        save_as_jsonl(train_test_dataset["train"], train_filename)
        save_as_jsonl(train_test_dataset["test"], val_filename)

        print(f"Train dataset saved as {train_filename}")
        print(f"Validation dataset saved as {val_filename}")

    # Create dataset card
    if output_config.get("create_dataset_card", True):
        output_dir = output_config.get("dataset_card_dir",
                                       os.path.dirname(output_config.get("full_dataset_filename", ".")))
        create_dataset_card(merged_data, config, output_dir)

    # Push to Hugging Face Hub if configured
    if output_config["push_to_hub"]["enabled"]:
        hub_config = output_config["push_to_hub"]

        if dataset_dict is None:
            test_size = output_config["test_size"]
            seed = config["processing"]["seed"]
            dataset_dict = merged_data.train_test_split(test_size=test_size, seed=seed)

        log_memory("Pushing to hub", max_memory_gb)
        dataset_dict.push_to_hub(
            hub_config["repo_name"],
            private=hub_config["private"]
        )
        print(f"Dataset pushed to Hugging Face Hub: {hub_config['repo_name']}")

    log_memory("Script completed", max_memory_gb)


if __name__ == "__main__":
    main()