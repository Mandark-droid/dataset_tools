# Usage Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Configuration

Create a JSON configuration file (or use `config_example.json` as a template):

```json
{
  "datasets": {
    "dataset_ids": ["dataset1", "dataset2"],
    "split": "train"
  },
  "processing": {
    "max_records_per_dataset": 10000,
    "max_tokens_per_record": 2048,
    "min_tokens_per_record": 10
  },
  "tokenizer": {
    "primary": "Qwen/Qwen3-0.6B"
  },
  "output": {
    "create_splits": true,
    "test_size": 0.2
  }
}
```

### 3. Run the Tool

```bash
python dataset_merger.py --config your_config.json
```

## Configuration Reference

### Dataset Configuration

#### `dataset_ids` (required)
List of HuggingFace dataset identifiers to merge.

```json
"dataset_ids": [
  "microsoft/orca-math-word-problems-200k",
  "HuggingFaceTB/cosmopedia-100k"
]
```

#### `split` (required)
Which split to load from each dataset.

```json
"split": "train"  // Options: "train", "test", "validation"
```

#### `field_mapping` (optional)
Map dataset-specific field names to standard format.

```json
"field_mapping": {
  "microsoft/orca-math-word-problems-200k": {
    "question": "input",
    "answer": "output"
  },
  "arcee-ai/synthetic-data-gen": {
    "Abstract": "task",
    "Question": "input",
    "Answer": "output"
  }
}
```

Standard fields:
- `input`: User input/question
- `output`: Expected response
- `task`: Task description/system prompt

#### `task_mapping` (optional)
Assign task descriptions to datasets for better context.

```json
"task_mapping": {
  "microsoft/orca-math-word-problems-200k": "Solve mathematical word problems with step-by-step reasoning",
  "HuggingFaceTB/cosmopedia-100k": "Educational content generation"
}
```

#### `datasets_with_messages` (optional)
List datasets that already have chat/message format.

```json
"datasets_with_messages": ["miromind-ai/MiroVerse-v0.1"]
```

#### `messages_field_name` (optional, default: "messages")
Field name for pre-existing message data.

```json
"messages_field_name": "messages"
```

#### `preserve_existing_fields` (optional, default: true)
Whether to preserve existing `task` and `source_dataset` fields.

```json
"preserve_existing_fields": true
```

### Processing Configuration

#### `max_records_per_dataset` (optional)
Maximum number of records to load from each dataset.

```json
"max_records_per_dataset": 100000  // null = no limit
```

#### `max_tokens_per_record` (optional)
Filter out examples exceeding this token count.

```json
"max_tokens_per_record": 8096  // null = no limit
```

#### `min_tokens_per_record` (optional)
Filter out examples below this token count.

```json
"min_tokens_per_record": 50  // null = no limit
```

#### `seed` (required)
Random seed for reproducibility.

```json
"seed": 42
```

#### `batch_size` (optional, default: 1000)
Processing batch size for memory management.

```json
"batch_size": 2000
```

#### `num_workers` (optional, default: auto)
Number of parallel workers for processing.

```json
"num_workers": 8  // Auto-adjusted based on CPU cores
```

#### `max_memory_gb` (optional, default: 32)
Maximum memory usage limit in GB.

```json
"max_memory_gb": 32  // Triggers warnings and adjustments
```

#### `force_threading` (optional, default: false)
Force thread-based processing instead of multiprocessing.

```json
"force_threading": false  // Auto-detected for Colab/Jupyter
```

#### `cache_dir` (optional)
Directory for temporary cache files.

```json
"cache_dir": "H:\\Hugging_Face_Cache"
```

### Tokenizer Configuration

#### `primary` (required)
Primary tokenizer to use for formatting.

```json
"primary": "Qwen/Qwen3-0.6B"
```

#### `fallback` (required)
Fallback tokenizer if primary fails to load.

```json
"fallback": "gpt2"
```

### Analysis Configuration

#### `token_thresholds` (optional)
Token count thresholds for statistics.

```json
"token_thresholds": [512, 1024, 2048, 4096, 8096]
```

### Output Configuration

#### `save_full_dataset` (optional, default: false)
Save the complete merged dataset as JSONL.

```json
"save_full_dataset": false
```

#### `full_dataset_filename` (optional)
Filename for full dataset output.

```json
"full_dataset_filename": "data/merged_dataset.jsonl"
```

#### `create_splits` (optional, default: true)
Create train/test splits.

```json
"create_splits": true
```

#### `test_size` (optional, default: 0.2)
Proportion of data for test split (0.0 to 1.0).

```json
"test_size": 0.2  // 20% test, 80% train
```

#### `train_filename` (optional)
Output filename for training split.

```json
"train_filename": "data/train_dataset.jsonl"
```

#### `val_filename` (optional)
Output filename for validation/test split.

```json
"val_filename": "data/val_dataset.jsonl"
```

#### `create_dataset_card` (optional, default: true)
Generate README.md dataset card.

```json
"create_dataset_card": false
```

#### `dataset_card_dir` (optional)
Directory for dataset card output.

```json
"dataset_card_dir": "data/"
```

#### `push_to_hub` (required)
HuggingFace Hub upload configuration.

```json
"push_to_hub": {
  "enabled": true,
  "repo_name": "username/dataset-name",
  "private": false
}
```

## Common Use Cases

### Use Case 1: Math Dataset for Fine-Tuning

```json
{
  "datasets": {
    "dataset_ids": [
      "microsoft/orca-math-word-problems-200k",
      "argilla/distilabel-math-preference-dpo",
      "SynthLabsAI/Big-Math-RL-Verified"
    ],
    "split": "train",
    "field_mapping": {
      "microsoft/orca-math-word-problems-200k": {
        "question": "input",
        "answer": "output"
      },
      "argilla/distilabel-math-preference-dpo": {
        "instruction": "input",
        "chosen_response": "output"
      },
      "SynthLabsAI/Big-Math-RL-Verified": {
        "problem": "input",
        "answer": "output"
      }
    },
    "task_mapping": {
      "microsoft/orca-math-word-problems-200k": "Mathematical word problem solving",
      "argilla/distilabel-math-preference-dpo": "Mathematical reasoning",
      "SynthLabsAI/Big-Math-RL-Verified": "Advanced mathematics"
    }
  },
  "processing": {
    "max_records_per_dataset": 50000,
    "max_tokens_per_record": 4096,
    "min_tokens_per_record": 20,
    "seed": 42
  },
  "tokenizer": {
    "primary": "Qwen/Qwen3-0.6B",
    "fallback": "gpt2"
  },
  "output": {
    "create_splits": true,
    "test_size": 0.1,
    "push_to_hub": {
      "enabled": true,
      "repo_name": "your-username/math-reasoning-dataset",
      "private": false
    }
  }
}
```

### Use Case 2: Code Generation Dataset

```json
{
  "datasets": {
    "dataset_ids": [
      "microsoft/NextCoderDataset",
      "microsoft/EpiCoder-func-380k"
    ],
    "split": "train",
    "field_mapping": {
      "microsoft/NextCoderDataset": {
        "prompt": "input",
        "completion": "output"
      },
      "microsoft/EpiCoder-func-380k": {
        "instruction": "input",
        "answer": "output"
      }
    },
    "task_mapping": {
      "microsoft/NextCoderDataset": "Code completion and editing",
      "microsoft/EpiCoder-func-380k": "Function-level code generation"
    }
  },
  "processing": {
    "max_tokens_per_record": 8096,
    "min_tokens_per_record": 100
  },
  "output": {
    "push_to_hub": {
      "enabled": true,
      "repo_name": "your-username/code-generation-merged"
    }
  }
}
```

### Use Case 3: Small Test Run

```json
{
  "datasets": {
    "dataset_ids": ["Open-Orca/1million-gpt-4"],
    "split": "train"
  },
  "processing": {
    "max_records_per_dataset": 1000,
    "max_memory_gb": 8,
    "num_workers": 2
  },
  "output": {
    "save_full_dataset": true,
    "create_splits": false,
    "push_to_hub": {
      "enabled": false
    }
  }
}
```

## Troubleshooting

### Memory Issues

**Problem**: `MemoryError: Memory usage exceeded XGB limit`

**Solution**:
1. Reduce `max_memory_gb` to available RAM
2. Reduce `batch_size` (try 500-1000)
3. Reduce `num_workers` (try 2-4)
4. Enable `force_threading: true`
5. Reduce `max_records_per_dataset`

```json
"processing": {
  "max_memory_gb": 16,
  "batch_size": 500,
  "num_workers": 2,
  "force_threading": true
}
```

### Field Mapping Errors

**Problem**: `Warning: Dataset X is missing required fields: {'input', 'output'}`

**Solution**: Check the dataset structure and add proper field mapping.

```bash
# Inspect dataset first
from datasets import load_dataset
ds = load_dataset("dataset-name", split="train")
print(ds.features)
```

Then update your config:
```json
"field_mapping": {
  "dataset-name": {
    "actual_input_field": "input",
    "actual_output_field": "output"
  }
}
```

### Tokenizer Issues

**Problem**: Tokenizer fails to load

**Solution**: The tool automatically falls back to the fallback tokenizer. Check:
1. Network connectivity
2. HuggingFace Hub access
3. Tokenizer name spelling

### Processing Speed

**Problem**: Processing is too slow

**Solution**:
1. Increase `num_workers` (match CPU cores)
2. Increase `batch_size` (if memory allows)
3. Use `force_threading: false` for multiprocessing (if not in Colab)

```json
"processing": {
  "num_workers": 16,
  "batch_size": 5000,
  "force_threading": false
}
```

## Advanced Features

### Custom Cache Directory

Store temporary files in a specific location:

```json
"processing": {
  "cache_dir": "/path/to/large/disk/cache"
}
```

### HuggingFace Authentication

For private datasets or hub uploads:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token as environment variable
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### Preprocessing Multiple Configs

```bash
# Process multiple configurations sequentially
for config in configs/*.json; do
  python dataset_merger.py --config "$config"
done
```

### Integration with Training Pipelines

```python
from datasets import load_dataset

# Load your merged dataset
dataset = load_dataset("your-username/merged-dataset")

# Use with transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForCausalLM.from_pretrained("model-name")

# Train on merged data
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.train()
```

### Batch Processing with Different Seeds

Create multiple variations for robustness:

```bash
for seed in 42 123 456; do
  # Update config with new seed
  jq ".processing.seed = $seed" config.json > "config_seed_${seed}.json"
  python dataset_merger.py --config "config_seed_${seed}.json"
done
```

---

For more examples, see the main [README.md](README.md).
