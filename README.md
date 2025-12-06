# Dataset Merger Tool 🚀

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/)

A powerful, production-ready tool for merging and processing multiple HuggingFace datasets into unified formats optimized for Large Language Model (LLM) fine-tuning and training.

## 🌟 Why This Tool?

HuggingFace recently launched exciting new features for dataset management, including:
- **Row-level editing** for CSV datasets
- **Instant dataset duplication** for personal use

While these features are great, simply duplicating datasets isn't enough. This tool **supercharges** dataset workflows by:

✅ **Intelligently merging** multiple datasets from different sources
✅ **Standardizing formats** into unified chat/instruction formats
✅ **Filtering & validating** data based on token counts and quality
✅ **Memory-optimized processing** for large-scale datasets (handles 100K+ examples)
✅ **Automatic train/test splits** with configurable ratios
✅ **Direct HuggingFace Hub integration** for seamless publishing
✅ **Production-ready** with parallel processing, error handling, and detailed logging

## 🎯 Key Features

### 1. **Multi-Dataset Merging**
- Merge 10+ datasets in a single run
- Automatic field mapping and normalization
- Preserves original metadata and source information

### 2. **Smart Chat Format Conversion**
- Converts input/output pairs to chat format
- Supports pre-existing message formats
- Tokenizer-aware formatting (Qwen, GPT-2, custom tokenizers)

### 3. **Advanced Filtering**
- Token-based filtering (min/max tokens per record)
- Quality validation and message format checking
- Configurable sampling limits per dataset

### 4. **Memory & Performance Optimized**
- Multi-threaded/multi-process processing
- Intelligent caching system
- Memory monitoring with configurable limits
- Environment auto-detection (Colab, Jupyter, standard)

### 5. **Production Features**
- Comprehensive dataset cards with statistics
- JSONL export format
- Direct HuggingFace Hub upload
- Detailed logging and error reporting

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dataset_tools.git
cd dataset_tools

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```bash
python dataset_merger.py --config config_example.json
```

### Configuration File

The tool uses JSON configuration files for maximum flexibility. Here's a minimal example:

```json
{
  "datasets": {
    "dataset_ids": [
      "microsoft/orca-math-word-problems-200k",
      "HuggingFaceTB/cosmopedia-100k",
      "Open-Orca/1million-gpt-4"
    ],
    "split": "train",
    "field_mapping": {
      "microsoft/orca-math-word-problems-200k": {
        "question": "input",
        "answer": "output"
      }
    },
    "task_mapping": {
      "microsoft/orca-math-word-problems-200k": "Mathematical reasoning and problem solving"
    }
  },
  "processing": {
    "max_records_per_dataset": 100000,
    "max_tokens_per_record": 8096,
    "min_tokens_per_record": 50,
    "num_workers": 8,
    "max_memory_gb": 32
  },
  "tokenizer": {
    "primary": "Qwen/Qwen3-0.6B",
    "fallback": "gpt2"
  },
  "output": {
    "create_splits": true,
    "test_size": 0.2,
    "push_to_hub": {
      "enabled": true,
      "repo_name": "your-username/merged-dataset",
      "private": false
    }
  }
}
```

See `config_example.json` for a complete configuration with 19 datasets!

## 📊 Output Format

The tool produces datasets with the following structure:

```json
{
  "task": "Mathematical reasoning and problem solving",
  "input": "What is 2+2?",
  "output": "2+2 equals 4",
  "message": [
    {"role": "system", "content": "Mathematical reasoning and problem solving"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4"}
  ],
  "formatted_text": "<|im_start|>system\nMathematical reasoning...",
  "encoded_text": [151644, 8948, ...],
  "total_token_count": 42,
  "message_length": 156,
  "num_turns": 3,
  "source_dataset": "microsoft/orca-math-word-problems-200k",
  "has_existing_messages": false
}
```

## 🎓 Use Cases

### 1. **LLM Fine-Tuning**
Create high-quality, diverse training datasets by merging multiple instruction-following datasets.

### 2. **Domain-Specific Models**
Combine datasets from specific domains (math, code, science) with proper task categorization.

### 3. **Research & Benchmarking**
Create reproducible dataset combinations with exact versioning and statistics.

### 4. **Data Augmentation**
Expand existing datasets with compatible sources while maintaining format consistency.

## 🔧 Advanced Features

### Memory Management
```json
{
  "processing": {
    "max_memory_gb": 16,  // Auto-adjusts batch sizes
    "batch_size": 2000,   // Processing batch size
    "force_threading": true  // Use threads instead of processes
  }
}
```

### Environment Detection
The tool automatically detects and optimizes for:
- Google Colab
- Jupyter Notebooks
- Standard Python environments

### Token Analysis
Get detailed statistics on token distribution:
```json
{
  "analysis": {
    "token_thresholds": [512, 1024, 2048, 4096, 8096]
  }
}
```

## 📈 Performance

Tested configurations:
- ✅ **19 datasets** merged successfully
- ✅ **1M+ examples** processed
- ✅ **32GB RAM** maximum usage (configurable)
- ✅ **Multi-core** parallel processing
- ✅ **HuggingFace Hub** direct upload

## 🤝 Integration with HuggingFace Ecosystem

This tool complements HuggingFace's new dataset features:

1. **Duplicate → Merge → Customize**: Use HF's duplication feature, then merge with other datasets
2. **Edit → Validate → Re-merge**: Make row-level edits, then re-merge with validation
3. **Create → Share → Iterate**: Build custom datasets and share via HF Hub

### Feature Comparison

| Feature | Manual Process | HF Duplication | **This Tool** |
|---------|---------------|----------------|---------------|
| Multi-dataset merge | ❌ | ❌ | ✅ |
| Format standardization | ❌ | ❌ | ✅ |
| Token filtering | ❌ | ❌ | ✅ |
| Memory optimization | ❌ | ❌ | ✅ |
| Chat format conversion | ❌ | ❌ | ✅ |
| Validation | ❌ | ❌ | ✅ |
| Auto dataset cards | ❌ | ❌ | ✅ |
| HF Hub upload | ✅ | ✅ | ✅ |
| Row-level edit | ❌ | ✅ | Planned |

## 🌍 Real-World Example

From the [TraceMind ecosystem](https://huggingface.co/blog/MCP-1st-Birthday/tracemind-ecosystem) (HuggingFace Hackathon project):

```bash
# Merge 19 diverse datasets for multi-domain reasoning
python dataset_merger.py --config config_v7.json

# Result: 800K+ examples covering:
# - Mathematical reasoning
# - Code generation & debugging
# - Scientific problem solving
# - Security & vulnerability analysis
# - Conversational AI
```

## 🛠️ Configuration Options

### Dataset Configuration
- `dataset_ids`: List of HuggingFace dataset IDs
- `split`: Dataset split to use (train/test/validation)
- `field_mapping`: Map dataset-specific fields to standard format
- `task_mapping`: Assign task descriptions to datasets
- `datasets_with_messages`: Datasets already in chat format

### Processing Options
- `max_records_per_dataset`: Limit samples per dataset
- `max_tokens_per_record`: Filter out long examples
- `min_tokens_per_record`: Filter out short examples
- `seed`: Random seed for reproducibility
- `batch_size`: Processing batch size
- `num_workers`: Parallel workers

### Output Options
- `save_full_dataset`: Save complete merged dataset
- `create_splits`: Generate train/test splits
- `test_size`: Test split ratio
- `create_dataset_card`: Auto-generate dataset card
- `push_to_hub`: Upload to HuggingFace Hub

## 📝 Dataset Card Generation

Automatically generates comprehensive dataset cards with:
- Dataset statistics (size, token counts, distributions)
- Source dataset breakdown
- Token distribution analysis
- Usage examples
- Full configuration snapshot

## 🐛 Error Handling

The tool includes robust error handling:
- Invalid message format detection and fixing
- Missing field warnings
- Memory overflow protection
- Tokenization error recovery
- Partial dataset processing (continues on errors)

## 🔒 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- ✅ Free to use, modify, and distribute
- ✅ Must share source code of modifications
- ✅ Network use = distribution (SaaS clause)
- ✅ Commercial use allowed with attribution

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

- Built for the **HuggingFace Hackathon** community
- Part of the **TraceMind ecosystem** project
- Inspired by the need for better dataset tooling in the LLM space

## 📧 Contact & Support

- **Author**: Kshitij Thakkar
- **HuggingFace**: [@kshitijthakkar](https://huggingface.co/kshitijthakkar)
- **Blog Post**: [TraceMind Ecosystem](https://huggingface.co/blog/MCP-1st-Birthday/tracemind-ecosystem)

## 🚀 Future Roadmap

- [ ] Web UI for visual dataset configuration
- [ ] Advanced deduplication algorithms
- [ ] Quality scoring and filtering
- [ ] Dataset versioning and diff tools
- [ ] Integration with HF Datasets Server API
- [ ] Support for streaming datasets
- [ ] Multi-modal dataset support (images, audio)

## ⭐ Star History

If this tool helps your work, please consider:
- ⭐ Starring the repository
- 🐦 Sharing on social media
- 📝 Writing about your use case
- 🤝 Contributing improvements

---

**Made with ❤️ for the HuggingFace community**
