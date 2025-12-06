# Contributing to Dataset Merger Tool

Thank you for your interest in contributing to the Dataset Merger Tool! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Body size
- Race
- Ethnicity
- Age
- Religion
- Nationality

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct that could reasonably be considered inappropriate

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When creating a bug report, include:
- **Clear title** describing the issue
- **Detailed description** of the problem
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, package versions)
- **Configuration file** used (sanitized if needed)
- **Error messages** and stack traces

**Example Bug Report:**

```markdown
**Title:** MemoryError when processing datasets larger than 500K examples

**Description:**
The tool crashes with a MemoryError when processing the full
microsoft/orca-math-word-problems-200k dataset.

**Steps to Reproduce:**
1. Use config with max_records_per_dataset: null
2. Run: python dataset_merger.py --config config.json
3. Observe crash after ~500K examples

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10.12
- RAM: 32GB
- datasets: 2.14.5

**Error Message:**
```
MemoryError: Memory usage exceeded 32GB limit
```

**Expected Behavior:**
Should process all 200K examples with proper memory management.
```

### Suggesting Features

Feature requests are welcome! Please include:
- **Clear use case** - What problem does this solve?
- **Proposed solution** - How would it work?
- **Alternatives considered** - Other approaches you've thought about
- **Additional context** - Examples, mockups, references

**Example Feature Request:**

```markdown
**Feature:** Support for streaming datasets

**Use Case:**
Processing extremely large datasets (>10M examples) that don't fit in memory.

**Proposed Solution:**
Add a "streaming_mode" config option that processes datasets in chunks
without loading everything into memory.

**Alternatives:**
- Manual dataset splitting before processing
- Cloud-based processing

**Additional Context:**
HuggingFace datasets library supports streaming:
https://huggingface.co/docs/datasets/stream
```

### Pull Requests

We actively welcome pull requests! Areas that need help:
- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New features
- 🎨 Code quality improvements
- 🧪 Test coverage
- 🌐 Internationalization

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/dataset_tools.git
   cd dataset_tools
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

5. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

6. **Test your changes**
   ```bash
   # Run the tool with test config
   python dataset_merger.py --config test_config.json

   # Run any tests (when available)
   pytest tests/
   ```

## Pull Request Process

### Before Submitting

- ✅ Test your changes thoroughly
- ✅ Update documentation if needed
- ✅ Follow coding standards (see below)
- ✅ Write clear commit messages
- ✅ Ensure no merge conflicts with main branch

### Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <short summary>

<detailed description (optional)>

<footer (optional)>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**

```
feat: Add support for streaming datasets

- Implement streaming mode for large datasets
- Add streaming_mode config option
- Update memory management for streaming

Closes #123
```

```
fix: Handle missing field mapping gracefully

Previously crashed when field_mapping was undefined.
Now provides helpful warning and uses default mapping.

Fixes #456
```

### Submitting

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

3. **PR Description Template**

   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Code refactoring

   ## Testing
   - [ ] Tested locally
   - [ ] Added/updated tests
   - [ ] All tests passing

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   - [ ] Commit messages are clear

   ## Related Issues
   Closes #123
   Related to #456
   ```

4. **Review Process**
   - Maintainers will review your PR
   - Address any feedback
   - Once approved, it will be merged

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **4 spaces** for indentation
- Maximum line length: **120 characters**
- Use **descriptive variable names**

### Code Organization

```python
# 1. Standard library imports
import os
import json
from typing import Dict, List

# 2. Third-party imports
import numpy as np
from datasets import load_dataset

# 3. Local imports
from utils import helper_function
```

### Documentation

**Function Documentation:**

```python
def process_dataset(dataset_id: str, config: Dict) -> Dataset:
    """
    Process a single dataset with given configuration.

    Args:
        dataset_id: HuggingFace dataset identifier
        config: Configuration dictionary with processing parameters

    Returns:
        Processed dataset with standardized format

    Raises:
        ValueError: If dataset_id is invalid
        MemoryError: If processing exceeds memory limit

    Example:
        >>> config = {"processing": {"max_records": 1000}}
        >>> ds = process_dataset("username/dataset", config)
    """
    pass
```

### Error Handling

```python
# Good: Specific exceptions with helpful messages
try:
    dataset = load_dataset(dataset_id)
except Exception as e:
    raise ValueError(f"Failed to load dataset {dataset_id}: {e}")

# Bad: Generic exceptions
try:
    dataset = load_dataset(dataset_id)
except:
    pass
```

### Type Hints

Use type hints for better code clarity:

```python
def merge_datasets(
    datasets: List[Dataset],
    config: Dict[str, Any]
) -> Dataset:
    """Merge multiple datasets."""
    pass
```

## Testing

### Writing Tests

```python
import pytest
from dataset_merger import process_dataset

def test_process_dataset_basic():
    """Test basic dataset processing."""
    config = {
        "processing": {
            "max_records_per_dataset": 100,
            "min_tokens_per_record": 10
        }
    }

    result = process_dataset("test/dataset", config)

    assert len(result) <= 100
    assert all(ex["total_token_count"] >= 10 for ex in result)

def test_process_dataset_invalid_id():
    """Test error handling for invalid dataset ID."""
    config = {}

    with pytest.raises(ValueError):
        process_dataset("invalid/dataset", config)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dataset_merger.py

# Run with coverage
pytest --cov=dataset_merger tests/
```

## Documentation

### Code Comments

```python
# Good: Explain WHY, not WHAT
# Use thread pool instead of process pool for Colab compatibility
executor = ThreadPoolExecutor(max_workers=num_workers)

# Bad: Obvious comments
# Create executor
executor = ThreadPoolExecutor(max_workers=num_workers)
```

### README Updates

When adding features, update:
- README.md - High-level description
- USAGE.md - Detailed usage examples
- config_example.json - Configuration example

## Recognition

Contributors will be:
- Listed in the project's Contributors section
- Mentioned in release notes for significant contributions
- Eligible for special recognition badges

## Questions?

- **General questions:** Open a GitHub Discussion
- **Bug reports:** Open an Issue
- **Feature ideas:** Open an Issue with [Feature Request] tag
- **Direct contact:** See README.md for contact information

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.

---

Thank you for contributing to Dataset Merger Tool! 🎉
