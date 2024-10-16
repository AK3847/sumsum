# Sumsum

A minimal CLI tool to locally summarize any text using LLM!

Built around [Ollama](https://github.com/ollama/ollama-python) and [fine-tuned version](https://huggingface.co/AKT47/Llama_3.2_3B_fine_tune_summarization) of [Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

## Installation:

```terminal
pip install sumsum
```

## Setup:

- Run the `init` command to check dependency download model and generate modelfile:
  ```terminal
  sumsum init
  ```

## Usage

- Use the command `run` with your text file path:
  ```terminal
  sumsum run path\to\file
  ```
- you can also use the flag `--verbose` to get additional information:
  (time taken to load the model, time taken for generating new tokens,number of generated tokens etc)
  `terminal
    sumsum run path\to\file --verbose
    `
