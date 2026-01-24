# LLM-KE-RoBerta-MFF

## Project Overview ✨

The `LLM-KE-RoBerta-MFF` project focuses on research and development in natural language processing and large language models (LLM). It covers several important tasks including model fine-tuning, knowledge-enhanced retrieval, and text classification. By integrating different technologies and model architectures, it aims to improve the performance of natural language processing tasks and provide strong support for research and applications in related fields.
dataset:https://github.com/leeeeee10/CSTA-Corpus
## Project Structure 📁

```
LLM-KE-RoBerta-MFF/
├── lora_rag_LLM/
│   ├── main.py               # Core program for model fine-tuning, responsible for loading models, preparing datasets, fine-tuning models, and saving fine-tuned models
│   ├── rag.py                # Knowledge-enhanced retrieval module, implementing the retrieval of relevant knowledge from the knowledge base and integrating it into the original prompt
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  # Model file directory, storing the base model files for fine-tuning
│   ├── datasets.jsonl        # Dataset file containing sample data for model fine-tuning
│   ├── final_models/         # Directory for saving the full final model, storing the completely fine-tuned model
│   └── saved_models/         # Directory for saving LoRA models, storing the model parameters after LoRA fine-tuning
├── CSTA-Corpus/
│   ├── keyword.txt           # Keyword text file, possibly containing keyword information related to text classification tasks
│   └── data/
│       └── class.txt         # Category text file, recording the list of categories in text classification tasks
├── models/
│   ├── bert.py               # Implementation of text classification model based on Roberta, using multi-head attention pooling mechanism
│   ├── bert自适应.py          # Implementation of adaptive Roberta text classification model, introducing adaptive attention mechanism
│   ├── bert_RCNN.py          # Implementation of text classification model combining BERT and RCNN (code not provided, but inferred from the structure)
│   ├── bert时间注意力.py       # Implementation of BERT text classification model with time attention, adding time attention mechanism
│   └── bert_RNN.py           # Implementation of text classification model combining BERT and RNN (code not provided, but inferred from the structure)
├── run.py                    # Running script, responsible for loading datasets, initializing models, and performing training and evaluation
├── train_eval.py             # Training and evaluation script, containing the specific logic for model training and evaluation
└── utils.py                  # Utility function script, providing utility functions such as building datasets and iterators
```

## Main Functional Modules

### 1. LoRA Fine-tuning and Knowledge-Enhanced Retrieval (`lora_rag_LLM`) 🧠

#### `main.py`

This file is the core program for model fine-tuning, and its main functions are as follows:
- Model Loading: Use `AutoModelForCausalLM` from the `transformers` library to load the `DeepSeek-R1-Distill-Qwen-1.5B` model from the specified path, and support model quantization (using `BitsAndBytesConfig`) to reduce memory usage.
- Dataset Preparation: Obtain sample data from `data_pre.py`, save it as a `datasets.jsonl` file, and use the `datasets` library to load and process the dataset, dividing it into training and validation sets.
- Model Fine-tuning: Use `LoraConfig` from the `peft` library for LoRA fine-tuning of the model, perform training through `Trainer` from the `transformers` library, and set training parameters such as the number of training epochs and batch size.
- Model Saving: Save the fine-tuned LoRA model to the `saved_models` directory, and save the full model to the `final_models` directory.

#### `rag.py`

This file implements the Knowledge-Enhanced Retrieval (RAG) function, and the main steps are as follows:
- Knowledge Loading: Load category-organized knowledge from the specified folder, and extract prompts and completion contents.
- TF-IDF Preprocessing: Use `TfidfVectorizer` to vectorize prompts and build a TF-IDF matrix.
- Knowledge Retrieval: Support retrieval of multi-category queries, giving priority to retrieval based on domain names, and using TF-IDF retrieval if no domain is detected.
- Prompt Enhancement: Integrate the retrieved knowledge into the original prompt in a specific format to enhance the model's answering ability.

### 2. Text Classification (`models` and `run.py`) 📚

#### `models` Directory

This directory contains implementations of various BERT-based text classification models. Take `bert.py` as an example, its main features are as follows:
- Model Structure: Use `BertModel` as the base model, add a multi-head attention pooling layer, splice the hidden states of the last four layers and perform attention pooling, and then use a two-layer MLP for classification.
- Configuration Parameters: Include training set, validation set, test set paths, category list, learning rate, batch size and other configuration information.

#### `run.py`

This is the running script for text classification tasks, and its main functions are as follows:
- Argument Parsing: Use `argparse` to parse command line arguments and specify the model to use (such as `bert`, `ERNIE`, etc.).
- Data Loading: Call the `build_dataset` and `build_iterator` functions in `utils.py` to load and process the dataset.
- Model Initialization: Import the corresponding model class according to the specified model name and initialize the model.
- Model Training: Call the `train` function in `train_eval.py` for model training and evaluation.

## Environment Requirements 🖥️

- Python 3.x: The project code is developed based on Python 3, ensuring Python version compatibility.
- PyTorch: Used for the construction and training of deep learning models.
- Transformers: Provides pre-trained models and related tools to facilitate model loading and fine-tuning.
- Datasets: Used for loading and processing datasets.
- Scikit-learn: Provides machine learning tools such as TF-IDF vectorizers.
- Numpy: Used for numerical calculations.

## Install Dependencies 💻

```bash
pip install torch transformers datasets scikit-learn numpy
```

## Usage

### Model Fine-tuning

1. Ensure that the `lora_rag_LLM/DeepSeek-R1-Distill-Qwen-1.5B` directory contains the required model files.
2. Run the following command for model fine-tuning:
```bash
python lora_rag_LLM/main.py
```
The fine-tuned LoRA model will be saved to the `lora_rag_LLM/saved_models` directory, and the full model will be saved to the `lora_rag_LLM/final_models` directory.

### Knowledge-Enhanced Retrieval

You can call the `KnowMoveRAG` class in `rag.py` in the code for knowledge retrieval and prompt enhancement:
```python
from lora_rag_LLM.rag import KnowMoveRAG

rag = KnowMoveRAG()
original_prompt = "Generate keywords in the field of intelligent manufacturing and equipment"
augmented_prompt = rag.augment_prompt(original_prompt)
print(augmented_prompt)
```

### Text Classification

Run the following command for text classification tasks:
```bash
python run.py --model bert
```
The `--model` parameter can be used to select different models, such as `bert`, `ERNIE`, etc.

## Notes ⚠️

- Before running model fine-tuning, please ensure that the `lora_rag_LLM/datasets.jsonl` file exists, or modify the dataset loading part in `main.py` as needed.
- The dataset path and model save path for text classification tasks can be modified in the configuration files under the `models` directory.

## Contribution 🤝

Contributions to this project are welcome, including but not limited to raising issues, submitting code, and improving documentation. Please follow the project's contribution guidelines when operating.

## Contact 📧

If you have any questions or suggestions, please contact us via GitHub Issues.


### Model Performance Comparison

| Model Name | ACC (%) | F1 (%) | Precision (%) | Recall (%) | Parameters (M) |
|------------|---------|--------|---------------|------------|----------------|
| macbert    | 73.02   | 69.86  | 69.36         | 70.73      | 102            |
| Rbt3       | 72.08   | 69.86  | 70.08         | 69.96      | 102            |
| Rbt6       | 71.87   | 68.59  | 67.85         | 70.69      | 334            |
| bert       | 70.86   | 67.80  | 65.95         | 70.96      | 102            |
| LLM-KE-RoBerta-MFF | 74.47 | 73.33 | 73.62 | 73.44 | 102 |

### Attention Mechanism Performance Comparison

| Attention Mechanism | ACC | F1 | Precision | Recall |
|---------------------|-----|----|-----------|--------|
| Soft Attention       | 72.93 | 73.36 | 69.79 | 70.62 |
| Cross Attention     | 71.14 | 68.69 | 68.33 | 69.92 |
| Adaptive Attention  | 70.40 | 67.70 | 67.24 | 69.29 |
| Single-head Attention| 72.39 | 69.53 | 68.73 | 72.83 |
| Multi-head Self-attention (Our Method) | 74.47 | 73.33 | 73.62 | 73.44 |
