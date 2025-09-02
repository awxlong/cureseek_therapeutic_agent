# CureSeek Therapeutic Agent: Code for CURE-Bench

We participated in the [CURE-Bench competition](https://www.kaggle.com/competitions/cure-bench) on therapeutic reasoning via external tool usage with LLMs, and we open-source our code. 

## Preliminaries
Prior to running our code, please: 
- `pip install -r requirements.txt`.
- Create a [Weights&Biases](https://wandb.ai/site) account and export your wandb login API key to the environment which is used to track metrics during supervised fine-tuning (SFT).
- We also use [PrimeKG](https://www.nature.com/articles/s41597-023-01960-3), which you can download by running `wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620`, and check https://github.com/mims-harvard/PrimeKG for more details. Computationally, it's a Pandas dataframe which you can query and check [PrimeKG_Querier](RAG.py) to see how we use it for retrieval augmented generation (RAG).
- Have a valid e-mail which will be used to make API calls to the European PubMed Central to retrieve articles for RAG.
- The validation and test sets can be downloaded through [Kaggle](https://www.kaggle.com/competitions/cure-bench) after joining and accepting the terms and conditions of the competition. For this code repo, you can place them inside `data/` 
- You can check https://curebench.ai/ for more information.

## Methodology

Our approaches emphasizes affordability on low computational budget:

1. We first tried supervised-finetuning a Qwen-distilled, quantized [DeepSeek-R1](https://www.kaggle.com/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b) of 1.5 billion parameters. The competition has no training data, and the validation data has no reasoning traces before arriving to an answer. We annotated validation data by using both quantized [TxAgent](https://huggingface.co/mradermacher/TxAgent-T1-Llama-3.1-8B-GGUF) and prompting DeepSeek-R1-distill-qwen-1.5b to explain how to arrive at the correct answer for questions in the validation set. Please see [experiments/self_correction_deepseek.py](experiments/self_correction_deepseek.py) where we perform batch inference of input validation data. 
      
      1.  This augmented validation set was treated as our 'training' set to fine-tune a base DeepSeek-R1-distill-qwen-1.5b, and the code is in [experiments/sft_deepseek.py](experiments/sft_deepseek.py), and the supervised finetuning pipeline is mainly elaborated upon https://www.kaggle.com/code/danielphalen/grpotrainer-deepseekr1. This approach didn't seem to work. Inference with base DeepSeek-R1-distill-qwen-1.5b yielded higher test score, suggesting fine-tuning may have impaired its internal reasoning. 

2. We next tried simply performing inference with quantized TxAgent, which yielded a score close to the competition's baseline of $0.5$. Please check [experiments/quantized_txagent_inference.py](experiments/quantized_txagent_inference.py). The code is written to support multiprocessing to speed up inference on Kaggle.
       
      1.  Inference with quantized TxAgent leads to many `No answer` entries as the model requires external tools to find out more information about novel drugs, their contraindications, their recommended dosage, among others, prior to inferring an answer. Nonetheless, because we are constrained by computational budget, we aimed at keeping answers within $2048$ tokens, so we didn't use the original [TxAgent](https://github.com/mims-harvard/TxAgent) which is very, very computationally expensive (e.g. recommended H100 GPU usage with more than 80GB of memory and maximum tokens of $>90000$). 
      
      2. To keep inference affordable, we needed to feed information on drugs that TxAgent didn't know about. To do this, we first use the above DeepSeek-R1-distill-qwen-1.5b to extract relevant drug entities in the test questions; please check [experiments/entity_extraction_deepseek.py](experiments/entity_extraction_deepseek.py). We use the extracted entities to construct (complex) queries which can be used to retrieve information about drug entities from the European PMC and PrimeKG; please check [experiments/rag_context_augmentation.py](experiments/rag_context_augmentation.py). 

      3. The extracted information served as 'augmented' context to do RAG-based inference for a second time on 'No answer' entries using quantized TxAgent with [experiments/quantized_txagent_inference.py](experiments/quantized_txagent_inference.py). This raised the test score a little bit. We didn't have enough time anymore, so for remaining `No answer` entries we simply filled them in via similarity-based semantic search using Qwen2-1.5B-instruct. This consists of extracting embeddings of the partial answer of quantized TxAgent, then measuring its cosine similarity with each of the MCQ answers and outputting the one with the highest similarity. This raised the final test score accuracy to $0.42$; please check [semantic_search.py](semantic_search.py).   

We are wondering whether there exists a very simple solution to this complicated CURE-Bench benchmark, perhaps by simply doing online RAG (searching Google via an API), and then using a quantized TxAgent to reason over the search results...

<!-- 
## CUREBench Starter Kit

[![ProjectPage](https://img.shields.io/badge/CUREBench-Page-red)](https://curebench.ai) [![ProjectPage](https://img.shields.io/badge/CUREBench-Kaggle-green)](https://www.kaggle.com/competitions/cure-bench)

A simple inference framework for the CURE-Bench bio-medical AI competition. This starter kit provides an easy-to-use interface for generating submission data in CSV format.

## Quick Start

### Installation Dependencies
```bash
pip install -r requirements.txt
```

## Baseline Setup

If you want to use the ChatGPT baseline:
1. Set up your Azure OpenAI resource
2. Configure environment variables:
```bash
export AZURE_OPENAI_API_KEY_O1="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

If you want to use the open-ended models, such as Qwen:
For local models, ensure you have sufficient GPU memory:
```bash
# Install CUDA-compatible PyTorch if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transfomers
```

## 📁 Project Structure

```
├── eval_framework.py      # Main evaluation framework
├── dataset_utils.py       # Dataset loading utilities
├── run.py                 # Command-line evaluation script
├── metadata_config.json   # Example metadata configuration
├── requirements.txt       # Python dependencies
└── competition_results/   # Output directory for your results
```

## Dataset Preparation

Download the val and test dataset from the Kaggle site:
```
https://www.kaggle.com/competitions/cure-bench
```

For val set, configure datasets in your `metadata_config_val.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  }
}
```

For test set, configure datasets in your `metadata_config_test.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_testset.jsonl",
    "description": "CureBench 2025 test questions"
  }
}
```

## Usage Examples

### Basic Evaluation with Config File
```bash
# Run with configuration file (recommended)
python run.py --config metadata_config_test.json
```

## 🔧 Configuration

### Metadata Configuration
Create a `metadata_config_val.json` file:
```json
{
  "metadata": {
    "model_name": "gpt-4o-1120",
    "model_type": "ChatGPTModel",
    "track": "internal_reasoning",
    "base_model_type": "API",
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  },
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  },
  "output_dir": "competition_results",
  "output_file": "submission.csv"
}
```

### Required Metadata Fields
- `model_name`: Display name of your model
- `track`: Either "internal_reasoning" or "agentic_reasoning"
- `base_model_type`: Either "API" or "OpenWeighted"
- `base_model_name`: Name of the underlying model
- `dataset`: Name of the dataset

Note: You can leave the following fields empty for the first round of submissions:
`additional_info`,`average_tokens_per_question`, `average_tools_per_question`, and `tool_category_coverage`.
**Please ensure these fields are filled for the final submission.**


### Question Type Support
The framework handles three distinct question types:
1. **Multiple Choice**: Questions with lettered options (A, B, C, D, E)
2. **Open-ended Multiple Choice**: Open-ended questions converted to multiple choice format  
3. **Open-ended**: Free-form text answers


## Output Format

The framework generates submission files in CSV format with a zip package containing metadata. The CSV structure includes:
- `id`: Question identifier
- `prediction`: Model's answer (choice for multiple choice, text for open-ended)
- `reasoning_trace`: Model's reasoning process
- `choice`: The choice for the multi-choice questions.

The accompanying metadata includes:
```json
{
  "meta_data": {
    "model_name": "gpt-4o-1120",
    "track": "internal_reasoning",
    "model_type": "ChatGPTModel",
    "base_model_type": "API", 
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  }
}
```

## Support

For issues and questions: 
1. Check the error messages (they're usually helpful!)
2. Ensure all dependencies are installed
3. Review the examples in this README
4. Open an Github Issue.

Happy competing! -->
