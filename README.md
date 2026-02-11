# SAFE

## Project Overview ✨

The `SAFE` (**Static-Adversarial Fusion Enhancement**) project focuses on the research and development of **scientific and technological achievement classification** based on natural language processing and pre-trained language models. Aiming at the pain points of scientific and technological texts such as dense professional terms, complex semantic levels, high noise, small samples and high training costs, this project constructs a dual-path semantic feature enhancement framework, and combines mixed precision training with FGM adversarial training collaborative optimization strategy. It effectively makes up for the deficiency of traditional models in capturing low-frequency vertical domain words, suppresses overfitting in small sample and high noise environment, and significantly improves the accuracy and robustness of scientific and technological achievement classification.

The model achieves **76.17% classification accuracy (ACC)** on the public dataset CSTA-Corpus, and reduces video memory usage by about 42% while ensuring performance, which provides a reliable technical solution for the accurate docking of scientific and technological achievements with regional industrial policies.

- Dataset: [CSTA-Corpus](https://github.com/leeeeee10/CSTA-Corpus)
- Paper: 基于异构特征增强与协同优化的科技成果分类方法

## Project Structure 📁

```
SAFE/
├── CSTA-Corpus/                # Dataset directory
│   ├── data/
│   │   ├── train.txt           # Training set
│   │   ├── dev.txt             # Validation set
│   │   ├── test.txt            # Test set
│   │   └── class.txt           # classification
│   └── saved_dict/             # saving directory
├── models/
│   └── bert.py					# BERT model with SAFE
|	├── base_bert.py			# based bert
|	├── cross_attention.py		# model of cross_attention
|	├── ernie.py				# model of ernie
|	├── hard_attention.py		# model of hard_attention
|	├── ...
├── pretrained/                 
│   ├── bert_pretrained/        # pretrained model of bert-base-chinese
|	├── ernie/					# pretrained model of ernie
├── run.py                      # Main entry point for text classification training
├── train_eval.py               # Training and evaluation logic
└── utils.py                    # Utility functions (dataset loading, etc.)
└── README.md
```

## Main Functional Modules

### 1. Static-Dynamic Feature Fusion Module 🧠

Realized in `models/bert.py`, it is the core feature enhancement module of the SAFE model, solving the problem of mismatch between general semantics and domain terms in scientific and technological texts:

- **Static Word Vector Loading**: Load 300-dimensional SGNS static word vectors, and realize precise token alignment with BERT vocabulary through `load_static_embedding` function, and fix FP32 precision to ensure semantic stability.
- **Adaptive Gated Fusion**: Introduce learnable global gate control parameters, and automatically learn the optimal fusion ratio of static term features and BERT dynamic context features through Softmax, making the model adapt to general vocabulary and professional terms.
- **Multi-head Attention Pooling**: Design a multi-head attention pooling layer to replace the traditional [CLS] vector, automatically filter the key features of classification from the long text sequence, and suppress the interference of redundant noise information.
- **Multi-layer Feature Splicing**: Splice the [CLS] vectors of the last four layers of BERT encoder to capture the complete information flow from syntactic structure to abstract semantics, and form a complete feature representation with the pooled features of attention.

### 2. Mixed Precision & Adversarial Training Module ⚡

Realized in `train_eval.py` and `utils.py`, it solves the dual challenges of high video memory overhead of high-dimensional features and easy overfitting in small sample scenarios:

- **Dynamic Mixed Precision Training (AMP)**: Based on `torch.amp`, divide the model into high-precision area (static word vector layer, classifier, FP32) and low-precision area (Transformer encoder, FP16), and introduce gradient scaling mechanism to avoid gradient underflow, reduce video memory usage by about 42% and improve training throughput by 1.6 times.
- **FGM Adversarial Regularization**: Implement non-symmetric adversarial training, only add tiny adversarial disturbance to BERT dynamic embedding layer, keep static word vector frozen as semantic anchor, force the model to learn noise-immune robust characteristics, and effectively improve the generalization ability in high noise + small sample environment.

### 3. Model Training & Evaluation Module 📚

Cooperatively realized by `run.py`, `train_eval.py` and `utils.py`, it provides a complete pipeline of data processing, model training, evaluation and result analysis:

- **Data Processing**: Realize data set construction, tokenization, padding and iterator building through `build_dataset` and `build_iterator` functions, and support the division of training/validation/test sets according to the ratio of 7:1.5:1.5.
- **Training Logic**: Include AdamW optimizer, linear warm-up learning rate scheduler, gradient clipping, class weight balancing and other strategies, and support mixed precision and adversarial training joint optimization.
- **Evaluation & Analysis**: Calculate multi-dimensional evaluation indicators (ACC, F1, Precision, Recall), automatically save the best model according to the validation set loss, and generate detailed analysis files of correct/misclassified samples (including attention weight, prediction probability, key token, etc.) in CSV format.
- **Visualization & Log**: Record training process indicators, video memory usage, and support the analysis of attention weight distribution and misclassified sample reasons.

## Environment Requirements 🖥️

- Python: 3.12.11 (recommended, compatible with Python 3.10+)
- PyTorch: 2.7.1+cu128 (support CUDA for GPU acceleration, necessary for mixed precision training)
- Transformers: 4.57.0 (for BERT model loading and Tokenizer)
- Scikit-learn: For calculation of classification indicators and data processing
- Numpy: For numerical calculation and feature matrix processing
- Matplotlib: For simple visualization of training results (optional)
- Tqdm: For progress bar display of data loading and training

## Install Dependencies 💻

```base
pip install torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install transformers==4.57.0 scikit-learn numpy matplotlib tqdm
```

## Usage

### 1. Preparations

1. Download the public dataset [CSTA-Corpus](https://github.com/leeeeee10/CSTA-Corpus) and place it in the root directory of the project (keep the directory structure unchanged).
2. Place the `bert-base-chinese` pre-trained model files in the `pretrained/bert_pretrained/` directory (available on [Hugging Face](https://huggingface.co/bert-base-chinese)).
3. Download the 300-dimensional SGNS static word vector file `sgns.merge.char` and place it in the `pretrained/bert_pretrained/` directory (Chinese word vector open source resource).
4. Ensure that the CUDA environment is configured correctly (GPU is recommended, otherwise the training speed is slow and mixed precision training is not supported).

### 2. Model Training

Run the following command in the project root directory to start model training (the only specified model is `bert`, which is the SAFE model implementation):

```base
# Default training: seed=109, static word vector dim=300, FGM epsilon=0.5
python run.py --model bert

# Custom training: specify random seed, static word vector path and dimension
python run.py --model bert --seed 109 --static-emb-path ./pretrained/bert_pretrained/sgns.merge.char --emb_dim 300
```

**Training Output**:

- The trained model weight is saved to `CSTA-Corpus/saved_dict/bert.ckpt`.
- The peak video memory usage, training/validation set indicators (ACC, Loss) are printed in real time during training.
- Automatic early stopping is supported when the validation set loss does not improve for a long time.

### 3. Model Evaluation & Result Analysis

The training script will automatically perform model testing on the test set after training, and output the following contents:

1. **Multi-dimensional evaluation indicators**: Accuracy (ACC), macro F1, Precision, Recall and other core indicators.
2. **Misclassified sample analysis**: Print the top 5 misclassified samples, including text content, true/predicted labels, prediction probability distribution and key attention tokens.
3. **Result file saving**: Generate `correct_details.csv` and `misclassified_details.csv` in `CSTA-Corpus/saved_dict/`, which record the detailed information of all samples (attention weight, key token, prediction probability, etc.).
4. **Benchmark data**: Print peak video memory usage, training time and other benchmark indicators.

## Experimental Results 📊

All experiments are carried out on the CSTA-Corpus dataset (no artificial cleaning of the original noise), and the super parameters are unified (seed=109, batch_size=64, lr=2e-5, FGM ε=0.5). The core experimental results are as follows:

### 1. Model Performance Comparison

SAFE model is significantly superior to traditional deep learning models, attention mechanism variants and knowledge-enhanced pre-trained models in all indicators:

|      Model Name       |  ACC (%)  |  F1 (%)   | Precision (%) | Recall (%) | Parameters (M) |
| :-------------------: | :-------: | :-------: | :-----------: | :--------: | :------------: |
|        TextCNN        |   72.55   |   70.90   |     73.30     |   72.55    |      104       |
|       TextRCNN        |   72.92   |   71.59   |     73.59     |   72.92    |      104       |
|    Soft-Attention     |   73.76   |   72.63   |     74.13     |   73.76    |      104       |
|         ERNIE         |   72.24   |   70.07   |     73.58     |   72.24    |      102       |
| Multi-level Attention |   74.20   |   73.33   |     73.62     |   73.44    |      108       |
|    **SAFE (Ours)**    | **76.17** | **74.82** |   **74.76**   | **75.37**  |      115       |

### 2. Ablation Experiment

Verify the effectiveness of each core module of the SAFE model (the complete model has the best performance):

|     Experimental Method      |  ACC (%)  |  F1 (%)   | Precision (%) | Recall (%) |
| :--------------------------: | :-------: | :-------: | :-----------: | :--------: |
|  Remove Static Word Vector   |   75.33   |   74.02   |     74.77     |   75.33    |
| Remove Feature Concatenation |   74.86   |   73.95   |     73.80     |   74.36    |
| Remove Adversarial Training  |   75.28   |   74.40   |     74.74     |   74.92    |
|     Only BERT (Original)     |   75.36   |   74.53   |     73.93     |   75.33    |
|  **SAFE (Complete Model)**   | **76.17** | **74.82** |   **74.76**   | **75.37**  |

### 3. Key Optimization Results

- **Video Memory Optimization**: Compared with the original BERT (FP32), the video memory usage is reduced by about **42%** under the premise of unchanged accuracy.
- **Training Efficiency**: The training throughput is increased by **1.6 times** through mixed precision training.
- **Model Stability**: Under 9 different random seeds, the ACC is stable in 75.23% ~ 76.17%, and the standard deviation is only 0.31%.
- **Adversarial Robustness**: The optimal FGM disturbance amplitude ε=0.5, which achieves the balance between effective attack and semantic retention.

## Notes ⚠️

- Before training, ensure that the pre-trained model, static word vector and dataset files are placed in the specified path, otherwise the file loading error will be reported.
- The model is based on GPU training (CUDA). If only CPU is used, please comment out the mixed precision training related code (`autocast`, `GradScaler`) in the code, and the training speed will be significantly reduced.
- The key super parameters (FGM ε=0.5, seed=109, dropout=0.8) have been optimized through experiments, and it is not recommended to modify them arbitrarily.
- The dataset is divided into training/validation/test sets according to 7:1.5:1.5 by default, and the division logic can be modified in the `build_dataset` function in `utils.py`.
- The static word vector is fixed as FP32 precision and does not participate in fine-tuning, which can reduce the training cost and ensure the stability of domain knowledge.

## Contribution 🤝

Contributions to the SAFE project are welcome, including but not limited to:

- Raising issues (model optimization, bug repair, function expansion).
- Submitting pull requests (code refactoring, adding new features, supporting more pre-trained models).
- Improving project documentation (supplementing usage examples, experimental details, code comments).
- Optimizing the model (feature space alignment, adaptive adversarial training, multi-model fusion).

Please follow the GitHub open source specification for operation when contributing.

## Contact 📧

If you have any questions, suggestions or research cooperation needs in the process of using the model, please contact us through **GitHub Issues** of the project. We will reply and solve the problem as soon as possible.
