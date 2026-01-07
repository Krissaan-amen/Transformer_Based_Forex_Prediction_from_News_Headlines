# Transformer-Based Forex Prediction from News Headlines

## Overview
This project fine-tunes **four BERT-based models** to predict forex price movements based on financial news. The models classify news headlines as indicators of **upward** or **downward** price movement. The approach leverages **pre-trained transformers** and applies **domain-specific fine-tuning** on forex-related financial text.

## Models Explored
This project fine-tunes and compares four different transformer architectures. Each model offers a unique trade-off between size, speed, and performance.

| Model Name | Hugging Face Identifier | Description |
| :--- | :--- | :--- |
| **TinyBERT** | `huawei-noah/TinyBERT_General_4L_312D` | A small, fast, and distilled version of BERT, optimized for performance on resource-constrained devices. |
| **ALBERT** | `albert-base-v2` | "A Lite BERT" that achieves significant parameter reduction, leading to lower memory usage and faster training. |
| **DistilBERT** | `distilbert-base-uncased-finetuned-sst-2-english` | A distilled, lighter version of BERT that retains substantial performance. This specific checkpoint is pre-tuned for sentiment analysis. |
| **MiniLM** | `microsoft/MiniLM-L12-H384-uncased` | A small pre-trained model that uses deep self-attention distillation to create a compact and efficient transformer. |


## Datasets
Two datasets are used, both available in the `merged_data/` directory:
1. **Daily Forex News Dataset** (`eurusd_daily_news.csv`)  
   - Aggregates financial news at a **daily** level with corresponding EUR/USD price movements.
2. **15-Minute Interval Dataset** (`eurusd_15min_data.csv`)  
   - Captures **high-frequency** market reactions to financial news, with price movements recorded every **15 minutes**.

## Features
- **Configurable Pipeline**: All parameters are managed via a central `config.yaml` file.
- **Multiple Modes**: Supports a `"train"` mode for full training and a `"evaluate"` mode for testing pre-trained models.
- **Class Balancing**: Implements class weighting to handle imbalanced datasets.
- **Robust Training**: Uses dropout for regularization and supports multiple optimizers like AdamW.
- **Model Checkpointing**: Automatically saves the best-performing model based on validation accuracy.

## Requirements
The necessary dependencies are listed in `requirements.txt`:
```bash
torch
transformers
scikit-learn
numpy
pandas
matplotlib
pyyaml
```
## Getting Started

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Krissaan-amen/Transformer_Based_Forex_Prediction_from_News_Headlines.git
    cd Transformer_Based_Forex_Prediction_from_News_Headlines
    ```

2.  **Install dependencies:**
    The necessary dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
All experiment parameters are controlled via the `config.yaml` file. This allows for easy modification of the training and evaluation process without changing the source code.

Key configurable parameters include:
- `operation_mode`: Set to `"train"` or `"evaluate"`.
- `model_name`: Choose the transformer model to use.
- `data_path`: Specify the path to the dataset.
- `epochs`, `batch_size`, `learning_rate`: Adjust training hyperparameters.

## Usage
The script is run from the command line and requires the path to the configuration file.

1.  **Configure the experiment** by editing `config.yaml`.
2.  **Run the script:**
    ```bash
    python main.py --config config.yaml
    ```
The script will automatically perform the action specified by the `operation_mode` in the config file.

## Future Enhancements
- **Real-time Prediction**: Develop a live pipeline with a dashboard to process news as it is released and display real-time predictions.
- **Strategy Backtesting**: Integrate the model's predictions into a simulated trading environment to backtest its financial viability.
- **Advanced NLP Analysis**: Use attention visualization to understand which parts of a news headline most influence the model's decision.
- **API Development**: Build a RESTful API to serve on-demand predictions from the fine-tuned model.

## Contact
For questions about the implementation or model performance:
- **Email**: amen.krissaan@gmail.com

## License
This project is for educational and research purposes. Please cite appropriately if used in academic work.

---
*This project demonstrates the intersection of financial markets and modern natural language processing, showcasing practical skills in both domains.*
