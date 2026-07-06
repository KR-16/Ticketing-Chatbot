# Ticketing-Chatbot

Multilingual customer-support ticket classification. Fine-tunes
`bert-base-multilingual-cased` to predict the **ticket type**
(Incident, Request, Problem, Change) from a ticket's subject and body,
with experiment tracking in MLflow.

## Dataset

20,000 support tickets (German and English) in
`data/dataset-tickets-multi-lang-4-20k.csv` with columns:

`subject, body, answer, type, queue, priority, language, tag_1..tag_8`

## Project structure

```
config/config.yaml      # data paths, preprocessing, model, and MLflow settings
data/                   # raw dataset
notebooks/              # exploratory analysis
src/data/               # loading, splitting, preprocessing (cleaning, encoding, tokenization)
src/models/             # model construction and training loop
main.py                 # end-to-end pipeline entry point
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows (source venv/bin/activate on Unix)
pip install -r requirements.txt
```

NLTK resources (stopwords, punkt) download automatically on first run.

## Running

```bash
python main.py
```

### Experiment tracking

MLflow is optional. If the server configured in `config.yaml`
(`http://localhost:5000` by default) isn't running, training automatically
falls back to local file tracking in `./mlruns` — inspect those runs with
`mlflow ui`. To use a tracking server instead:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

The pipeline loads and splits the data (80/20), cleans and tokenizes the
text, fine-tunes the model, logs train/validation loss per epoch to
MLflow, and saves a complete **inference bundle** to `models/bundle/`
(configurable via `model.bundle_dir` in `config/config.yaml`).

### Inference bundle

Everything needed to serve predictions lives in one directory:

```
models/bundle/
  config.json, model.safetensors    # fine-tuned model (save_pretrained)
  tokenizer_config.json, vocab.txt  # tokenizer (save_pretrained)
  label_map.json                    # target column + ordered class names
  training_config.yaml              # exact config the model was trained with
```

Load it with `src.models.bundle.load_bundle(path)`, which returns the
model (in eval mode), tokenizer, class names, and training config.
