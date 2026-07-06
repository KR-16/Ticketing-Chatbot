"""Unit tests for the data pipeline: loader and preprocessor.

Everything runs on small in-memory frames. The tokenizer download and NLTK
network calls are stubbed out; only the stopwords test touches nltk data
(downloading it once if missing).
"""

from types import SimpleNamespace

import nltk
import pandas as pd
import pytest

import src.data.data_preprocessor as preprocessor_module
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor


def make_frame():
    return pd.DataFrame({
        "subject": ["Crash on login"],
        "body": ["The app crashes"],
        "type": ["Incident"],
        "queue": ["IT Support"],
        "priority": ["low"],
        "language": ["en"],
        "tag_1": ["Bug"],
    })


@pytest.fixture
def preprocessor(monkeypatch, tmp_path):
    """A DataPreprocessor with no tokenizer download and no NLTK network."""
    monkeypatch.setattr(
        preprocessor_module, "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda name: object()),
    )
    monkeypatch.setattr(preprocessor_module, "_ensure_nltk_data", lambda: None)
    config = tmp_path / "config.yaml"
    config.write_text(
        "preprocessing:\n"
        "  max_length: 200\n"
        "  language_detection: False\n"
        "  remove_stopwords: False\n"
    )
    return DataPreprocessor(str(config))


# --- clean_text ---

def test_clean_text_masks_emails_and_urls():
    cleaned = DataPreprocessor.clean_text(
        "Mail john@doe.com or visit https://example.com <b>now</b>"
    )
    # the special-character pass strips the placeholder brackets, so the
    # guarantee is: PII gone, bare placeholder token remains
    assert "EMAIL" in cleaned
    assert "URL" in cleaned
    assert "john@doe.com" not in cleaned
    assert "example.com" not in cleaned
    assert "<b>" not in cleaned


def test_clean_text_non_string_returns_empty():
    assert DataPreprocessor.clean_text(None) == ""
    assert DataPreprocessor.clean_text(float("nan")) == ""


def test_clean_text_collapses_whitespace():
    assert DataPreprocessor.clean_text("a   b\n\nc") == "a b c"


# --- stopword removal ---

def test_remove_stopwords_maps_iso_codes():
    nltk.download("stopwords", quiet=True)
    assert DataPreprocessor.remove_stopwords("this is the server", "en") == "server"
    assert DataPreprocessor.remove_stopwords("der Server ist kaputt", "de") == "Server kaputt"


def test_remove_stopwords_unknown_language_passes_through():
    assert DataPreprocessor.remove_stopwords("hello world", "xx") == "hello world"
    assert DataPreprocessor.remove_stopwords("hello world", "unknown") == "hello world"


# --- validate_data ---

def test_validate_data_accepts_complete_frame(preprocessor):
    assert preprocessor.validate_data(make_frame()) is True


def test_validate_data_rejects_missing_queue(preprocessor):
    assert preprocessor.validate_data(make_frame().drop(columns=["queue"])) is False


def test_validate_data_rejects_missing_tag_columns(preprocessor):
    assert preprocessor.validate_data(make_frame().drop(columns=["tag_1"])) is False


def test_validate_data_rejects_empty_frame(preprocessor):
    assert preprocessor.validate_data(make_frame().iloc[0:0]) is False


# --- prepare_labels fit/transform ---

def test_prepare_labels_transform_reuses_train_mapping(preprocessor):
    train = pd.DataFrame({
        "type": ["Incident", "Request", "Change"],
        "priority": ["low", "high", "medium"],
        "queue": ["A", "B", "C"],
    })
    test = pd.DataFrame({"type": ["Request"], "priority": ["low"], "queue": ["B"]})

    fitted = preprocessor.prepare_labels(train, fit=True)
    transformed = preprocessor.prepare_labels(test, fit=False)

    train_mapping = dict(zip(fitted["type"], fitted["type_label"]))
    assert transformed["type_label"].iloc[0] == train_mapping["Request"]


def test_prepare_labels_transform_without_fit_raises(preprocessor):
    with pytest.raises(ValueError, match="not fitted"):
        preprocessor.prepare_labels(make_frame(), fit=False)


# --- process_tags ---

def test_process_tags_uses_all_tag_columns_and_aligns_index(preprocessor):
    frame = pd.DataFrame(
        {
            "tag_1": ["Bug", "Crash"],
            "tag_2": [None, "Outage"],
            "tag_8": ["Hardware", None],
        },
        index=[10, 42],  # non-contiguous, like train_test_split output
    )
    result = preprocessor.process_tags(frame, fit=True)

    assert list(result.index) == [10, 42]
    assert "tag_encoded_Hardware" in result.columns  # tag_8 not dropped
    assert result.loc[10, "tag_encoded_Bug"] == 1
    assert result.loc[42, "tag_encoded_Outage"] == 1
    assert not result.filter(like="tag_encoded_").isna().any().any()


def test_process_tags_transform_drops_unseen_tags(preprocessor):
    train = pd.DataFrame({"tag_1": ["Bug", "Crash"]})
    test = pd.DataFrame({"tag_1": ["Bug", "NeverSeen"]})

    preprocessor.process_tags(train, fit=True)
    result = preprocessor.process_tags(test, fit=False)

    assert "tag_encoded_NeverSeen" not in result.columns
    assert result.iloc[0]["tag_encoded_Bug"] == 1


# --- data loader ---

def test_data_loader_loads_and_splits(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"value": range(100)}).to_csv(csv_path, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data:\n"
        f"  raw_data_path: {csv_path.as_posix()}\n"
        f"  test_size: 0.2\n"
        f"  random_state: 42\n"
    )

    loader = DataLoader(str(config))
    data = loader.load_data()
    train, test = loader.split_data(data)

    assert len(train) == 80
    assert len(test) == 20
    assert set(train.index).isdisjoint(test.index)
