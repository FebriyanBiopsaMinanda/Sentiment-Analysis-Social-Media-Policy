from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Social Media Restrictions for Minors",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# KONFIGURASI
# =========================================================
MODEL_ORDER = [
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "IndoBERT",
    "NusaBERT",
]

LABEL_DEFAULTS = [
    "PENOLAKAN_KEBIJAKAN",
    "DUKUNGAN_KEBIJAKAN",
    "KRITIK_PEMERINTAHAN",
    "NETRAL",
]

NER_DEFAULTS = [
    "ORGANISASI",
    "PLATFORM",
    "AGE_GROUP",
    "POLICY",
    "DIGITAL_RISK",
]

CLASSICAL_FILENAMES = {
    "Decision Tree": {
        "Label": "decision_tree_multi_label_accept.joblib",
        "NER": "decision_tree_multi_ner.joblib",
    },
    "Random Forest": {
        "Label": "random_forest_multi_label_accept.joblib",
        "NER": "random_forest_multi_ner.joblib",
    },
    "XGBoost": {
        "Label": "xgboost_multi_label_accept.joblib",
        "NER": "xgboost_multi_ner.joblib",
    },
}


# =========================================================
# PATH MODEL
# =========================================================
def find_project_root() -> Path:
    page_file = Path(__file__).resolve()

    candidates = [
        page_file.parents[1],
        Path.cwd(),
        Path.cwd().parent,
    ]

    for candidate in candidates:
        if (candidate / "Outputs").exists():
            return candidate.resolve()

    return page_file.parents[1].resolve()


PROJECT_ROOT = find_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "Outputs"


def classical_model_path(model_name: str, task_name: str) -> Path:
    task_folder = "Multi Label" if task_name == "Label" else "Multi NER"
    return (
        OUTPUTS_ROOT
        / "Classical ML"
        / task_folder
        / CLASSICAL_FILENAMES[model_name][task_name]
    )


def deep_model_path(model_name: str, task_name: str) -> Path:
    task_folder = "Multi Label" if task_name == "Label" else "Multi NER"
    return OUTPUTS_ROOT / "Deep Sequence" / task_folder / model_name


# =========================================================
# PEMUAT MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def load_classical_payload(model_path_string: str) -> dict[str, Any]:
    import joblib

    payload = joblib.load(model_path_string)

    if isinstance(payload, dict):
        return payload

    # Dukungan untuk file lama yang hanya menyimpan pipeline/model.
    return {"pipeline": payload}


@st.cache_resource(show_spinner=False)
def load_deep_model(model_dir_string: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir_string)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


# =========================================================
# HELPER PREDIKSI
# =========================================================
def labels_from_binary_prediction(
    prediction: Any,
    labels: list[str],
    empty_message: str,
) -> str:
    if hasattr(prediction, "toarray"):
        prediction = prediction.toarray()

    prediction_array = np.asarray(prediction)

    if prediction_array.ndim > 1:
        prediction_array = prediction_array[0]

    detected = [
        label
        for index, label in enumerate(labels)
        if index < len(prediction_array) and int(prediction_array[index]) == 1
    ]

    return ", ".join(detected) if detected else empty_message


def predict_classical(
    model_name: str,
    task_name: str,
    text: str,
) -> str:
    model_path = classical_model_path(model_name, task_name)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"File model belum ditemukan: {model_path}"
        )

    payload = load_classical_payload(str(model_path))
    model = payload.get("pipeline") or payload.get("model")

    if model is None:
        raise ValueError(
            "Isi file joblib tidak memiliki key 'pipeline' atau 'model'."
        )

    if task_name == "Label":
        labels = list(payload.get("accept_labels", LABEL_DEFAULTS))
        empty_message = "Tidak ada label terdeteksi"
    else:
        labels = list(payload.get("entity_labels", NER_DEFAULTS))
        empty_message = "Tidak ada NER terdeteksi"

    prediction = model.predict([text])

    return labels_from_binary_prediction(
        prediction=prediction,
        labels=labels,
        empty_message=empty_message,
    )


def read_threshold_from_metrics(
    metrics_path: Path,
    model_name: str,
) -> float | None:
    if not metrics_path.is_file():
        return None

    try:
        dataframe = pd.read_excel(metrics_path, sheet_name="Metrics")
    except Exception:
        return None

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    model_column = normalized_columns.get("model")
    threshold_column = normalized_columns.get("threshold")

    if model_column is None or threshold_column is None:
        return None

    rows = dataframe[
        dataframe[model_column].astype(str).str.casefold()
        == model_name.casefold()
    ]

    if rows.empty:
        return None

    try:
        return float(rows.iloc[0][threshold_column])
    except (TypeError, ValueError):
        return None


def get_deep_threshold(model_name: str, task_name: str) -> float:
    task_folder = "Multi Label" if task_name == "Label" else "Multi NER"
    task_root = OUTPUTS_ROOT / "Deep Sequence" / task_folder

    if task_name == "Label":
        config_path = task_root / "multi_label_config.json"

        if config_path.is_file():
            try:
                with config_path.open("r", encoding="utf-8") as file:
                    config = json.load(file)

                threshold_key = f"{model_name.lower()}_threshold"
                threshold_key = threshold_key.replace("bert", "bert")

                # Key notebook: indobert_threshold / nusabert_threshold.
                compact_key = (
                    "indobert_threshold"
                    if model_name == "IndoBERT"
                    else "nusabert_threshold"
                )

                if compact_key in config:
                    return float(config[compact_key])
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        metrics_path = task_root / "Multi_Label_evaluasi_deep_sequence.xlsx"
    else:
        metrics_path = task_root / "NER_evaluasi_deep_sequence.xlsx"

    threshold = read_threshold_from_metrics(metrics_path, model_name)
    return threshold if threshold is not None else 0.50


def deep_labels_from_config(
    model: Any,
    fallback_labels: list[str],
) -> list[str]:
    id2label = getattr(model.config, "id2label", None)

    if not isinstance(id2label, dict) or not id2label:
        return fallback_labels

    labels: list[str] = []

    for index in range(len(fallback_labels)):
        value = id2label.get(index, id2label.get(str(index)))

        if value is None or str(value).upper().startswith("LABEL_"):
            return fallback_labels

        labels.append(str(value))

    return labels


def predict_deep(
    model_name: str,
    task_name: str,
    text: str,
) -> str:
    import torch

    model_dir = deep_model_path(model_name, task_name)

    required_files = [
        model_dir / "config.json",
        model_dir / "tokenizer_config.json",
    ]

    if not model_dir.is_dir() or not all(path.exists() for path in required_files):
        raise FileNotFoundError(
            f"Folder model belum lengkap: {model_dir}"
        )

    model, tokenizer, device = load_deep_model(str(model_dir))
    fallback_labels = LABEL_DEFAULTS if task_name == "Label" else NER_DEFAULTS
    labels = deep_labels_from_config(model, fallback_labels)
    threshold = get_deep_threshold(model_name, task_name)

    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    prediction = (probabilities >= threshold).astype(int)
    empty_message = (
        "Tidak ada label terdeteksi"
        if task_name == "Label"
        else "Tidak ada NER terdeteksi"
    )

    return labels_from_binary_prediction(
        prediction=prediction,
        labels=labels,
        empty_message=empty_message,
    )


def run_prediction(
    model_name: str,
    task_name: str,
    text: str,
) -> tuple[str, str]:
    try:
        if model_name in CLASSICAL_FILENAMES:
            result = predict_classical(model_name, task_name, text)
        else:
            result = predict_deep(model_name, task_name, text)

        return result, "Berhasil"
    except FileNotFoundError as error:
        return "Model belum tersedia", str(error)
    except ImportError as error:
        return "Dependensi belum tersedia", str(error)
    except Exception as error:
        return "Prediksi gagal", str(error)


def format_result_badges(value: Any) -> str:
    """Mengubah hasil prediksi menjadi badge HTML yang mudah dibaca."""
    raw_value = str(value or "").strip()

    if not raw_value:
        return '<span class="result-empty">Tidak ada hasil</span>'

    special_messages = {
        "Model belum tersedia",
        "Dependensi belum tersedia",
        "Prediksi gagal",
        "Tidak ada label terdeteksi",
        "Tidak ada NER terdeteksi",
    }

    if raw_value in special_messages:
        css_class = (
            "result-empty"
            if raw_value.startswith("Tidak ada")
            else "result-error"
        )
        return (
            f'<span class="{css_class}">' 
            f'{html.escape(raw_value)}</span>'
        )

    values = [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]

    badges = []
    for item in values:
        display_value = item.replace("_", " ")
        badges.append(
            '<span class="result-badge">'
            f'{html.escape(display_value)}'
            '</span>'
        )

    return '<div class="result-badge-group">' + "".join(badges) + "</div>"


def prediction_status(
    label_status: str,
    ner_status: str,
) -> tuple[str, str]:
    label_success = label_status == "Berhasil"
    ner_success = ner_status == "Berhasil"

    if label_success and ner_success:
        return "Berhasil", "status-success"

    if label_success or ner_success:
        return "Sebagian berhasil", "status-warning"

    return "Belum berhasil", "status-error"


def prediction_table_html(dataframe: pd.DataFrame) -> str:
    rows = []

    for index, row in dataframe.iterrows():
        status_text, status_class = prediction_status(
            str(row["Status Label"]),
            str(row["Status NER"]),
        )

        rows.append(
            "<tr>"
            f'<td class="number-cell">{index + 1}</td>'
            '<td class="model-cell">'
            '<div class="model-name">'
            f'{html.escape(str(row["Model"]))}'
            '</div>'
            '</td>'
            '<td class="prediction-cell">'
            f'{format_result_badges(row["Multi-Label Classification"])}'
            '</td>'
            '<td class="prediction-cell">'
            f'{format_result_badges(row["Named Entity Recognition (NER)"])}'
            '</td>'
            '<td class="status-cell">'
            f'<span class="status-badge {status_class}">'
            f'{html.escape(status_text)}'
            '</span>'
            '</td>'
            "</tr>"
        )

    return (
        '<div class="prediction-table-card">'
        '<div class="prediction-table-wrapper">'
        '<table class="prediction-table">'
        '<thead><tr>'
        '<th class="number-column">No.</th>'
        '<th>Metode</th>'
        '<th>Hasil Multi-Label Classification</th>'
        '<th>Hasil Named Entity Recognition (NER)</th>'
        '<th>Status</th>'
        '</tr></thead>'
        '<tbody>'
        + "".join(rows)
        + '</tbody></table></div></div>'
    )


# =========================================================
# HERO DAN NAVIGASI
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">
            ANALISIS OPINI PUBLIK TERHADAP KEBIJAKAN PEMBATASAN MEDIA SOSIAL ANAK
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title=None,
    options=["Beranda", "Data", "Visualisasi", "Result", "Prediction"],
    icons=["house-heart", "database", "bar-chart-line", "clipboard-data", "robot"],
    default_index=4,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "transparent",
        },
        "icon": {
            "color": "#ffffff",
            "font-size": "16px",
        },
        "nav-link": {
            "font-size": "15px",
            "font-weight": "600",
            "text-align": "center",
            "margin": "0px 6px",
            "padding": "12px 18px",
            "border-radius": "14px",
            "color": "#FFFFFF",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #1d4ed8, #2563eb)",
            "color": "white",
            "box-shadow": "0 10px 24px rgba(37, 99, 235, 0.20)",
        },
    },
)

if selected == "Beranda":
    st.switch_page("app.py")
elif selected == "Data":
    st.switch_page("pages/data.py")
elif selected == "Visualisasi":
    st.switch_page("pages/visualisasi.py")
elif selected == "Result":
    st.switch_page("pages/result.py")


# =========================================================
# KONTEN PREDIKSI
# =========================================================
st.markdown(
    """
    <div class="section-card prediction-intro-card">
        <div>
            <h2>Prediksi Opini Publik</h2>
            <p>
                Masukkan satu teks opini, kemudian tekan tombol
                <strong>Prediksi</strong>. Sistem akan menjalankan
                Decision Tree, Random Forest, XGBoost, IndoBERT, dan NusaBERT
                secara otomatis.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    st.markdown(
        """
        <div class="input-section-header">
            <div class="input-section-icon">✎</div>
            <div>
                <div class="input-section-title">Teks yang Akan Diprediksi</div>
                <div class="input-section-description">
                    Tuliskan opini publik secara lengkap dan jelas pada kolom berikut.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_text = st.text_area(
        "Teks opini",
        placeholder=(
            "Contoh: Pemerintah perlu membatasi penggunaan media sosial "
            "bagi anak untuk mengurangi risiko perundungan digital."
        ),
        height=220,
        max_chars=1500,
        label_visibility="collapsed",
        key="prediction_input_text",
    )

    helper_column, button_column = st.columns([3.4, 1.6], gap="large")

    with helper_column:
        st.markdown(
            """
            <div class="input-helper">
                <span class="input-helper-dot"></span>
                Lima metode akan diproses sekaligus dan ditampilkan sesuai urutan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with button_column:
        submitted = st.form_submit_button(
            "Prediksi",
            use_container_width=True,
            type="primary",
        )

if submitted:
    clean_text = input_text.strip()

    if not clean_text:
        st.warning(
            "Teks belum diisi. Silakan masukkan teks yang akan diprediksi."
        )
    else:
        prediction_rows = []

        # Spinner bulat akan terus berputar selama seluruh model diproses.
        with st.spinner(
            "Sedang menjalankan prediksi menggunakan seluruh metode..."
        ):
            for model_name in MODEL_ORDER:
                label_result, label_status = run_prediction(
                    model_name=model_name,
                    task_name="Label",
                    text=clean_text,
                )
                ner_result, ner_status = run_prediction(
                    model_name=model_name,
                    task_name="NER",
                    text=clean_text,
                )

                prediction_rows.append(
                    {
                        "Model": model_name,
                        "Multi-Label Classification": label_result,
                        "Named Entity Recognition (NER)": ner_result,
                        "Status Label": label_status,
                        "Status NER": ner_status,
                    }
                )

        st.session_state.prediction_result = pd.DataFrame(prediction_rows)
        st.session_state.prediction_text = clean_text

if "prediction_result" in st.session_state:
    result_dataframe = st.session_state.prediction_result.copy()

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="result-section-header">
            <div>
                <div class="result-section-eyebrow">HASIL ANALISIS</div>
                <div class="result-section-title">Perbandingan Hasil Prediksi</div>
                <div class="result-section-description">
                    Hasil ditampilkan berdasarkan urutan metode yang telah ditentukan.
                </div>
            </div>
            <div class="method-count-badge">5 Metode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="input-preview">
            <div class="input-preview-icon">“</div>
            <div>
                <div class="input-preview-title">Teks yang Diprediksi</div>
                <div class="input-preview-text">{
                    html.escape(st.session_state.prediction_text)
                }</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        prediction_table_html(result_dataframe),
        unsafe_allow_html=True,
    )

    problem_rows = result_dataframe[
        (result_dataframe["Status Label"] != "Berhasil")
        | (result_dataframe["Status NER"] != "Berhasil")
    ]

    if not problem_rows.empty:
        with st.expander("Lihat detail model yang belum berhasil diproses"):
            for _, row in problem_rows.iterrows():
                st.markdown(f"**{row['Model']}**")
                if row["Status Label"] != "Berhasil":
                    st.caption(f"Multi-Label: {row['Status Label']}")
                if row["Status NER"] != "Berhasil":
                    st.caption(f"NER: {row['Status NER']}")


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        section[data-testid="stSidebar"] {display: none !important;}

        .stApp {
            background-color: #EAF4FF;
            background: linear-gradient(135deg, #EAF4FF, #A2D2FF);
        }

        .block-container,
        [data-testid="stMainBlockContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            padding-top: 1.1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        .hero-box {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #60a5fa 100%);
            border-radius: 32px;
            padding: 3rem 3rem;
            color: white;
            box-shadow: 0 20px 48px rgba(37, 99, 235, 0.20);
            border: 1px solid rgba(255,255,255,0.16);
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .hero-box::before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -60px;
            top: -70px;
            background: rgba(255,255,255,0.10);
            border-radius: 50%;
        }

        .hero-box::after {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            left: -40px;
            bottom: -60px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.25;
            max-width: 980px;
            position: relative;
            z-index: 1;
        }

        .section-card {
            background: #F5FEFF;
            border-radius: 24px;
            padding: 24px;
            border: 1px solid rgba(147, 197, 253, 0.4);
            box-shadow: 0 10px 25px rgba(30, 64, 175, 0.06);
        }

        .section-card h2 {
            margin-top: 0;
            color: #1f2937;
            font-size: 2rem;
            font-weight: 800;
        }

        .section-card p {
            color: #374151;
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 0;
        }

        .subsection-title {
            background: linear-gradient(135deg, #dbeafe, #eff6ff);
            border-left: 6px solid #2563eb;
            padding: 14px 18px;
            border-radius: 16px;
            font-size: 1.25rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }

        div[data-testid="stForm"] {
            background: rgba(245, 254, 255, 0.92);
            border: 1px solid rgba(147, 197, 253, 0.5);
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 12px 28px rgba(30, 64, 175, 0.08);
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #1D4ED8, #2563EB);
            color: white;
            border: none;
            border-radius: 14px;
            min-height: 48px;
            font-weight: 700;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
            transform: translateY(-1px);
        }

        .input-preview {
            background: #F8FBFF;
            border: 1px solid #D6E6FF;
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
        }

        .input-preview-title {
            color: #1E3A8A;
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .input-preview-text {
            color: #1E293B;
            font-size: 1rem;
            line-height: 1.7;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .prediction-intro-card {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .prediction-intro-icon {
            width: 58px;
            height: 58px;
            flex: 0 0 58px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            color: #FFFFFF;
            font-size: 1.65rem;
            background: linear-gradient(135deg, #1D4ED8, #60A5FA);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
        }

        .input-section-header {
            display: flex;
            align-items: center;
            gap: 13px;
            margin-bottom: 16px;
        }

        .input-section-icon {
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 13px;
            color: #1D4ED8;
            font-size: 1.25rem;
            font-weight: 800;
            background: #DBEAFE;
        }

        .input-section-title {
            color: #172554;
            font-size: 1.22rem;
            font-weight: 800;
            line-height: 1.3;
        }

        .input-section-description {
            color: #64748B;
            font-size: 0.94rem;
            margin-top: 3px;
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 220px !important;
            padding: 18px 20px !important;
            border: 1.5px solid #BFDBFE !important;
            border-radius: 18px !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
            font-size: 1.03rem !important;
            line-height: 1.75 !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03) !important;
            resize: vertical !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12) !important;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: #94A3B8 !important;
        }

        .input-helper {
            display: flex;
            align-items: center;
            gap: 9px;
            min-height: 48px;
            color: #64748B;
            font-size: 0.93rem;
        }

        .input-helper-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #2563EB;
            box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.10);
        }

        .result-section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 18px 22px;
            margin-bottom: 14px;
            border: 1px solid rgba(147, 197, 253, 0.48);
            border-radius: 22px;
            background: rgba(245, 254, 255, 0.88);
            box-shadow: 0 10px 25px rgba(30, 64, 175, 0.06);
        }

        .result-section-eyebrow {
            color: #2563EB;
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.12em;
        }

        .result-section-title {
            color: #172554;
            font-size: 1.45rem;
            font-weight: 850;
            margin-top: 3px;
        }

        .result-section-description {
            color: #64748B;
            font-size: 0.92rem;
            margin-top: 3px;
        }

        .method-count-badge {
            flex: 0 0 auto;
            padding: 9px 14px;
            border-radius: 999px;
            color: #1D4ED8;
            font-size: 0.88rem;
            font-weight: 800;
            background: #DBEAFE;
            border: 1px solid #BFDBFE;
        }

        .input-preview {
            display: flex;
            gap: 15px;
            align-items: flex-start;
            background: linear-gradient(135deg, #F8FBFF, #EFF6FF);
            border: 1px solid #D6E6FF;
            border-radius: 20px;
            padding: 19px 21px;
            margin-bottom: 16px;
            box-shadow: 0 8px 20px rgba(30, 64, 175, 0.05);
        }

        .input-preview-icon {
            color: #2563EB;
            font-family: Georgia, serif;
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 0.9;
        }

        .prediction-table-card {
            overflow: hidden;
            border: 1px solid #CFE0FF;
            border-radius: 22px;
            background: #FFFFFF;
            box-shadow: 0 16px 34px rgba(30, 64, 175, 0.10);
        }

        .prediction-table-wrapper {
            width: 100%;
            overflow-x: auto;
        }

        .prediction-table {
            width: 100%;
            min-width: 980px;
            border-collapse: separate;
            border-spacing: 0;
            color: #1E293B;
            font-size: 0.93rem;
        }

        .prediction-table thead th {
            padding: 16px 15px;
            color: #FFFFFF;
            text-align: left;
            font-size: 0.91rem;
            font-weight: 800;
            line-height: 1.35;
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            border-right: 1px solid rgba(255, 255, 255, 0.14);
        }

        .prediction-table thead th:last-child {
            border-right: none;
        }

        .prediction-table tbody td {
            padding: 17px 15px;
            vertical-align: top;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            background: #FFFFFF;
        }

        .prediction-table tbody tr:nth-child(even) td {
            background: #F8FBFF;
        }

        .prediction-table tbody tr:hover td {
            background: #EEF6FF;
        }

        .prediction-table tbody tr:last-child td {
            border-bottom: none;
        }

        .prediction-table tbody td:last-child {
            border-right: none;
        }

        .number-column,
        .number-cell {
            width: 54px;
            text-align: center !important;
        }

        .number-cell {
            color: #64748B;
            font-weight: 800;
        }

        .model-cell {
            min-width: 145px;
        }

        .model-name {
            color: #172554;
            font-size: 0.98rem;
            font-weight: 850;
            line-height: 1.4;
        }

        .prediction-cell {
            min-width: 250px;
            line-height: 1.55;
        }

        .result-badge-group {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 9px;
            border: 1px solid #BFDBFE;
            border-radius: 9px;
            color: #1E40AF;
            font-size: 0.78rem;
            font-weight: 750;
            line-height: 1.25;
            background: #EFF6FF;
        }

        .result-empty,
        .result-error {
            display: inline-flex;
            padding: 7px 10px;
            border-radius: 9px;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .result-empty {
            color: #475569;
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
        }

        .result-error {
            color: #B91C1C;
            background: #FEF2F2;
            border: 1px solid #FECACA;
        }

        .status-cell {
            min-width: 140px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.77rem;
            font-weight: 850;
            white-space: nowrap;
        }

        .status-success {
            color: #047857;
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
        }

        .status-warning {
            color: #B45309;
            background: #FFFBEB;
            border: 1px solid #FDE68A;
        }

        .status-error {
            color: #B91C1C;
            background: #FEF2F2;
            border: 1px solid #FECACA;
        }

        /* Spinner loading berbentuk bulat dan berputar */
        div[data-testid="stSpinner"] {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            min-height: 92px;
            margin: 18px 0;
            padding: 18px 22px;
            border: 1px solid rgba(147, 197, 253, 0.55);
            border-radius: 18px;
            background: rgba(245, 254, 255, 0.94);
            box-shadow: 0 12px 28px rgba(30, 64, 175, 0.08);
        }

        div[data-testid="stSpinner"] svg {
            width: 42px !important;
            height: 42px !important;
            color: #2563EB !important;
        }

        div[data-testid="stSpinner"] p {
            margin: 0 !important;
            color: #1E3A8A !important;
            font-size: 1rem !important;
            font-weight: 750 !important;
        }

        @media (max-width: 768px) {
            .hero-box {padding: 2rem 1.5rem;}
            .hero-title {font-size: 1.65rem;}
            .prediction-intro-card {align-items: flex-start;}
            .prediction-intro-icon {width: 48px; height: 48px; flex-basis: 48px;}
            .result-section-header {align-items: flex-start; flex-direction: column;}
            .input-preview {padding: 16px;}
        }
        
        @media (max-width: 768px) {
            .block-container,
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
