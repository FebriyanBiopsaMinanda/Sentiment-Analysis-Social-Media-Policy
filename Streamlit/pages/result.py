from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
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

CLASSICAL_MODELS = [
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]

DEEP_MODELS = [
    "IndoBERT",
    "NusaBERT",
]

METRIC_COLUMNS = [
    "Accuracy",
    "Hamming Loss",
    "Precision",
    "F1 Score",
]

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

FOLDER_MAP = {
    "Label": [
        ("Classical ML", Path("Classical ML") / "Multi Label"),
        ("Deep Sequence", Path("Deep Sequence") / "Multi Label"),
    ],
    "NER": [
        ("Classical ML", Path("Classical ML") / "Multi NER"),
        ("Deep Sequence", Path("Deep Sequence") / "Multi NER"),
    ],
}

TASK_DISPLAY_NAMES = {
    "Label": "Multi-Label",
    "NER": "Multi-NER",
}

METRIC_ALIASES = {
    "Accuracy": [
        "accuracy",
        "accuracy_score",
        "acc",
        "subset_accuracy",
        "exact_match_ratio",
        "exact_match",
    ],
    "Hamming Loss": [
        "hamming_loss",
        "hammingloss",
        "hamming loss",
    ],
    "Precision": [
        "precision",
        "precision_score",
        "weighted_precision",
        "micro_precision",
        "macro_precision",
        "precision_weighted",
        "precision_micro",
        "precision_macro",
    ],
    "F1 Score": [
        "f1_score",
        "f1 score",
        "f1",
        "f1score",
        "f1-score",
        "weighted_f1",
        "micro_f1",
        "macro_f1",
        "f1_weighted",
        "f1_micro",
        "f1_macro",
    ],
}

MODEL_FIELD_ALIASES = {
    "model",
    "model_name",
    "nama_model",
    "algorithm",
    "algoritma",
    "classifier",
    "architecture",
}


# =========================================================
# PATH
# =========================================================
def find_outputs_root() -> Path:
    page_file = Path(__file__).resolve()

    candidates = [
        page_file.parents[2] / "Outputs",
        page_file.parents[1] / "Outputs",
        Path.cwd() / "Outputs",
        Path.cwd().parent / "Outputs",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    return (page_file.parents[2] / "Outputs").resolve()


def find_images_root() -> Path:
    page_file = Path(__file__).resolve()

    candidates = [
        page_file.parents[2] / "images",
        page_file.parents[1] / "images",
        Path.cwd() / "images",
        Path.cwd().parent / "images",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    return (page_file.parents[2] / "images").resolve()


OUTPUTS_ROOT = find_outputs_root()
IMAGES_ROOT = find_images_root()


# =========================================================
# NORMALISASI MODEL DAN METRIK
# =========================================================
def normalize_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_model_name(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)

    if "decisiontree" in compact or compact in {"dt", "cart"}:
        return "Decision Tree"

    if "randomforest" in compact or compact in {
        "rf",
        "randomforestclassifier",
    }:
        return "Random Forest"

    if "xgboost" in compact or compact.startswith("xgb"):
        return "XGBoost"

    if "indobert" in compact:
        return "IndoBERT"

    if "nusabert" in compact or (
        "nusa" in compact and "bert" in compact
    ):
        return "NusaBERT"

    return None


def infer_model_name(*values: Any) -> str | None:
    for value in values:
        model = canonical_model_name(value)
        if model:
            return model
    return None


def normalize_metric_value(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        is_percent = cleaned.endswith("%")
        cleaned = cleaned.rstrip("%").strip()

        try:
            number = float(cleaned)
        except ValueError:
            return None

        if is_percent:
            number /= 100
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

    if 1 < number <= 100:
        number /= 100

    return number


def pick_metric(
    mapping: dict[str, Any],
    metric_name: str,
) -> float | None:
    normalized_mapping = {
        normalize_key(key): value
        for key, value in mapping.items()
    }

    for alias in METRIC_ALIASES[metric_name]:
        alias_key = normalize_key(alias)

        if alias_key in normalized_mapping:
            value = normalize_metric_value(
                normalized_mapping[alias_key]
            )

            if value is not None:
                return value

    return None


def mapping_has_metric(mapping: dict[str, Any]) -> bool:
    return any(
        pick_metric(mapping, metric) is not None
        for metric in METRIC_COLUMNS
    )


def model_from_mapping(
    mapping: dict[str, Any],
) -> str | None:
    normalized_mapping = {
        normalize_key(key): value
        for key, value in mapping.items()
    }

    for alias in MODEL_FIELD_ALIASES:
        alias_key = normalize_key(alias)

        if alias_key in normalized_mapping:
            model = infer_model_name(
                normalized_mapping[alias_key]
            )

            if model:
                return model

    return None


def build_record(
    mapping: dict[str, Any],
    fallback_model: str | None,
    file_path: Path,
) -> dict[str, Any] | None:
    model = model_from_mapping(mapping) or fallback_model

    if not model:
        model = infer_model_name(
            file_path.stem,
            file_path.parent.name,
            str(file_path),
        )

    if not model or not mapping_has_metric(mapping):
        return None

    record: dict[str, Any] = {
        "Model": model,
        "Sumber": file_path.name,
        "Modified": file_path.stat().st_mtime,
    }

    for metric in METRIC_COLUMNS:
        record[metric] = pick_metric(mapping, metric)

    return record


# =========================================================
# PEMBACA FILE HASIL EVALUASI
# =========================================================
def extract_json_records(
    obj: Any,
    file_path: Path,
    parent_hint: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        fallback_model = infer_model_name(
            parent_hint,
            file_path.stem,
            str(file_path),
        )

        record = build_record(
            mapping=obj,
            fallback_model=fallback_model,
            file_path=file_path,
        )

        if record:
            records.append(record)

        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                records.extend(
                    extract_json_records(
                        obj=value,
                        file_path=file_path,
                        parent_hint=str(key),
                    )
                )

    elif isinstance(obj, list):
        for item in obj:
            records.extend(
                extract_json_records(
                    obj=item,
                    file_path=file_path,
                    parent_hint=parent_hint,
                )
            )

    return records


def records_from_dataframe(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []

    records: list[dict[str, Any]] = []

    fallback_model = infer_model_name(
        file_path.stem,
        file_path.parent.name,
        str(file_path),
    )

    for _, row in dataframe.iterrows():
        record = build_record(
            mapping=row.to_dict(),
            fallback_model=fallback_model,
            file_path=file_path,
        )

        if record:
            records.append(record)

    # Format alternatif:
    # kolom pertama berisi nama metrik dan kolom berikutnya nama model.
    if not records and len(dataframe.columns) > 1:
        first_column = dataframe.columns[0]

        metric_names = (
            dataframe[first_column]
            .astype(str)
            .map(normalize_key)
        )

        known_aliases = {
            normalize_key(alias)
            for aliases in METRIC_ALIASES.values()
            for alias in aliases
        }

        if metric_names.isin(known_aliases).any():
            indexed_dataframe = dataframe.set_index(first_column)

            for model_column in indexed_dataframe.columns:
                model = infer_model_name(model_column)

                if not model:
                    continue

                mapping = {
                    str(metric_index): indexed_dataframe.loc[
                        metric_index,
                        model_column,
                    ]
                    for metric_index in indexed_dataframe.index
                }

                record = build_record(
                    mapping=mapping,
                    fallback_model=model,
                    file_path=file_path,
                )

                if record:
                    records.append(record)

    return records


def read_metric_file(
    file_path: Path,
) -> list[dict[str, Any]]:
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            return records_from_dataframe(
                pd.read_csv(file_path),
                file_path,
            )

        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(
                file_path,
                sheet_name=None,
            )

            records: list[dict[str, Any]] = []

            for sheet_dataframe in sheets.values():
                records.extend(
                    records_from_dataframe(
                        sheet_dataframe,
                        file_path,
                    )
                )

            return records

        if suffix == ".json":
            with file_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                json_data = json.load(handle)

            return extract_json_records(
                json_data,
                file_path,
            )

        if suffix == ".jsonl":
            records: list[dict[str, Any]] = []

            with file_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    line = line.strip()

                    if not line:
                        continue

                    records.extend(
                        extract_json_records(
                            json.loads(line),
                            file_path,
                        )
                    )

            return records

    except Exception as error:
        return [
            {
                "Model": None,
                "Sumber": file_path.name,
                "Modified": file_path.stat().st_mtime,
                "Error": str(error),
            }
        ]

    return []


def combine_model_records(
    records: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    valid_records = [
        record
        for record in records
        if record.get("Model")
    ]

    if not valid_records:
        return pd.DataFrame(
            columns=[
                "Model",
                *METRIC_COLUMNS,
            ]
        )

    raw_dataframe = pd.DataFrame(
        valid_records
    ).sort_values(
        "Modified",
        ascending=False,
    )

    rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        model_rows = raw_dataframe[
            raw_dataframe["Model"] == model
        ]

        if model_rows.empty:
            continue

        combined: dict[str, Any] = {
            "Model": model,
        }

        for metric in METRIC_COLUMNS:
            if metric in model_rows.columns:
                values = model_rows[metric].dropna()
            else:
                values = pd.Series(dtype=float)

            combined[metric] = (
                values.iloc[0]
                if not values.empty
                else None
            )

        rows.append(combined)

    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            *METRIC_COLUMNS,
        ],
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_task_results(
    task_name: str,
    outputs_root_string: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    outputs_root = Path(outputs_root_string)

    all_records: list[dict[str, Any]] = []
    missing_folders: list[str] = []
    read_errors: list[str] = []

    for _, relative_folder in FOLDER_MAP[task_name]:
        folder = outputs_root / relative_folder

        if not folder.exists():
            missing_folders.append(str(folder))
            continue

        files = sorted(
            (
                path
                for path in folder.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for file_path in files:
            file_records = read_metric_file(file_path)

            for record in file_records:
                if record.get("Error"):
                    read_errors.append(
                        f"{file_path.name}: "
                        f"{record['Error']}"
                    )
                else:
                    all_records.append(record)

    result = combine_model_records(all_records)

    return result, missing_folders, read_errors


def complete_expected_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=[
                "Model",
                *METRIC_COLUMNS,
            ]
        )

    return (
        dataframe
        .set_index("Model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )


# =========================================================
# TABEL HASIL
# =========================================================
def display_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value) * 100:.2f}%"


def dataframe_to_html(
    dataframe: pd.DataFrame,
) -> str:
    display_dataframe = dataframe[
        [
            "Model",
            "Accuracy",
            "Hamming Loss",
            "Precision",
            "F1 Score",
        ]
    ].copy()

    for metric in METRIC_COLUMNS:
        display_dataframe[metric] = (
            display_dataframe[metric]
            .map(display_metric)
        )

    return display_dataframe.to_html(
        index=False,
        border=0,
        classes="result-table",
        escape=True,
    )


def render_result_table(
    title: str,
    subtitle: str,
    dataframe: pd.DataFrame,
) -> None:
    table_html = dataframe_to_html(dataframe)

    html_content = (
        '<div class="result-card">'
        '<div class="result-card-header">'
        f'<div class="result-card-title">{title}</div>'
        f'<div class="result-card-subtitle">{subtitle}</div>'
        '</div>'
        '<div class="result-table-wrapper">'
        f"{table_html}"
        "</div>"
        "</div>"
    )

    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(
            html_content,
            unsafe_allow_html=True,
        )


# =========================================================
# CONFUSION MATRIX BERBASIS GAMBAR
# =========================================================
def normalize_image_stem(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def list_image_files(
    images_root_string: str,
) -> list[str]:
    images_root = Path(images_root_string)

    if not images_root.exists():
        return []

    return [
        str(path)
        for path in images_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


def find_confusion_image(
    task_name: str,
    model_name: str,
    detail: bool,
) -> Path | None:
    task_display = TASK_DISPLAY_NAMES[task_name]

    if detail:
        expected_name = (
            f"Detail Confusion Matrix "
            f"{task_display} - {model_name}"
        )
    else:
        expected_name = (
            f"Confusion Matrix "
            f"{task_display} - {model_name}"
        )

    expected_normalized = normalize_image_stem(
        expected_name
    )

    image_paths = [
        Path(path)
        for path in list_image_files(
            str(IMAGES_ROOT)
        )
    ]

    # Pencarian nama yang sama persis setelah dinormalisasi.
    for image_path in image_paths:
        if (
            normalize_image_stem(image_path.stem)
            == expected_normalized
        ):
            return image_path

    # Pencarian cadangan untuk perbedaan kecil pada nama file.
    for image_path in image_paths:
        normalized_stem = normalize_image_stem(
            image_path.stem
        )

        if (
            expected_normalized in normalized_stem
            or normalized_stem in expected_normalized
        ):
            return image_path

    return None


# Ukuran seragam untuk gambar overview sebelum tombol Detail.
OVERVIEW_IMAGE_SIZE = (1000, 750)

# Jumlah piksel bagian atas yang dipotong setelah ukuran diseragamkan.
OVERVIEW_CROP_TOP = 30

DETAIL_CROP_RATIO = 0.06


def resize_image_to_same_size(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    target_width, target_height = target_size

    resized_image = ImageOps.contain(
        image,
        target_size,
        method=Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        mode="RGB",
        size=target_size,
        color="white",
    )

    paste_x = (
        target_width - resized_image.width
    ) // 2

    paste_y = (
        target_height - resized_image.height
    ) // 2

    canvas.paste(
        resized_image,
        (paste_x, paste_y),
    )

    return canvas


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_cropped_image(
    image_path_string: str,
    modified_time: float,
    detail: bool,
) -> bytes:
    del modified_time

    image_path = Path(image_path_string)

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

    if detail:
        # Gambar detail mempertahankan ukuran aslinya.
        # Hanya judul bawaan gambar yang dipotong.
        width, height = image.size

        crop_top = max(
            0,
            min(
                int(height * DETAIL_CROP_RATIO),
                height - 1,
            ),
        )

        cropped_image = image.crop(
            (
                0,
                crop_top,
                width,
                height,
            )
        )

    else:
        # Gambar overview:
        # 1. Diseragamkan ukurannya.
        # 2. Dipotong bagian atasnya setelah ukuran sama.
        same_size_image = resize_image_to_same_size(
            image=image,
            target_size=OVERVIEW_IMAGE_SIZE,
        )

        width, height = same_size_image.size

        crop_top = max(
            0,
            min(
                OVERVIEW_CROP_TOP,
                height - 1,
            ),
        )

        cropped_image = same_size_image.crop(
            (
                0,
                crop_top,
                width,
                height,
            )
        )

    buffer = BytesIO()

    cropped_image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


def get_display_image(
    image_path: Path,
    detail: bool,
) -> bytes:
    return load_cropped_image(
        image_path_string=str(image_path),
        modified_time=image_path.stat().st_mtime,
        detail=detail,
    )


if "confusion_detail_image" not in st.session_state:
    st.session_state.confusion_detail_image = None


def open_confusion_detail(
    task_name: str,
    model_name: str,
) -> None:
    st.session_state.confusion_detail_image = {
        "task": task_name,
        "model": model_name,
    }


def close_confusion_detail() -> None:
    st.session_state.confusion_detail_image = None


def render_confusion_card(
    task_name: str,
    model_name: str,
) -> None:
    task_display = TASK_DISPLAY_NAMES[task_name]

    overview_path = find_confusion_image(
        task_name=task_name,
        model_name=model_name,
        detail=False,
    )

    detail_path = find_confusion_image(
        task_name=task_name,
        model_name=model_name,
        detail=True,
    )

    with st.container(border=True):
        st.markdown(
            (
                '<div class="confusion-card-title">'
                f"{model_name}"
                "</div>"
                '<div class="confusion-card-subtitle">'
                f"Confusion Matrix {task_display}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if overview_path is not None:
            st.image(
                get_display_image(
                    image_path=overview_path,
                    detail=False,
                ),
                use_container_width=True,
            )
        else:
            st.markdown(
                (
                    '<div class="missing-image-box">'
                    "Gambar belum ditemukan"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        st.button(
            "Detail",
            key=(
                f"detail_confusion_"
                f"{task_name}_"
                f"{model_name}"
            ),
            use_container_width=True,
            disabled=detail_path is None,
            on_click=open_confusion_detail,
            args=(
                task_name,
                model_name,
            ),
        )


def render_confusion_row(
    task_name: str,
) -> None:
    task_display = TASK_DISPLAY_NAMES[task_name]

    st.markdown(
        (
            '<div class="confusion-section-header">'
            '<div class="confusion-section-title">'
            f"Confusion Matrix {task_display}"
            "</div>"
            '<div class="confusion-section-description">'
            "Decision Tree, Random Forest, XGBoost, "
            "IndoBERT, dan NusaBERT"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    algorithm_columns = st.columns(
        5,
        gap="small",
    )

    for column, model_name in zip(
        algorithm_columns,
        MODEL_ORDER,
    ):
        with column:
            render_confusion_card(
                task_name=task_name,
                model_name=model_name,
            )


@st.dialog(
    "Detail Confusion Matrix",
    width="large",
)
def show_confusion_detail(
    task_name: str,
    model_name: str,
) -> None:
    task_display = TASK_DISPLAY_NAMES[task_name]

    st.markdown(
        (
            '<div class="confusion-dialog-title">'
            f"Detail Confusion Matrix "
            f"{task_display} - {model_name}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    detail_path = find_confusion_image(
        task_name=task_name,
        model_name=model_name,
        detail=True,
    )

    if detail_path is None:
        st.warning(
            "Gambar detail confusion matrix "
            "tidak ditemukan."
        )
    else:
        st.image(
            get_display_image(
                image_path=detail_path,
                detail=True,
            ),
            use_container_width=True,
        )

    close_left, close_center, close_right = (
        st.columns([1, 1, 1])
    )

    with close_center:
        if st.button(
            "Tutup",
            key=(
                f"close_confusion_"
                f"{task_name}_"
                f"{model_name}"
            ),
            use_container_width=True,
        ):
            close_confusion_detail()
            st.rerun()


# =========================================================
# HERO DAN NAVIGASI
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">
            ANALISIS OPINI PUBLIK TERHADAP KEBIJAKAN
            PEMBATASAN MEDIA SOSIAL ANAK
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title=None,
    options=["Beranda", "Data", "Visualisasi", "Result", "Prediction"],
    icons=["house-heart", "database", "bar-chart-line", "clipboard-data", "robot"],
    default_index=3,
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
            "background": (
                "linear-gradient("
                "135deg, #1d4ed8, #2563eb)"
            ),
            "color": "white",
            "box-shadow": (
                "0 10px 24px "
                "rgba(37, 99, 235, 0.20)"
            ),
        },
    },
)

if selected == "Beranda":
    st.switch_page("app.py")

elif selected == "Data":
    st.switch_page("pages/data.py")

elif selected == "Visualisasi":
    st.switch_page("pages/visualisasi.py")

elif selected == "Prediction":
    st.switch_page("pages/prediction.py")
    
# =========================================================
# KONTEN RESULT
# =========================================================
st.markdown(
    """
    <div class="section-card intro-card">
        <h2>Hasil Evaluasi Model</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='height:18px'></div>",
    unsafe_allow_html=True,
)

label_dataframe, label_missing, label_errors = (
    load_task_results(
        task_name="Label",
        outputs_root_string=str(OUTPUTS_ROOT),
    )
)

ner_dataframe, ner_missing, ner_errors = (
    load_task_results(
        task_name="NER",
        outputs_root_string=str(OUTPUTS_ROOT),
    )
)

label_dataframe = complete_expected_rows(
    label_dataframe
)

ner_dataframe = complete_expected_rows(
    ner_dataframe
)

left_table, right_table = st.columns(
    2,
    gap="large",
)

with left_table:
    render_result_table(
        title="LABEL",
        subtitle="Multi Label Classification",
        dataframe=label_dataframe,
    )

with right_table:
    render_result_table(
        title="NER",
        subtitle="Named Entity Recognition",
        dataframe=ner_dataframe,
    )


# =========================================================
# CONFUSION MATRIX
# =========================================================
st.markdown(
    "<div style='height:32px'></div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card intro-card">
        <h2>Confusion Matrix</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='height:18px'></div>",
    unsafe_allow_html=True,
)

render_confusion_row("Label")

st.markdown(
    "<div style='height:28px'></div>",
    unsafe_allow_html=True,
)

render_confusion_row("NER")


selected_confusion = (
    st.session_state.confusion_detail_image
)

if selected_confusion is not None:
    show_confusion_detail(
        task_name=selected_confusion["task"],
        model_name=selected_confusion["model"],
    )


# =========================================================
# INFORMASI FILE
# =========================================================
missing_folders = (
    label_missing
    + ner_missing
)

read_errors = (
    label_errors
    + ner_errors
)

if missing_folders or read_errors:
    with st.expander(
        "Informasi pembacaan file"
    ):
        if missing_folders:
            missing_text = "\n".join(
                f"- `{folder}`"
                for folder in sorted(
                    set(missing_folders)
                )
            )

            st.warning(
                "Folder berikut belum ditemukan:"
                f"\n\n{missing_text}"
            )

        if read_errors:
            error_text = "\n".join(
                f"- {error}"
                for error in sorted(
                    set(read_errors)
                )
            )

            st.error(
                "Beberapa file tidak dapat dibaca:"
                f"\n\n{error_text}"
            )


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        .stApp {
            background-color: #EAF4FF;
            background:
                linear-gradient(
                    135deg,
                    #EAF4FF,
                    #A2D2FF
                );
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1800px;
        }

        .hero-box {
            background:
                linear-gradient(
                    135deg,
                    #1e3a8a 0%,
                    #2563eb 55%,
                    #60a5fa 100%
                );
            border-radius: 32px;
            padding: 3rem;
            color: white;
            box-shadow:
                0 20px 48px
                rgba(37, 99, 235, 0.20);
            border:
                1px solid
                rgba(255,255,255,0.16);
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
            background:
                rgba(255,255,255,0.10);
            border-radius: 50%;
        }

        .hero-box::after {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            left: -40px;
            bottom: -60px;
            background:
                rgba(255,255,255,0.08);
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
            background:
                rgba(245, 254, 255, 0.95);
            border-radius: 24px;
            padding: 22px 24px;
            border:
                1px solid
                rgba(147, 197, 253, 0.45);
            box-shadow:
                0 14px 30px
                rgba(37, 99, 235, 0.08);
        }

        .intro-card h2 {
            margin: 0;
            color: #1e3a8a;
            font-size: 1.8rem;
            font-weight: 800;
        }

        .result-card {
            background:
                rgba(255,255,255,0.95);
            border-radius: 24px;
            border:
                1px solid
                rgba(147, 197, 253, 0.55);
            box-shadow:
                0 14px 32px
                rgba(37, 99, 235, 0.12);
            padding: 14px;
            min-height: 420px;
        }

        .result-card-header {
            background:
                linear-gradient(
                    135deg,
                    #1e3a8a,
                    #2563eb
                );
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
            text-align: center;
            color: white;
        }

        .result-card-title {
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }

        .result-card-subtitle {
            margin-top: 4px;
            font-size: 0.92rem;
            color: #dbeafe;
        }

        .result-table-wrapper {
            width: 100%;
            overflow-x: auto;
            border-radius: 16px;
            border: 1px solid #dbeafe;
            background: white;
        }

        table.result-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.78rem;
            margin: 0;
        }

        table.result-table thead th {
            background: #dbeafe;
            color: #1e3a8a !important;
            padding: 12px 5px;
            text-align: center;
            font-weight: 800;
            border-bottom: 2px solid #93c5fd;
            white-space: normal;
            overflow-wrap: break-word;
            line-height: 1.25;
        }

        table.result-table tbody td {
            background: #ffffff;
            color: #1e293b !important;
            padding: 13px 5px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
            white-space: normal;
            overflow-wrap: break-word;
            line-height: 1.25;
        }

        table.result-table tbody tr:nth-child(even) td {
            background: #f4f8ff;
        }

        table.result-table tbody tr:hover td {
            background: #e8f1ff;
        }

        table.result-table th:first-child,
        table.result-table td:first-child {
            width: 25%;
            text-align: left;
            padding-left: 8px;
        }

        table.result-table th:nth-child(2),
        table.result-table td:nth-child(2),
        table.result-table th:nth-child(3),
        table.result-table td:nth-child(3),
        table.result-table th:nth-child(4),
        table.result-table td:nth-child(4),
        table.result-table th:nth-child(5),
        table.result-table td:nth-child(5) {
            width: 18.75%;
        }

        table.result-table tbody td:first-child {
            font-weight: 800;
            color: #1d4ed8 !important;
        }

        .confusion-section-header {
            background:
                linear-gradient(
                    135deg,
                    #eff6ff,
                    #dbeafe
                );
            border-left: 6px solid #2563eb;
            border-radius: 16px;
            padding: 14px 18px;
            margin-bottom: 16px;
            box-shadow:
                0 8px 20px
                rgba(37, 99, 235, 0.08);
        }

        .confusion-section-title {
            color: #1e3a8a;
            font-size: 1.25rem;
            font-weight: 900;
        }

        .confusion-section-description {
            color: #475569;
            font-size: 0.88rem;
            margin-top: 3px;
        }

        .confusion-card-title {
            color: #1e3a8a;
            text-align: center;
            font-size: 1rem;
            font-weight: 900;
            min-height: 26px;
        }

        .confusion-card-subtitle {
            color: #64748b;
            text-align: center;
            font-size: 0.73rem;
            font-weight: 600;
            min-height: 32px;
            margin-bottom: 8px;
        }

        .missing-image-box {
            min-height: 180px;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: #f8fafc;
            color: #dc2626;
            border: 1px dashed #fca5a5;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .confusion-dialog-title {
            background:
                linear-gradient(
                    135deg,
                    #1e3a8a,
                    #2563eb
                );
            color: white;
            padding: 16px 20px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 900;
            margin-bottom: 20px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:
                rgba(255, 255, 255, 0.94);
            border:
                1px solid
                rgba(147, 197, 253, 0.55);
            border-radius: 18px;
            box-shadow:
                0 8px 20px
                rgba(37, 99, 235, 0.10);
        }

        div.stButton > button {
            border-radius: 14px;
            min-height: 44px;
            font-weight: 700;
            border: none;
            background:
                linear-gradient(
                    135deg,
                    #2563eb,
                    #3b82f6
                );
            color: white;
            box-shadow:
                0 8px 22px
                rgba(37, 99, 235, 0.22);
        }

        div.stButton > button:hover {
            background:
                linear-gradient(
                    135deg,
                    #1d4ed8,
                    #2563eb
                );
            color: white;
        }

        div[data-testid="stDialog"] > div {
            border-radius: 24px !important;
            border:
                1px solid
                rgba(147, 197, 253, 0.55)
                !important;
            box-shadow:
                0 30px 80px
                rgba(15, 23, 42, 0.30)
                !important;
        }

        div[data-testid="stDialog"]
        div.stButton > button {
            background:
                linear-gradient(
                    135deg,
                    #dc2626,
                    #ef4444
                ) !important;
            color: white !important;
            border: none !important;
            border-radius: 999px !important;
        }

        div[data-testid="stDialog"]
        div.stButton > button:hover {
            background:
                linear-gradient(
                    135deg,
                    #b91c1c,
                    #dc2626
                ) !important;
        }

        @media (max-width: 1200px) {
            .block-container {
                max-width: 100%;
            }
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.65rem;
            }

            .hero-box {
                padding: 2rem;
            }

            .result-card {
                min-height: auto;
            }

            .result-table-wrapper {
                overflow-x: auto;
            }

            table.result-table {
                min-width: 600px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
