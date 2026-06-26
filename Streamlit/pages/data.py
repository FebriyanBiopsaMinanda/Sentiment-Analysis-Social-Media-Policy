import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

st.set_page_config(
    page_title="Social Media Restrictions for Minors",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# HERO SECTION
# =========================
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

# =========================
# OPTION MENU
# =========================
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Data", "Visualisasi", "Result", "Prediction"],
    icons=["house-heart", "database", "bar-chart-line", "clipboard-data", "robot"],
    default_index=1,
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
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    file_path = "../dataset/Kelp2_hasil_anotasi.xlsx"
    df = pd.read_excel(file_path, sheet_name="Gabungan")

    cols_needed = [
        "Text",
        "annotator1", "annotator2",
        "Organisasi_ann1", "Platform_ann1", "Age_Group_ann1", "Policy_ann1", "Digital_Risk_ann1",
        "Organisasi_ann2", "Platform_ann2", "Age_Group_ann2", "Policy_ann2", "Digital_Risk_ann2"
    ]

    df = df[cols_needed].copy()

    for col in cols_needed:
        df[col] = df[col].fillna("").astype(str)

    return df

# =========================
# HELPER
# =========================
def is_filled(val):
    val = str(val).strip().lower()
    return val not in ["", "nan", "none", "-", "[]", "{}"]

def ner_match_any_annotator(row, selected_ner):
    ner_map = {
        "ORGANISASI": [row["Organisasi_ann1"], row["Organisasi_ann2"]],
        "PLATFORM": [row["Platform_ann1"], row["Platform_ann2"]],
        "AGE_GROUP": [row["Age_Group_ann1"], row["Age_Group_ann2"]],
        "POLICY": [row["Policy_ann1"], row["Policy_ann2"]],
        "DIGITAL_RISK": [row["Digital_Risk_ann1"], row["Digital_Risk_ann2"]],
    }

    for ner in selected_ner:
        vals = ner_map.get(ner, [])
        if any(is_filled(v) for v in vals):
            return True
    return False

# =========================
# CONTENT
# =========================
if selected == "Beranda":
    st.switch_page("app.py")

elif selected == "Data":
    df = load_data()

    # =========================
    # SEARCH + FILTER CARD
    # =========================
    with st.container():

        st.markdown(
        """
        <div class='subsection-title'>Pencarian dan Filter</div>
        """,
        unsafe_allow_html=True)

        colmn1, colmn2 = st.columns(2)
        
        with colmn1:
            search_text = st.text_input(
                "Cari Text",
                placeholder="Masukkan kata kunci..."
            )

        with colmn2:
            col1, col2 = st.columns(2)

            label_options = [
                "PENOLAKAN_KEBIJAKAN",
                "DUKUNGAN_KEBIJAKAN",
                "KRITIK_PEMERINTAHAN",
                "NETRAL"
            ]

            ner_options = [
                "ORGANISASI",
                "PLATFORM",
                "AGE_GROUP",
                "POLICY",
                "DIGITAL_RISK"
            ]

        with col1:
            selected_labels = st.multiselect(
                "Filter Label",
                options=label_options,
                placeholder="Pilih label"
            )

        with col2:
            selected_ner = st.multiselect(
                "Filter NER",
                options=ner_options,
                placeholder="Pilih NER"
            )

        st.markdown("</div>", unsafe_allow_html=True)


    # =========================
    # FILTER DATA
    # =========================
    filtered_df = df.copy()

    # search text
    if search_text:
        filtered_df = filtered_df[
            filtered_df["Text"].str.contains(search_text, case=False, na=False)
        ]

    # filter label
    if selected_labels:
        filtered_df = filtered_df[
            filtered_df["annotator1"].isin(selected_labels) |
            filtered_df["annotator2"].isin(selected_labels)
        ]
    if selected_ner:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: ner_match_any_annotator(row, selected_ner), axis=1)
        ]

    filtered_df = filtered_df[["Text"]].copy()
    if len(filtered_df) > 0:
        filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # =========================
    # PAGINATION RESET SAAT FILTER BERUBAH
    # =========================
    filter_signature = (
        search_text,
        tuple(selected_labels),
        tuple(selected_ner)
    )

    if "last_filter_signature" not in st.session_state:
        st.session_state.last_filter_signature = filter_signature

    if st.session_state.last_filter_signature != filter_signature:
        st.session_state.page = 1
        st.session_state.last_filter_signature = filter_signature

    if "page" not in st.session_state:
        st.session_state.page = 1

    # =========================
    # PAGINATION
    # =========================
    rows_per_page = 10
    total_rows = len(filtered_df)
    total_pages = max(1, (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0))

    if st.session_state.page < 1:
        st.session_state.page = 1
    if st.session_state.page > total_pages:
        st.session_state.page = total_pages

    start_idx = (st.session_state.page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    df_page = filtered_df.iloc[start_idx:end_idx].copy()

    # tabel html
    table_html = df_page.to_html(
        index=False,
        classes="custom-table",
        border=0,
        escape=True
    )


    # =========================
    # TABLE + PAGINATION
    # =========================
    st.markdown(
        """
        <div class='subsection-title'>Data Text</div>
        """,
        unsafe_allow_html=True)
   
    with st.container():
        st.markdown(
            f"""
            <div class="data-card">
                <div class="table-wrapper">
                    {table_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='pagination-wrapper-streamlit'>", unsafe_allow_html=True)

        left_space, center_area, right_space = st.columns([1.8, 2.2, 1.8])

        with center_area:
            prev_col, mid_col, next_col = st.columns([1, 1.2, 1])

            with prev_col:
                if st.button("<<", key="prev_page", use_container_width=True):
                    if st.session_state.page > 1:
                        st.session_state.page -= 1
                        st.rerun()

            with mid_col:
                st.markdown(
                    f"<div class='page-number'>{st.session_state.page} / {total_pages}</div>",
                    unsafe_allow_html=True
                )

            with next_col:
                if st.button(">>", key="next_page", use_container_width=True):
                    if st.session_state.page < total_pages:
                        st.session_state.page += 1
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Visualisasi":
    st.switch_page("pages/visualisasi.py")

elif selected == "Result":
    st.switch_page("pages/result.py")

elif selected == "Prediction":
    st.switch_page("pages/prediction.py")

# =========================
# CUSTOM CSS
# =========================
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
            margin-bottom: 0.9rem;
            max-width: 980px;
            position: relative;
            z-index: 1;
        }
        
        .filter-title {
            font-size: 50px;
            font-weight: 800;
            color: #000249;
            padding-top: 8px;
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
        

        .table-wrapper {
            background: #F5FEFF;
            border-radius: 18px;
            overflow-x: auto;
            border: 1px solid #D6E6FF;
        }

        .custom-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
        }

        .custom-table thead th {
            background: linear-gradient(135deg, #1D4ED8, #2563EB);
            color: white !important;
            text-align: center;
            font-weight: 700;
            font-size: 16px;
        }

        .custom-table tbody td {
            background: #F8FBFF;
            color: #1E293B !important;
            border-bottom: 1px solid #D9E6FF;
            vertical-align: top;
            text-align: left;
            white-space: normal;
            word-break: break-word;
            line-height: 1.8;
        }

        .custom-table tbody tr:nth-child(even) td {
            background: #EEF5FF;
        }

        .custom-table tbody tr:hover td {
            background: #E0EEFF;
        }

        .pagination-wrapper-streamlit {
            padding-top: 12px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1D4ED8, #2563EB);
            color: white;
            border: none;
            border-radius: 14px;
            font-weight: 700;
            height: 48px;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.20);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1E40AF, #1D4ED8);
            color: white;
        }

        .page-number {
            text-align: center;
            font-size: 26px;
            font-weight: 800;
            color: #1D4ED8;
            padding-top: 8px;
        }

        .stTextInput > div > div > input {
            border-radius: 14px;
        }

        .stMultiSelect > div > div {
            border-radius: 14px;
        }
        
        div[data-baseweb="input"] > div {
            background-color: #1e293b !important;  
            border: 2px solid #0f172a !important; 
            border-radius: 14px !important;
        }

        div[data-baseweb="input"] input {
            color: #ffffff !important;          
            font-size: 15px !important;
        }

        div[data-baseweb="input"] input::placeholder {
            color: #cbd5e1 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #1e293b !important;  
            border: 2px solid #0f172a !important; 
            border-radius: 14px !important;
            color: #ffffff !important;
        }

        div[data-baseweb="select"] span {
            color: #ffffff !important;
        }
        ul[role="listbox"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
        }

        li[role="option"] {
            color: #1e293b !important;           
            background-color: #ffffff !important;
        }

        li[role="option"]:hover {
            background-color: #e2e8f0 !important;
        }

        div[data-baseweb="tag"] {
            background-color: #1d4ed8 !important;
            color: white !important;
            border-radius: 8px !important;
        }

        .stTextInput label,
        .stMultiSelect label {
            color: #1e293b !important;
            font-size: 50px !important;
            font-weight: 800 !important;
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