import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64
from wordcloud import WordCloud
import re
import json

st.set_page_config(
    page_title="Social Media Restrictions for Minors",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# HELPER
# =========================================================
def safe_count_entities(df, cols):
    temp = df[cols].copy()
    temp = temp.fillna("-").astype(str)

    for col in temp.columns:
        temp[col] = temp[col].str.strip().str.lower()

    return ((temp != "-") & (temp != "") & (temp != "nan")).sum()

def open_detail(detail_type):
    st.session_state.detail_type = detail_type

def close_detail():
    st.session_state.detail_type = None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

def render_plot_card(title, fig):
    img_base64 = fig_to_base64(fig)
    st.markdown(
        f"""
        <div class="plot-card">
            <div class="viz-title">{title}</div>
            <div class="plot-area">
                <img src="data:image/png;base64,{img_base64}" class="plot-img"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def clean_text_for_wc(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def generate_wordcloud(text, colormap="Set2"):
    wc = WordCloud(
        width=1200,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=100,
        collocations=False
    ).generate(text if text.strip() else "data kosong")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def get_text_by_label(df, label_name):
    filtered = df[
        (df["annotator1"].str.lower() == label_name.lower()) |
        (df["annotator2"].str.lower() == label_name.lower())
    ]
    return " ".join(filtered["Text"].dropna().astype(str).tolist())

def get_text_by_entity(df, col_ann1, col_ann2):
    vals1 = df[col_ann1].fillna("-").astype(str)
    vals2 = df[col_ann2].fillna("-").astype(str)

    entities = []
    entities.extend(vals1[(vals1 != "-") & (vals1.str.lower() != "nan") & (vals1.str.strip() != "")].tolist())
    entities.extend(vals2[(vals2 != "-") & (vals2.str.lower() != "nan") & (vals2.str.strip() != "")].tolist())

    return " ".join(entities)

if "detail_type" not in st.session_state:
    st.session_state.detail_type = None

# =========================================================
# HERO SECTION
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

# =========================================================
# OPTION MENU
# =========================================================
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Data", "Visualisasi", "Result", "Prediction"],
    icons=["house-heart", "database", "bar-chart-line", "clipboard-data", "robot"],
    default_index=2,
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

# =========================================================
# CONTENT
# =========================================================
if selected == "Beranda":
    st.switch_page("app.py")

elif selected == "Data":
    st.switch_page("pages/data.py")

elif selected == "Result":
    st.switch_page("pages/result.py")

elif selected == "Prediction":
    st.switch_page("pages/prediction.py")

elif selected == "Visualisasi":

    # =========================================================
    # LOAD DATA
    # =========================================================
    file_path = "../dataset/Kelp2_hasil_anotasi.xlsx"
    iaaa_path = "../dataset/Kelp2_iaaa.jsonl"
    iaaa_text_path = "../dataset/Kelp2_iaaa_textcat.jsonl"
    
    df_hasil_anotasi = pd.read_excel(file_path, sheet_name="Gabungan")
    
    with open(iaaa_path, "r") as file:
        iaaa_data = json.load(file)
        
    with open(iaaa_text_path, "r") as file:
        iaaa_text = json.load(file)

    df_hasil_anotasi["Text"] = df_hasil_anotasi["Text"].fillna("").astype(str)
    df_hasil_anotasi["annotator1"] = df_hasil_anotasi["annotator1"].fillna("Tidak Ada Label").astype(str)
    df_hasil_anotasi["annotator2"] = df_hasil_anotasi["annotator2"].fillna("Tidak Ada Label").astype(str)

    # =========================================================
    # FITUR EDA
    # =========================================================
    df_hasil_anotasi["panjang_karakter"] = df_hasil_anotasi["Text"].apply(len)
    df_hasil_anotasi["jumlah_kata"] = df_hasil_anotasi["Text"].apply(lambda x: len(x.split()))

    # =========================================================
    # DISTRIBUSI LABEL
    # =========================================================
    count_annotator1 = df_hasil_anotasi["annotator1"].value_counts()
    count_annotator2 = df_hasil_anotasi["annotator2"].value_counts()

    all_label_index = sorted(set(count_annotator1.index).union(set(count_annotator2.index)))

    df_compare_label = pd.DataFrame({
        "Annotator 1": count_annotator1.reindex(all_label_index, fill_value=0),
        "Annotator 2": count_annotator2.reindex(all_label_index, fill_value=0)
    })

    # =========================================================
    # DISTRIBUSI NER
    # =========================================================
    ner_cols_ann1 = ["Organisasi_ann1", "Platform_ann1", "Age_Group_ann1", "Policy_ann1", "Digital_Risk_ann1"]
    ner_cols_ann2 = ["Organisasi_ann2", "Platform_ann2", "Age_Group_ann2", "Policy_ann2", "Digital_Risk_ann2"]

    count_ann1 = safe_count_entities(df_hasil_anotasi, ner_cols_ann1)
    count_ann2 = safe_count_entities(df_hasil_anotasi, ner_cols_ann2)

    entity_names = ["Organisasi", "Platform", "Age_Group", "Policy", "Digital_Risk"]

    df_jumlah_entitas = pd.DataFrame({
        "Annotator 1": count_ann1.values,
        "Annotator 2": count_ann2.values
    }, index=entity_names)

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    y_true = df_hasil_anotasi["annotator1"]
    y_pred = df_hasil_anotasi["annotator2"]
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # =========================================================
    # DIALOG DETAIL
    # =========================================================
    @st.dialog("Detail Visualisasi", width="large")
    def show_detail_dialog(detail_type):
        colors = ["#6366F1", "#22C55E", "#F59E0B", "#EF4444", "#14B8A6"]
        
        if detail_type == "label":
            st.markdown("### Detail Distribusi Label per Annotator")
            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots(figsize=(7, 4))
                count_annotator1.reindex(all_label_index, fill_value=0).plot(kind="bar", ax=ax, color=colors)
                ax.set_title("Annotator 1")
                ax.set_xlabel("Label")
                ax.set_ylabel("Jumlah")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with c2:
                fig, ax = plt.subplots(figsize=(7, 4))
                count_annotator2.reindex(all_label_index, fill_value=0).plot(kind="bar", ax=ax, color=colors)
                ax.set_title("Annotator 2")
                ax.set_xlabel("Label")
                ax.set_ylabel("Jumlah")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        elif detail_type == "ner":
            st.markdown("### Detail Distribusi NER per Annotator")

            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots(figsize=(7, 4))
                pd.Series(count_ann1.values, index=entity_names).plot(
                    kind="bar",
                    ax=ax,
                    color=colors)
                ax.set_title("Annotator 1")
                ax.set_xlabel("Entitas")
                ax.set_ylabel("Jumlah")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with c2:
                fig, ax = plt.subplots(figsize=(7, 4))
                pd.Series(count_ann2.values, index=entity_names).plot(
                    kind="bar",
                    ax=ax,
                    color=colors
                )
                
                ax.set_title("Annotator 2")
                ax.set_xlabel("Entitas")
                ax.set_ylabel("Jumlah")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
        elif detail_type == "wc_penolakan":
            st.markdown("### Word Cloud - Penolakan Kebijakan")
            text_data = clean_text_for_wc(get_text_by_label(df_hasil_anotasi, "Penolakan_Kebijakan"))
            fig = generate_wordcloud(text_data, colormap="Reds")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_dukungan":
            st.markdown("### Word Cloud - Dukungan Kebijakan")
            text_data = clean_text_for_wc(get_text_by_label(df_hasil_anotasi, "Dukungan_Kebijakan"))
            fig = generate_wordcloud(text_data, colormap="Greens")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_kritik":
            st.markdown("### Word Cloud - Kritik Pemerintahan")
            text_data = clean_text_for_wc(get_text_by_label(df_hasil_anotasi, "Kritik_Pemerintahan"))
            fig = generate_wordcloud(text_data, colormap="Oranges")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_netral":
            st.markdown("### Word Cloud - Netral")
            text_data = clean_text_for_wc(get_text_by_label(df_hasil_anotasi, "Netral"))
            fig = generate_wordcloud(text_data, colormap="Blues")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_organisasi":
            st.markdown("### Word Cloud - Organisasi")
            text_data = clean_text_for_wc(
                get_text_by_entity(df_hasil_anotasi, "Organisasi_ann1", "Organisasi_ann2")
            )
            fig = generate_wordcloud(text_data, colormap="Purples")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_platform":
            st.markdown("### Word Cloud - Platform")
            text_data = clean_text_for_wc(
                get_text_by_entity(df_hasil_anotasi, "Platform_ann1", "Platform_ann2")
            )
            fig = generate_wordcloud(text_data, colormap="PuBuGn")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_age":
            st.markdown("### Word Cloud - Age Group")
            text_data = clean_text_for_wc(
                get_text_by_entity(df_hasil_anotasi, "Age_Group_ann1", "Age_Group_ann2")
            )
            fig = generate_wordcloud(text_data, colormap="YlGnBu")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_policy":
            st.markdown("### Word Cloud - Policy")
            text_data = clean_text_for_wc(
                get_text_by_entity(df_hasil_anotasi, "Policy_ann1", "Policy_ann2")
            )
            fig = generate_wordcloud(text_data, colormap="cool")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        elif detail_type == "wc_risk":
            st.markdown("### Word Cloud - Digital Risk")
            text_data = clean_text_for_wc(
                get_text_by_entity(df_hasil_anotasi, "Digital_Risk_ann1", "Digital_Risk_ann2")
            )
            fig = generate_wordcloud(text_data, colormap="magma")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Tutup", key=f"close_{detail_type}", use_container_width=True):
                close_detail()
                st.rerun()

    if st.session_state.detail_type is not None:
        show_detail_dialog(st.session_state.detail_type)

    # =========================================================
    # SECTION 1
    # =========================================================
    st.markdown(
        """
        <div class='subsection-title'>Informasi Dataset</div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Jumlah Data</div>
                <div class="plot-area">
                    <h2>{iaaa_data["n_coincident_examples"]}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Jumlah Annotator</div>
                <div class="plot-area">
                    <h2>{iaaa_data["n_annotators"]}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Pairwise Precision</div>
                <div class="plot-area">
                    <h2>{iaaa_data["pairwise_precision"]}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Pairwise Recall</div>
                <div class="plot-area">
                    <h2>{iaaa_data["pairwise_recall"]}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col5:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Pairwise F1-score</div>
                <div class="plot-area">
                    <h2>{iaaa_data["pairwise_f1"]}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # =========================================================
    # SECTION 2
    # =========================================================
    st.markdown(
        """
        <div class='subsection-title'>Percent Agreement</div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Penolakan Kebijakan</div>
                <div class="plot-area">
                    <h2>{(iaaa_text["PENOLAKAN_KEBIJAKAN"]["percent_agreement"] * 100):.2f}%</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Dukungan Kebijakan</div>
                <div class="plot-area">
                    <h2>{(iaaa_text["DUKUNGAN_KEBIJAKAN"]["percent_agreement"] * 100):.2f}%</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Kritik Pemerintahan</div>
                <div class="plot-area">
                    <h2>{(iaaa_text["KRITIK_PEMERINTAHAN"]["percent_agreement"] * 100):.2f}%</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class="plot-card">
                <div class="viz-title">Netral</div>
                <div class="plot-area">
                    <h2>{(iaaa_text["PENOLAKAN_KEBIJAKAN"]["percent_agreement"] * 100):.2f}%</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    
    # =========================================================
    # SECTION 3
    # =========================================================
    st.markdown(
        """
        <div class='subsection-title'>Visualisasi Data</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        y = df_hasil_anotasi.index
        x = df_hasil_anotasi["panjang_karakter"]
        ax1.fill_betweenx(y, x)
        ax1.set_xlabel("Panjang Karakter")
        ax1.set_ylabel("Baris ke-")
        ax1.grid(True, alpha=0.2)
        render_plot_card("Panjang Karakter dari Setiap Baris", fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        y = df_hasil_anotasi.index
        x = df_hasil_anotasi["jumlah_kata"]
        ax2.fill_betweenx(y, x)
        ax2.set_xlabel("Jumlah Kata")
        ax2.set_ylabel("Baris ke-")
        ax2.grid(True, alpha=0.3)
        render_plot_card("Jumlah Kata dari Setiap Baris", fig2)
        plt.close(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax3
        )
        ax3.set_xlabel("Annotator 2")
        ax3.set_xticklabels(labels, rotation=40)
        ax3.set_ylabel("Annotator 1")
        render_plot_card("Korelasi Matrix", fig3)
        plt.close(fig3)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================
    # SECTION 4
    # =========================================================
    st.markdown(
        """
        <div class='subsection-title'>Distribusi Label dan Named Entity Recognition (NER)</div>
        """,
        unsafe_allow_html=True
    )

    col4, col5 = st.columns(2)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        labels_x = df_compare_label.index
        x = np.arange(len(labels_x))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, df_compare_label["Annotator 1"], width, label="Annotator 1", color="#000249")
        bars2 = ax4.bar(x + width / 2, df_compare_label["Annotator 2"], width, label="Annotator 2", color="#ff69b4")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    str(int(height)),
                    ha="center",
                    va="bottom"
                )

        ax4.set_xlabel("Label")
        ax4.set_ylabel("Jumlah Data")
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels_x, rotation=15)
        ax4.legend()
        render_plot_card("Distribusi Label", fig4)
        plt.close(fig4)

        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.button("Detail", key="detail_label", on_click=open_detail, args=("label",), use_container_width=True)

    with col5:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        labels_ent = df_jumlah_entitas.index
        x2 = np.arange(len(labels_ent))
        width2 = 0.35

        bars1 = ax5.bar(x2 - width2 / 2, df_jumlah_entitas["Annotator 1"], width2, label="Annotator 1", color="#000249")
        bars2 = ax5.bar(x2 + width2 / 2, df_jumlah_entitas["Annotator 2"], width2, label="Annotator 2", color="#ff69b4")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    str(int(height)),
                    ha="center",
                    va="bottom"
                )

        ax5.set_xlabel("Entitas")
        ax5.set_ylabel("Jumlah Data")
        ax5.set_xticks(x2)
        ax5.set_xticklabels(labels_ent, rotation=15)
        ax5.legend()
        render_plot_card("Distribusi NER", fig5)
        plt.close(fig5)

        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.button("Detail", key="detail_ner", on_click=open_detail, args=("ner",), use_container_width=True)
    
    st.markdown("""<br>""",unsafe_allow_html=True)
    
    # =========================================================
    # SECTION 5
    # =========================================================
    st.markdown(
        """
        <div class='subsection-title'>Word Cloud</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<br>""", unsafe_allow_html=True)

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    with row1_col1:
        st.button("Penolakan Kebijakan", key="wc_penolakan", on_click=open_detail, args=("wc_penolakan",), use_container_width=True)
    with row1_col2:
        st.button("Dukungan Kebijakan", key="wc_dukungan", on_click=open_detail, args=("wc_dukungan",), use_container_width=True)
    with row1_col3:
        st.button("Kritik Pemerintahan", key="wc_kritik", on_click=open_detail, args=("wc_kritik",), use_container_width=True)
    with row1_col4:
        st.button("Netral", key="wc_netral", on_click=open_detail, args=("wc_netral",), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns(5)
    with row2_col1:
        st.button("Organisasi", key="wc_organisasi", on_click=open_detail, args=("wc_organisasi",), use_container_width=True)
    with row2_col2:
        st.button("Platform", key="wc_platform", on_click=open_detail, args=("wc_platform",), use_container_width=True)
    with row2_col3:
        st.button("Age Group", key="wc_age", on_click=open_detail, args=("wc_age",), use_container_width=True)
    with row2_col4:
        st.button("Policy", key="wc_policy", on_click=open_detail, args=("wc_policy",), use_container_width=True)
    with row2_col5:
        st.button("Digital Risk", key="wc_risk", on_click=open_detail, args=("wc_risk",), use_container_width=True)

# =========================================================
# CUSTOM CSS
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
            margin-bottom: 0.9rem;
            max-width: 980px;
            position: relative;
            z-index: 1;
        }

        .section-card {
            background: #F5FEFF;
            border-radius: 24px;
            padding: 20px;
            border: 1px solid rgba(147, 197, 253, 0.4);
            box-shadow: 0 14px 30px rgba(37, 99, 235, 0.08);
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
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
        }

        .plot-card {
            background: rgba(255,255,255,0.92);
            border-radius: 20px;
            padding: 16px;
            border: 1px solid rgba(147, 197, 253, 0.45);
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.10);
            margin-bottom: 12px;
        }

        .viz-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #1e3a8a;
            text-align: center;
            padding: 12px 14px;
            border-radius: 14px;
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border: 1px solid rgba(147, 197, 253, 0.35);
            margin-bottom: 14px;
        }

        .plot-area {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .plot-area h2 {
            font-size: 2rem;
            font-weight: 800;
            color: #1e3a8a;
            text-align: center;
        }

        .plot-img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 12px;
        }

        div.stButton > button {
            border-radius: 999px;
            padding: 0.60rem 1.25rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.25);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            transform: translateY(-1px);
        }

        div[data-testid="stDialog"] > div {
            border: 1px solid rgba(147, 197, 253, 0.35) !important;
            box-shadow: 0 30px 80px rgba(0,0,0,0.30) !important;
        }
        
        div[data-testid="stDialog"] button[aria-label="Close"],
        div[data-testid="stDialog"] button[aria-label="close"],
        div[data-testid="stDialog"] .st-emotion-cache-1umgz6k,
        div[data-testid="stDialog"] .st-emotion-cache-19rxjzo {
            display: none !important;
        }

        /* tombol di popup jadi merah */
        div[data-testid="stDialog"] div.stButton > button {
            border-radius: 999px !important;
            padding: 0.65rem 1.35rem !important;
            font-weight: 700 !important;
            border: none !important;
            background: linear-gradient(135deg, #dc2626, #ef4444) !important;
            color: white !important;
            box-shadow: 0 10px 24px rgba(239, 68, 68, 0.28) !important;
        }

        /* hover tombol popup */
        div[data-testid="stDialog"] div.stButton > button:hover {
            background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
            transform: translateY(-1px);
        }

        /* dataframe / chart area di popup tetap putih agar jelas */
        div[data-testid="stDialog"] .stDataFrame,
        div[data-testid="stDialog"] iframe,
        div[data-testid="stDialog"] canvas {
            border-radius: 14px !important;
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