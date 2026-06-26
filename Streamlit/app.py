import streamlit as st
from streamlit_option_menu import option_menu

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
#  OPTION MENU
# =========================
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Data", "Visualisasi", "Result", "Prediction"],
    icons=["house-heart", "database", "bar-chart-line", "clipboard-data", "robot"],
    default_index=0,
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
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CONTENT BY MENU
# =========================
if selected == "Beranda":
    
    st.markdown(
    """
       <div class='section-card'>
            <h2>Selamat datang di aplikasi Analisis Opini Publik!</h2>
            <p>
                Aplikasi ini dikembangkan untuk menganalisis opini publik terhadap kebijakan
                pembatasan media sosial pada anak, dengan memanfaatkan teknologi <strong style="color: #030766cc;"> Machine Learning </strong> .
            </p>
            
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("""<br>""",unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='subsection-title'>Teknologi yang Digunakan</div>
        """,
        unsafe_allow_html=True)
    
    st.markdown("""<br>""",unsafe_allow_html=True)
    
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown("""
        <div class="tech-card">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" class="tech-logo">
            <div class="tech-title">Pandas</div>
            <p class="tech-desc">Digunakan untuk membaca, mengelola, dan memproses dataset secara terstruktur.</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown("""
        <div class="tech-card">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" class="tech-logo">
            <div class="tech-title">NumPy</div>
            <p class="tech-desc">Digunakan untuk komputasi numerik dan pengolahan array dalam analisis data.</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col3:
        st.markdown("""
        <div class="tech-card">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" class="tech-logo">
            <div class="tech-title">Scikit-learn</div>
            <p class="tech-desc">Digunakan untuk membangun, melatih, dan menguji model machine learning.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.markdown("""
        <div class="tech-card">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg" class="tech-logo">
            <div class="tech-title">GoogleApiClient</div>
            <p class="tech-desc">Digunakan untuk integrasi data atau layanan berbasis Google API.</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown("""
        <div class="tech-card">
            <img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" class="tech-logo">
            <div class="tech-title">Seaborn</div>
            <p class="tech-desc">Digunakan untuk visualisasi statistik yang informatif dan estetik.</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col3:
        st.markdown("""
        <div class="tech-card">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" class="tech-logo">
            <div class="tech-title">Matplotlib</div>
            <p class="tech-desc">Digunakan untuk membangun grafik dan visualisasi data.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Data":
    st.switch_page("pages/data.py")

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
            background-color: #EAF4FF; /* fallback */
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
        
        /* HERO */
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
        }
        
        .section-card h2 {
            margin-top: 0;
            color: #1f2937;
            font-size: 2rem;
            font-weight: 800;
        }

        .section-card p {
            color: #374151;
            font-size: 1.25rem;
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
        
        
        .tech-card {
            background: linear-gradient(180deg, #dff1ff 0%, #d7edff 100%);
            border: 1px solid rgba(125, 179, 229, 0.35);
            border-radius: 24px;
            padding: 1.8rem 1.3rem;
            text-align: center;
            min-height: 270px;

            box-shadow: 
                0 10px 25px rgba(0,0,0,0.05),
                0 3px 8px rgba(0,0,0,0.03);

            transition: all 0.25s ease;
        }

        .tech-card:hover {
            transform: translateY(-6px);
            box-shadow: 
                0 12px 28px rgba(3, 7, 102, 0.8),
                0 4px 10px rgba(3, 7, 102, 0.8);
        }

        .tech-icon {
            font-size: 2.8rem;
            margin-bottom: 0.8rem;
        }

        .tech-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }

        .tech-desc {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.6;
        }
        
        .tech-logo {
            width: 70px;
            height: 70px;
            object-fit: contain;
            margin-bottom: 12px;
            transition: transform 0.25s ease;
        }

        /* hover biar hidup */
        .tech-card:hover .tech-logo {
            transform: scale(1.1);
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
