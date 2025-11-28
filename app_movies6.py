import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")

# ----------------- BASIC PAGE SETUP -----------------
st.set_page_config(
    page_title="Movie Ratings Dashboard",
    layout="wide",
    page_icon="🎬"
)

# ----------------- TITLE -----------------
st.markdown(
    "<h1 style='color:#ff4b4b; text-align:center; font-weight:800;'>Movie Ratings Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='color:#cccccc; text-align:center;'>Exploring critic scores, audience reactions and budgets</h3>",
    unsafe_allow_html=True
)

# ----------------- DATA LOADING -----------------
DATA_PATH = "Movie-Rating.csv"

@st.cache_data
def load_movies(path: str) -> pd.DataFrame:
    movies = pd.read_csv(path)
    movies.columns = ['Film','Genre','CriticRating','AudienceRating','BudgetMillions','Year']
    movies.Film = movies.Film.astype('category')
    movies.Genre = movies.Genre.astype('category')
    movies.Year = movies.Year.astype('category')
    return movies

movies = load_movies(DATA_PATH)

# ----------------- KPI CARDS (FINAL FIX — USING <p> TAGS) -----------------
col1, col2, col3, col4 = st.columns(4)

kpi_style = (
    "background:#222831; padding:15px; border-radius:12px; "
    "text-align:center; border:1px solid #393e46;"
)

title_style = "color:#eeeeee; font-size:18px; font-weight:600; margin:0;"
value_style = "color:#00adb5; font-size:28px; font-weight:800; margin:0;"

col1.markdown(
    f"""
    <div style="{kpi_style}">
        <p style="{title_style}">Avg Critic Rating</p>
        <p style="{value_style}">{movies.CriticRating.mean():.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

col2.markdown(
    f"""
    <div style="{kpi_style}">
        <p style="{title_style}">Avg Audience Rating</p>
        <p style="{value_style}">{movies.AudienceRating.mean():.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

col3.markdown(
    f"""
    <div style="{kpi_style}">
        <p style="{title_style}">Avg Budget (M)</p>
        <p style="{value_style}">{movies.BudgetMillions.mean():.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

col4.markdown(
    f"""
    <div style="{kpi_style}">
        <p style="{title_style}">Total Movies</p>
        <p style="{value_style}">{len(movies)}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.title("🎯 Movie Filters")

genre_options = ["All Genres"] + list(movies.Genre.cat.categories)
genre_selected = st.sidebar.selectbox("Select Genre", genre_options)

year_min = int(movies.Year.astype(int).min())
year_max = int(movies.Year.astype(int).max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

filtered = movies.copy()
filtered = filtered[filtered.Year.astype(int).between(year_range[0], year_range[1])]

if genre_selected != "All Genres":
    filtered = filtered[filtered.Genre == genre_selected]

st.sidebar.write(f"Filtered movies: **{len(filtered)}**")

# ----------------- TABS -----------------
tab_overview, tab_rel, tab_dist, tab_genre, tab_advanced = st.tabs(
    ["📊 Overview", "📈 Relationships", "📦 Distributions", "🎬 Genre & Year", "🧩 Advanced Dashboard"]
)

# ---------------- TAB 1: OVERVIEW ----------------
with tab_overview:
    st.subheader("Filtered Movies Table")
    if st.button("Show First 15 Filtered Movies"):
        st.dataframe(filtered.head(15))
    else:
        st.dataframe(filtered.head(10))

    st.markdown("#### Summary Statistics")
    st.dataframe(filtered[['CriticRating','AudienceRating','BudgetMillions']].describe().T)

# ---------------- TAB 2: RELATIONSHIPS ----------------
with tab_rel:
    st.subheader("Critic vs Audience Rating")

    joint_kind = st.radio(
        "Joint plot type",
        ["hex","scatter","reg","kde","hist","resid"],
        index=0,
        horizontal=True
    )

    j = sns.jointplot(
        data=filtered,
        x="CriticRating",
        y="AudienceRating",
        kind=joint_kind,
        height=6
    )
    st.pyplot(j.fig)
    plt.close(j.fig)

    st.markdown("---")
    st.subheader("Scatter Plot by Genre")

    vis1 = sns.lmplot(
        data=filtered,
        x="CriticRating",
        y="AudienceRating",
        fit_reg=False,
        hue="Genre",
        height=5,
        aspect=1
    )
    st.pyplot(vis1.fig)
    plt.close(vis1.fig)

# ---------------- TAB 3: DISTRIBUTIONS ----------------
with tab_dist:
    st.subheader("Ratings & Budget Distributions")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Audience Rating Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(filtered.AudienceRating, bins=20, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Budget vs Audience KDE**")
        fig, ax = plt.subplots()
        sns.kdeplot(data=filtered, x="BudgetMillions", y="AudienceRating",
                    fill=True, cmap="Greens", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with d2:
        st.markdown("**Budget Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(filtered.BudgetMillions, bins=20, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Budget vs Critic KDE**")
        fig, ax = plt.subplots()
        sns.kdeplot(data=filtered, x="BudgetMillions", y="CriticRating",
                    fill=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Stacked Histogram of Budget by Genre")

    genre_list = list(movies.Genre.cat.categories)
    budget_by_genre = [movies[movies.Genre == g].BudgetMillions for g in genre_list]

    fig, ax = plt.subplots()
    ax.hist(budget_by_genre, bins=20, stacked=True, label=genre_list)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---------------- TAB 4: GENRE & YEAR ----------------
with tab_genre:
    st.subheader("Box & Violin Plots by Genre")

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Boxplot of Critic Rating by Genre**")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=filtered, x="Genre", y="CriticRating", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with g2:
        st.markdown("**Violin Plot of Critic Rating by Genre**")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.violinplot(data=filtered, x="Genre", y="CriticRating", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("FacetGrid: Genre vs Year")

    g = sns.FacetGrid(movies, row="Genre", col="Year", hue="Genre")
    g.map(plt.hist, "BudgetMillions")
    st.pyplot(g.fig)
    plt.close(g.fig)

# ---------------- TAB 5: ADVANCED DASHBOARD ----------------
with tab_advanced:
    st.subheader("Advanced 2×2 Visualization Dashboard")

    fig, axes = plt.subplots(2,2, figsize=(13,10))

    sns.kdeplot(data=movies, x="BudgetMillions", y="AudienceRating",
                fill=True, cmap="viridis", ax=axes[0,0])
    axes[0,0].set_title("Budget vs Audience (KDE)")

    sns.kdeplot(data=movies, x="BudgetMillions", y="CriticRating",
                fill=True, cmap="magma", ax=axes[0,1])                      
    axes[0,1].set_title("Budget vs Critic (KDE)")

    sns.violinplot(data=movies[movies.Genre=="Drama"],
                   x="Year", y="CriticRating", ax=axes[1,0])
    axes[1,0].set_title("Drama Rating by Year")

    sns.kdeplot(data=movies, x="CriticRating", y="AudienceRating",
                fill=True, cmap="Reds", ax=axes[1,1])
    sns.kdeplot(data=movies, x="CriticRating", y="AudienceRating",
                cmap="Greys", ax=axes[1,1])
    axes[1,1].set_title("Critic vs Audience (Contours)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

