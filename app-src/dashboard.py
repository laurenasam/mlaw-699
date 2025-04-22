import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter


# Set up Streamlit page config
st.set_page_config(
    page_title="🧅 H-2A Job Order Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
    }

    /* Make selected tab underline maize or blue */
    div[data-baseweb="tab-highlight"] {
        background-color: #00274C !important;  /* Use #FFCB05 for maize */
    }

    /* Improve selected tab font color */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        color: #00274C !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-baseweb="tab"] button[role="tab"][aria-selected="true"] {
        border-bottom: 3px solid #FFCB05 !important;
        color: #00274C !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-baseweb="tab"] button[role="tab"][aria-selected="true"] {
        border-bottom: 3px solid #FFCB05 !important;
        color: #00274C !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.image("block-m.png", width=100)
st.markdown("""
    <style>
        section[data-testid="stSidebar"] * {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        body {
            font-family: "Helvetica Neue", sans-serif;
        }
        .stButton>button {
            background-color: #FFCB05;
            color: black;
            border: none;
            font-weight: 600;
        }
        .stMultiSelect>div>div {
            background-color: #FFCB05 !important;
            color: black !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: bold;
            color: #00274C;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* === Multiselect / Dropdown Maize & Blue Theme === */

    /* Tag (selected filter) background & text */
    div[data-baseweb="tag"] {
        background-color: #FFCB05 !important;  /* Maize */
        color: #00274C !important;             /* Michigan Blue text */
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1.5px solid #00274C !important;
    }

    /* Tag close "X" icon */
    div[data-baseweb="tag"] span[aria-hidden="true"] {
        color: #00274C !important;
        font-weight: bold;
    }

    /* Dropdown input box styling */
    .stMultiSelect > div, .stSelectbox > div {
        background-color: white !important;
        border: 2px solid #00274C !important;
        border-radius: 6px !important;
        color: #00274C !important;
        box-shadow: none !important;
    }

    /* Dropdown label text */
    label, .st-bx, .st-ag {
        color: #00274C !important;
        font-weight: bold !important;
    }

    /* Dropdown hover styling */
    div[data-baseweb="option"]:hover {
        background-color: #FFCB05 !important;
        color: #00274C !important;
    }

    /* Force font color for tag content */
    div[data-baseweb="tag"] span {
        color: #00274C !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Maize color for dropdown section titles */
    .stMultiSelect label, .stSelectbox label {
        color: #FFCB05 !important;  /* Maize */
        font-weight: bold !important;
        font-size: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Light background behind folium map */
    .stContainer > iframe {
        background-color: white !important;
    }

    /* Remove any padding or margin around folium container */
    .element-container:has(> iframe) {
        padding: 0 !important;
        margin: 0 !important;
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# Define colors
MAIZE = "#FFCB05"
BLUE = "#00274C"

# ---------- STYLE ----------
st.markdown(f"""
    <style>
        .stApp {{
            background-color: white;
            color: {BLUE};
        }}
        .css-1d391kg .e1fqkh3o4 {{
            color: {BLUE};
        }}
        .css-1v3fvcr {{
            color: {BLUE};
        }}
        .st-bw {{
            background-color: {MAIZE} !important;
            color: {BLUE} !important;
        }}
        .st-eb {{
            color: {BLUE} !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("🧅 H-2A Job Order Explorer")

# ---------- LOAD DATA ----------
df = pd.read_csv("ONION-MASTER.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.rename(columns={"best_address": "address", "recruiter_cleaned": "recruiter"})
df = df.drop(columns=["person", "hit_or_no_hit", "has_facebook"], errors="ignore")
df["year"] = df["year"].astype(int)
df["certainty"] = pd.to_numeric(df["certainty"], errors="coerce")
df["recruiter"] = df["recruiter"].astype(str).str.replace(r"\s*-\s*\d+$", "", regex=True)

# ---------- FILTERS ----------
with st.sidebar:
    st.title("🔍 Filters")
    years = sorted(df["year"].dropna().unique())
    recruiters = sorted(df["recruiter"].dropna().unique())
    selected_years = st.multiselect("Select Year(s):", years, default=years)
    selected_recruiters = st.multiselect("Select Recruiter(s):", recruiters, default=recruiters)

filtered_df = df[df["year"].isin(selected_years) & df["recruiter"].isin(selected_recruiters)]

# ---------- COLOR MAP ----------
color_map = {
    name: MAIZE if i % 2 == 0 else BLUE
    for i, name in enumerate(filtered_df["recruiter"].unique())
}

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
    "🌍 Map", 
    "📄 Table", 
    "☁️ Retailers", 
    "🌐 Network", 
    "📊 Recruiters", 
    "🔁 Sankey", 
    "🛒 Retail Diversity"
])

# ---------- MAP ----------
with tab1:
    st.markdown("### 🌍 Map of Employers")

    m = folium.Map(location=[filtered_df["latitude"].mean(), filtered_df["longitude"].mean()], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered_df.iterrows():
        popup_html = f"""
        <strong>{row['farm']}</strong><br>
        <em>{row['address']}</em><br>
        <a href="{row['link_to_h2a']}" target="_blank">View H-2A Order</a>
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_html,
            tooltip=row["farm"],
            icon=folium.Icon(color="blue")
        ).add_to(marker_cluster)

    st_data = st_folium(m, width=None, height=550)

# ---------- TABLE ----------
with tab2:
    st.subheader("📄 H-2A Orders Table")

    df_table = filtered_df.copy()
    df_table["Link"] = df_table["link_to_h2a"].apply(lambda x: f'<a href="{x}" target="_blank">View Order</a>')
    df_table = df_table.drop(columns=["link_to_h2a"])
    df_table.columns = df_table.columns.str.replace("_", " ").str.title()

    # 👇 Interactive column selector with custom styling
    cols = st.multiselect(
        "Choose columns to display:",
        df_table.columns.tolist(),
        default=df_table.columns.tolist(),
    )

    st.markdown(
    df_table[cols].to_html(escape=False, index=False),
    unsafe_allow_html=True
)

# ---------- WORD CLOUD ----------
with tab3:
    st.subheader("☁️ Retailer WordCloud")

    try:
        from wordcloud import WordCloud
        import random

        # Custom maize and blue color function
        maize_blue_colors = ["#FFCB05", "#00274C"]

        def maize_blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return random.choice(maize_blue_colors)

        market_text = " ".join(df_raw["market"].dropna().astype(str).tolist())
        market_text = re.sub(r"\([^)]*\)", "", market_text.replace("\n", ",").replace(";", ","))
        retailers = [r.strip().replace(" ", "_") for r in market_text.split(",") if r.strip()]
        cleaned_text = " ".join(retailers)

        if not cleaned_text:
            raise ValueError("Empty market text")

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            collocations=False,
            color_func=maize_blue_color_func
        ).generate(cleaned_text)

        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    except Exception:
        #st.warning("No 'market' column found. Using fallback retailer list.")
        fallback = ["Walmart", "Costco", "Kroger", "Meijer", "Publix", "Whole Foods", "HEB", "Sam's Club", "Trader Joe's", "Target"]
        text = " ".join([r.replace(" ", "_") for r in fallback * 10])

        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            collocations=False,
            color_func=maize_blue_color_func
        ).generate(text)

        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

with tab4:
    st.subheader("🌐 Farm–Retailer Network")

    try:
        import plotly.graph_objects as go
        from collections import Counter

        edge_df = filtered_df[["farm", "market"]].dropna()
        edges = []
        for _, row in edge_df.iterrows():
            farm = row["farm"]
            markets = re.split(r",|;|\n", str(row["market"]))
            for retailer in markets:
                cleaned = retailer.strip()
                if cleaned and 3 < len(cleaned) < 50 and not cleaned.lower().startswith("and "):
                    edges.append((farm, cleaned))

        edge_weights = Counter(edges)
        G = nx.Graph()
        for edge, weight in edge_weights.items():
            G.add_edge(edge[0], edge[1], weight=weight)

        pos = nx.spring_layout(G, seed=42, k=1.2)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="gray"),
            mode='lines',
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )

        # Top retailers
        retailer_counts = Counter([e[1] for e in edges])
        top_retailers = set([r for r, _ in retailer_counts.most_common(10)])

        # Build nodes
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            label = node if len(node) < 35 else node[:30] + "..."
            node_text.append(label)

            if node in df["farm"].values:
                node_color.append("#FFCB05")  # maize
                node_size.append(10)
            elif node in top_retailers:
                node_color.append("#00274C")  # blue
                node_size.append(18)
            else:
                node_color.append("#A2C4E0")  # gray-blue
                node_size.append(8)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode="markers+text",
            textposition="bottom center",
            hoverinfo="text",
            showlegend=False,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1.5, color="black")
            ),
            textfont=dict(size=10)
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black"),
            height=750,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                title=None,
                x=0.01,
                y=0.99,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12, color="black")
            )
        )

        # Custom legend icons
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color="#FFCB05", line=dict(width=1.5, color="black")),
            name="Farm"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=18, color="#00274C", line=dict(width=1.5, color="black")),
            name="Top Retailer"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=8, color="#A2C4E0", line=dict(width=1.5, color="black")),
            name="Other Retailer"
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering farm–retailer network: {e}")


# ---------- RECRUITERS ----------
with tab5:
    st.subheader("📊 Number of Orders per Recruiter")

    rc = filtered_df["recruiter"].value_counts().reset_index()
    rc.columns = ["Recruiter", "Orders"]

    fig = px.bar(
        rc,
        x="Recruiter",
        y="Orders",
        text="Orders",
        color_discrete_sequence=["#00274C"],  # Michigan Blue
    )

    fig.update_traces(
        textfont=dict(color="black", size=14),
        marker_line_color="#FFCB05",  # Maize outline
        marker_line_width=1.5,
    )

    fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        tickangle=-45,
        title=None,
        tickfont=dict(color="black"),
        showgrid=False,
        linecolor="black",
        ticks="outside"
    ),
    yaxis=dict(
        title=dict(
            text="Number of Retailers",
            font=dict(color="black", size=14)
        ),
        tickfont=dict(color="black"),
        showgrid=True,
        gridcolor="#DDDDDD",
        zeroline=False,
        linecolor="black",
        ticks="outside"
    ),
    margin=dict(l=40, r=20, t=40, b=120),
    font=dict(color="black")
)


    st.plotly_chart(fig, use_container_width=True)


# ---------- SANKEY ----------
with tab6:
    st.subheader("🔁 Recruiter ➝ Farm Sankey Diagram")

    try:
        sankey_df = filtered_df[["recruiter", "farm"]].dropna()

        # Create unique list of nodes
        all_nodes = list(pd.unique(sankey_df[["recruiter", "farm"]].values.ravel()))
        node_indices = {name: i for i, name in enumerate(all_nodes)}

        # Count recruiter → farm relationships
        grouped_links = sankey_df.groupby(["recruiter", "farm"]).size().reset_index(name="count")

        sankey_data = go.Sankey(
            node=dict(
                label=all_nodes,
                pad=15,
                thickness=20,
                color="#FFCB05",  # maize
                line=dict(color="#00274C", width=1.2)  # blue outline
            ),
            link=dict(
                source=grouped_links["recruiter"].map(node_indices),
                target=grouped_links["farm"].map(node_indices),
                value=grouped_links["count"],
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>',
                color="#00274C"  # blue links
            )
        )

        fig = go.Figure(sankey_data)
        fig.update_layout(
            font=dict(size=12, color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to render Recruiter→Farm Sankey: {e}")

    st.markdown("---")

    # ---------- Double Sankey ----------
    st.subheader("🔀 Recruiter ➝ Farm ➝ Top Retailer Sankey Diagram")

    try:
        filtered_df["recruiter_clean"] = filtered_df["recruiter"].str.replace(r"\s*-\s*\d+", "", regex=True)
        sankey_data = filtered_df[["recruiter_clean", "farm", "market"]].dropna()

        sankey_edges = []
        all_retailers = []

        for _, row in sankey_data.iterrows():
            recruiter = row["recruiter_clean"]
            farm = row["farm"]
            markets = re.split(r",|;|\n", str(row["market"]))
            for retailer in markets:
                cleaned = retailer.strip()
                if cleaned and 3 < len(cleaned) < 50:
                    sankey_edges.append((recruiter, farm))
                    sankey_edges.append((farm, cleaned))
                    all_retailers.append(cleaned)

        top_retailers = set([r for r, _ in Counter(all_retailers).most_common(10)])

        sankey_edges = [edge for edge in sankey_edges if edge[1] in top_retailers or edge[0] in top_retailers]

        all_nodes = list(set([x for edge in sankey_edges for x in edge]))
        node_index = {name: i for i, name in enumerate(all_nodes)}

        sankey_links = dict(source=[], target=[], value=[])
        for src, tgt in sankey_edges:
            sankey_links["source"].append(node_index[src])
            sankey_links["target"].append(node_index[tgt])
            sankey_links["value"].append(1)

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=all_nodes,
                color="#FFCB05",
                line=dict(color="#00274C", width=1.2)
            ),
            link=dict(
                source=sankey_links["source"],
                target=sankey_links["target"],
                value=sankey_links["value"],
                color="#00274C"
            )
        )])

        sankey_fig.update_layout(
            paper_bgcolor="white",
            font=dict(color="black", size=12),
            margin=dict(t=10, b=10, l=10, r=10),
            height=600
        )
        st.plotly_chart(sankey_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error rendering Recruiter ➝ Farm ➝ Retailer Sankey: {e}")


with tab7:
    st.subheader("🛒 Retailer Diversity per Farm")

    if "retailer_count" in filtered_df.columns and "farm" in filtered_df.columns:
        # Aggregate max retailer count per farm
        diversity_df = (
            filtered_df.groupby("farm")["retailer_count"]
            .max()
            .reset_index()
            .sort_values("retailer_count", ascending=False)
        )
        diversity_df.columns = ["Farm", "Unique Retailers"]

        fig = px.bar(
            diversity_df,
            x="Farm",
            y="Unique Retailers",
            color="Unique Retailers",
            color_continuous_scale=["#FFCB05", "#00274C"],  # Maize to Blue
            title="Retailer Diversity per Farm",
            labels={"Unique Retailers": "Retailers"},
            height=600
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black", size=14),
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(color="black", size=12),
                title=None
            ),
            yaxis=dict(
                tickfont=dict(color="black", size=12),
                title=None
            ),
            margin=dict(l=40, r=20, t=40, b=120),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Retailer count data not available. Please ensure 'retailer_count' column exists.")


