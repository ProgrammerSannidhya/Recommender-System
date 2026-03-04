import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv("laptop.csv")

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# -----------------------------
# Price cleaning
# -----------------------------

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")[0]
    .astype(float)
)


# -----------------------------
# Feature creation
# -----------------------------

cols = ["Model", "Generation", "Core", "Ram", "SSD", "Display", "Graphics", "OS"]

for c in cols:
    if c not in df.columns:
        df[c] = ""

df[cols] = df[cols].astype(str)

df["combined_features"] = (
    df["Model"] + " " +
    df["Generation"] + " " +
    df["Core"] + " " +
    df["Ram"] + " " +
    df["SSD"] + " " +
    df["Display"] + " " +
    df["Graphics"] + " " +
    df["OS"]
)


# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_features"] = df["combined_features"].apply(clean_text)


# -----------------------------
# Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(df["clean_features"])


# -----------------------------
# Helpers
# -----------------------------

def extract_budget(query):
    nums = re.findall(r"\d+", str(query))
    if len(nums) == 0:
        return None
    return float(nums[0])


def extract_number_from_text(text):
    nums = re.findall(r"\d+", str(text))
    if len(nums) == 0:
        return None
    return int(nums[0])


# -----------------------------
# AMD detection (CPU or GPU)
# -----------------------------

def is_amd(row):
    text = (str(row["Core"]) + " " + str(row["Graphics"])).lower()
    return ("amd" in text) or ("ryzen" in text) or ("radeon" in text)


# -----------------------------
# Recommender with budget + AMD priority
# -----------------------------

def recommend_laptops_with_budget(query, top_n=3):

    budget = extract_budget(query)
    filtered_df = df.copy()

    if budget is not None:
        filtered_df = filtered_df[filtered_df["Price"] <= budget]

    if len(filtered_df) == 0:
        return filtered_df

    query_clean = clean_text(query)

    vectors = vectorizer.transform(filtered_df["clean_features"])
    query_vec = vectorizer.transform([query_clean])

    similarities = cosine_similarity(query_vec, vectors)[0]

    filtered_df = filtered_df.copy()
    filtered_df["sim_score"] = similarities

    # -------- AMD priority logic --------
    wants_amd = ("amd" in query_clean) or ("ryzen" in query_clean)

    if wants_amd:
        filtered_df["amd_boost"] = filtered_df.apply(is_amd, axis=1)
        filtered_df = filtered_df.sort_values(
            by=["amd_boost", "sim_score"],
            ascending=[False, False]
        )
    else:
        filtered_df = filtered_df.sort_values(
            by="sim_score",
            ascending=False
        )

    return filtered_df.head(top_n)


# -----------------------------
# Consumer friendly spec explanation
# -----------------------------

def simplify_specs(row):
    points = []

    # RAM
    ram = extract_number_from_text(row["Ram"])
    if ram is not None:
        if ram >= 16:
            points.append("Enough RAM for heavy multitasking and ML tools")
        elif ram >= 8:
            points.append("Good RAM for coding and daily work")
        else:
            points.append("Basic RAM for light usage")

    # Storage
    if "ssd" in str(row["SSD"]).lower():
        ssd = extract_number_from_text(row["SSD"])
        if ssd is not None and ssd >= 512:
            points.append("Fast and spacious SSD storage")
        else:
            points.append("Fast SSD storage")

    # Graphics
    g = str(row["Graphics"]).lower()
    if "nvidia" in g or "rtx" in g or "gtx" in g or "amd" in g or "radeon" in g:
        points.append("Dedicated graphics – useful for ML and graphics workloads")
    else:
        points.append("Integrated graphics – suitable for normal use")

    # Display
    points.append("Comfortable display for long coding and study sessions")

    return points


# -----------------------------
# Final function used by website
# -----------------------------

def final_recommendation(query, top_n=3):

    rows = recommend_laptops_with_budget(query, top_n)

    output = []

    for _, row in rows.iterrows():
        output.append({
            "model": row["Model"],
            "price": float(row["Price"]),
            "ram": row["Ram"],
            "core": row["Core"],
            "graphics": row["Graphics"],
            "why": simplify_specs(row)
        })

    return output
