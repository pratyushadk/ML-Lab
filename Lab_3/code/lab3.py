import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski

# --- A2: Encoding Functions ---

def label_encode(series):
    unique_categories = list(dict.fromkeys(series.dropna()))
    mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
    encoded_series = series.map(lambda x: mapping.get(x, -1))
    return encoded_series, mapping


def one_hot_encode(series):
    col_name = series.name if series.name else 'feature'
    unique_categories = sorted(series.dropna().unique())
    ohe_dict = {
        f"{col_name}_{cat}": (series == cat).astype(int)
        for cat in unique_categories
    }
    return pd.DataFrame(ohe_dict)


# --- A4: Generalized Minkowski Distance ---

def minkowski_distance(vector_a, vector_b, p):
    vec_a = np.array(vector_a, dtype=float)
    vec_b = np.array(vector_b, dtype=float)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector shapes do not match: {vec_a.shape} vs {vec_b.shape}")
    if p < 1:
        raise ValueError("Parameter p must be >= 1")

    return float(np.sum(np.abs(vec_a - vec_b) ** p) ** (1.0 / p))


# --- A7: Dot Product & Vector Norm ---

def dot_product(vector_a, vector_b):
    vec_a = np.array(vector_a, dtype=float)
    vec_b = np.array(vector_b, dtype=float)
    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must have identical dimensions.")
    return float(np.sum(vec_a * vec_b))


def euclidean_norm(vector):
    vec = np.array(vector, dtype=float)
    return float(np.sqrt(np.sum(vec ** 2)))


# --- A8: Statistical Functions ---

def compute_mean(data):
    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        raise ValueError("Empty array")
    return float(np.sum(arr) / len(arr))


def compute_variance(data, ddof=0):
    arr = np.array(data, dtype=float)
    mean_val = compute_mean(arr)
    return float(np.sum((arr - mean_val) ** 2) / (len(arr) - ddof))


def compute_std(data, ddof=0):
    return float(np.sqrt(compute_variance(data, ddof=ddof)))


def dataset_statistics(matrix):
    mat = np.array(matrix, dtype=float)
    num_features = mat.shape[1]
    means = [compute_mean(mat[:, j]) for j in range(num_features)]
    variances = [compute_variance(mat[:, j]) for j in range(num_features)]
    stds = [compute_std(mat[:, j]) for j in range(num_features)]
    return means, variances, stds


# --- A11: Custom K-Means Clustering ---

def kmeans(data_matrix, k, max_iterations=100, tolerance=1e-4, random_seed=42):
    data = np.array(data_matrix, dtype=float)
    num_samples, num_features = data.shape
    
    rng = np.random.default_rng(random_seed)
    centroids = data[rng.choice(num_samples, size=k, replace=False)].copy()
    labels = np.zeros(num_samples, dtype=int)
    inertia_history = []

    for iteration in range(max_iterations):
        for i in range(num_samples):
            dists = [minkowski_distance(data[i], c, p=2) for c in centroids]
            labels[i] = int(np.argmin(dists))

        new_centroids = np.zeros((k, num_features))
        for c_idx in range(k):
            members = data[labels == c_idx]
            if len(members) == 0:
                new_centroids[c_idx] = data[rng.choice(num_samples)]
            else:
                new_centroids[c_idx] = [compute_mean(members[:, j]) for j in range(num_features)]

        inertia = sum(minkowski_distance(data[i], new_centroids[labels[i]], p=2) ** 2 for i in range(num_samples))
        inertia_history.append(inertia)

        shift = max(minkowski_distance(centroids[c], new_centroids[c], p=2) for c in range(k))
        centroids = new_centroids

        if shift < tolerance:
            break

    return labels, centroids, inertia_history[-1], inertia_history


# --- Main execution script ---

def main():
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(script_dir, "..", "Lab Session Data.xlsx")
    else:
        excel_path = "../Lab Session Data.xlsx"

    df_raw = pd.read_excel(excel_path, sheet_name='marketing_campaign')

    # Task A1
    print("\n--- Task A1: Datatype Identification ---")
    feature_types = {
        'ID': 'Nominal', 'Year_Birth': 'Interval', 'Education': 'Ordinal',
        'Marital_Status': 'Nominal', 'Income': 'Ratio', 'Kidhome': 'Ratio',
        'Teenhome': 'Ratio', 'Dt_Customer': 'Interval', 'Recency': 'Ratio',
        'MntWines': 'Ratio', 'MntFruits': 'Ratio', 'MntMeatProducts': 'Ratio',
        'MntFishProducts': 'Ratio', 'MntSweetProducts': 'Ratio', 'MntGoldProds': 'Ratio',
        'NumDealsPurchases': 'Ratio', 'NumWebPurchases': 'Ratio', 'NumCatalogPurchases': 'Ratio',
        'NumStorePurchases': 'Ratio', 'NumWebVisitsMonth': 'Ratio', 'AcceptedCmp3': 'Nominal',
        'AcceptedCmp4': 'Nominal', 'AcceptedCmp5': 'Nominal', 'AcceptedCmp1': 'Nominal',
        'AcceptedCmp2': 'Nominal', 'Complain': 'Nominal', 'Z_CostContact': 'Ratio',
        'Z_Revenue': 'Ratio', 'Response': 'Nominal'
    }
    a1_df = pd.DataFrame(list(feature_types.items()), columns=['Feature', 'Measurement_Scale'])
    print(a1_df.to_string(index=False))

    # Task A2 & A3
    df_encoded = df_raw.copy().drop(columns=['Dt_Customer']).dropna(subset=['Income']).reset_index(drop=True)

    edu_order = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
    edu_map = {cat: idx for idx, cat in enumerate(edu_order)}
    df_encoded['Education_LE'] = df_encoded['Education'].map(edu_map)
    df_encoded = df_encoded.drop(columns=['Education'])

    ohe_marital = one_hot_encode(df_encoded['Marital_Status'])
    df_encoded = pd.concat([df_encoded.drop(columns=['Marital_Status']), ohe_marital], axis=1)

    print("\n--- Task A3: Dimensionality After Encoding ---")
    print(f"Original shape: {df_raw.shape}, Encoded shape: {df_encoded.shape}")

    # Task A4 & A5
    num_cols = [c for c in df_encoded.select_dtypes(include=[np.number]).columns if c != 'ID']
    vec_1 = df_encoded.loc[0, num_cols].values.astype(float)
    vec_2 = df_encoded.loc[1, num_cols].values.astype(float)

    p_vals = list(range(1, 11))
    custom_dists = [minkowski_distance(vec_1, vec_2, p) for p in p_vals]

    print("\n--- Task A5: Minkowski Distances (p = 1 to 10) ---")
    for p, d in zip(p_vals, custom_dists):
        print(f"p={p:2d}: {d:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, custom_dists, marker='o')
    plt.title('Minkowski Distance vs p')
    plt.xlabel('p')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('minkowski_plot.png')
    plt.close()

    # Task A6
    scipy_dists = [scipy_minkowski(vec_1, vec_2, p) for p in p_vals]
    print("\n--- Task A6: Scipy Distance Comparison ---")
    print(f"Match with scipy: {np.isclose(custom_dists, scipy_dists).all()}")

    # Task A7
    c_dot = dot_product(vec_1, vec_2)
    np_dot = np.dot(vec_1, vec_2)
    c_norm1 = euclidean_norm(vec_1)
    np_norm1 = np.linalg.norm(vec_1)

    print("\n--- Task A7: Dot Product & Norm ---")
    print(f"Dot product match: {np.isclose(c_dot, np_dot)} ({c_dot:.4f})")
    print(f"Norm match       : {np.isclose(c_norm1, np_norm1)} ({c_norm1:.4f})")

    # Task A8 & A9
    matrix = df_encoded[num_cols].values
    c_means, c_vars, c_stds = dataset_statistics(matrix)
    np_means = np.mean(matrix, axis=0)
    np_stds = np.std(matrix, axis=0)

    print("\n--- Task A9: Mean & Std Comparison ---")
    print(f"Means match NumPy: {np.allclose(c_means, np_means)}")
    print(f"Stds match NumPy : {np.allclose(c_stds, np_stds)}")

    # Task A10
    inc_data = df_encoded['Income'].values
    inc_mean = compute_mean(inc_data)
    inc_std = compute_std(inc_data)

    print("\n--- Task A10: Income Statistics ---")
    print(f"Mean: {inc_mean:.2f}, Std Dev: {inc_std:.2f}")

    plt.figure(figsize=(7, 4))
    plt.hist(inc_data, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(inc_mean, color='red', linestyle='--', label='Mean')
    plt.title('Income Distribution Histogram')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('income_histogram.png')
    plt.close()

    # Task A11
    km_feats = ['Income', 'MntWines']
    raw_km = df_encoded[km_feats].values
    km_data_norm = (raw_km - raw_km.min(axis=0)) / (raw_km.max(axis=0) - raw_km.min(axis=0) + 1e-9)

    labels, centroids, inertia, history = kmeans(km_data_norm, k=3, max_iterations=50)

    print("\n--- Task A11: K-Means Clustering ---")
    print(f"Inertia (WCSS): {inertia:.4f}")

    plt.figure(figsize=(7, 4))
    plt.scatter(km_data_norm[:, 0], km_data_norm[:, 1], c=labels, cmap='viridis', s=15, alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=100, label='Centroids')
    plt.title('K-Means (K=3)')
    plt.xlabel('Income (Normalized)')
    plt.ylabel('MntWines (Normalized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png')
    plt.close()


if __name__ == '__main__':
    main()
