import pandas as pd
import umap


def apply_umap_dimensionality_reduction(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply UMAP for dimensionality reduction on a DataFrame.

    Parameters:
    df (DataFrame): The input data frame to be reduced.
    n_components (int): The number of dimensions to reduce to.
    n_neighbors (int): The size of local neighborhood used for manifold approximation.
    min_dist (float): The minimum distance apart that points are allowed to be in the low-dimensional representation.
    random_state (int): Random seed for reproducibility.

    Returns:
    DataFrame: A DataFrame with reduced dimensions.
    """
    # Initialize UMAP reducer with provided parameters
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )

    # Fit UMAP and transform the DataFrame
    umap_embeddings = reducer.fit_transform(df)

    # Create a DataFrame for the embeddings with descriptive column names
    embedding_df = pd.DataFrame(
        umap_embeddings, columns=[f"UMAP_Dim{i+1}" for i in range(n_components)]
    )

    return embedding_df
