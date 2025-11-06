import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display
from matplotlib import animation
from IPython.display import Image
import warnings




def remove_nan_plus_constant_and_correlated_features(df, corr_threshold=0.95, verbose=True):
    """
    Remove constant and highly correlated features from a DataFrame. 
    If NaN values represent less than 2% of dataset, the corresponding rows are removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (numerical features only).
    corr_threshold : float, default=0.95
        Absolute correlation threshold above which one of the correlated
        features will be removed.
    verbose : bool, default=True
        If True, prints information about removed columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with constant and highly correlated features removed.
    list
        List of removed features.
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import VarianceThreshold

    df = df.copy()

    nan_percentage = df.isna().any(axis=1).mean() * 100
    print("The dataset contains : " + str(nan_percentage) + "% of NaN.")
    if nan_percentage < 2:
        df.dropna()
        print("There are few NaN, therefore we make the choice to remove them.")
    
    removed_features = []

    # --- 1. Remove constant features ---
    selector = VarianceThreshold(threshold=0)
    selector.fit(df)
    constant_columns = [col for col, var in zip(df.columns, selector.variances_) if var == 0]

    if constant_columns:
        if verbose:
            print(f"🧩 Constant features removed: {constant_columns}")
        df.drop(columns=constant_columns, inplace=True)
        removed_features.extend(constant_columns)
    else:
        if verbose:
            print("✅ No constant features found.")

    # --- 2. Remove correlated features ---
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_to_remove = set()
    for col in upper.columns:
        correlated_cols = [idx for idx, val in upper[col].items() if val > corr_threshold and idx != col]
        correlated_to_remove.update(correlated_cols)

    if correlated_to_remove:
        if verbose:
            print(f"⚙️ {len(correlated_to_remove)} correlated features removed (|r| > {corr_threshold}):")
            print(list(correlated_to_remove))
        df.drop(columns=list(correlated_to_remove), inplace=True)
        removed_features.extend(correlated_to_remove)
    else:
        if verbose:
            print("✅ No highly correlated features found.")

    return df, removed_features



def reshape_repeated_columns(df, n_groups=15, n_features_per_group=3, feature_names=['f_x', 'f_y', 'f_ori'], prefix="f"):
    """
    Reshape a DataFrame with repeated feature blocks into a long format.

    Example:
        Columns: f0_x, f0_y, f0_ori, f1_x, f1_y, f1_ori, ..., f14_ori
        n_groups = 15 (fish)
        n_features_per_group = 3 (x, y, ori)
        feature_names = ['f_x', 'f_y', 'f_ori']

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with repeated columns per group.
    n_groups : int
        Number of repeated groups (e.g., number of fish).
    n_features_per_group : int
        Number of features per group (e.g., x, y, ori → 3).
    feature_names : list of str
        Names of the features to assign in the reshaped DataFrame.
    prefix : str, default='f'
        Column prefix (e.g., 'f' → f0_x, f1_y, ...)

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per (frame, group_id).
    """
    import pandas as pd

    # Sanity check
    expected_cols = n_groups * n_features_per_group
    repeated_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(repeated_cols) != expected_cols:
        raise ValueError(f"Expected {expected_cols} repeated columns, found {len(repeated_cols)}.")

    # Identify shared (non-repeated) columns
    shared_cols = [c for c in df.columns if not c.startswith(prefix)]

    fish_dfs = []
    for i in range(n_groups):
        cols_to_select = [f"{prefix}{i}_{f.replace('f_', '')}" for f in feature_names]
        temp = df[cols_to_select + shared_cols].copy()
        temp.columns = feature_names + shared_cols
        temp["fish_id"] = i
        temp["frame"] = df.index
        fish_dfs.append(temp)

    long_df = pd.concat(fish_dfs, ignore_index=True)
    long_df = long_df.sort_values(by=["fish_id", "frame"]).reset_index(drop=True)

    print(f"✅ Reshape complete: {len(df)} frames × {n_groups} fish → {len(long_df)} rows")
    print("Columns:", long_df.columns.tolist()[:10], "...")
    return long_df



def compute_speed(df, x_col="f_x", y_col="f_y", name="f_speed"):
    """
    Compute the instantaneous speed (magnitude of displacement)
    based on x and y positions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing x and y position columns.
    x_col, y_col : str
        Names of the columns representing position coordinates.
    name : str
        Name of the output column for speed.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with a new column for speed.
    """
    df = df.copy()
    df["dx"] = df[x_col].diff()
    df["dy"] = df[y_col].diff()
    df[name] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df = df.drop(columns=["dx", "dy"])
    
    first = [name]
    others = [c for c in df.columns if c not in first]
    df = df[first + others]
    
    return df



def compute_d_orientation(df, ori_col="f_ori", name="d_ori"):
    """
    Compute the change in orientation (Δθ) between consecutive frames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an orientation column.
    ori_col : str
        Name of the orientation column.
    name : str
        Name of the output column for Δθ.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with a new column for change in orientation.
    """
    df = df.copy()
    df[name] = np.diff(df[ori_col], prepend=df[ori_col].iloc[0])
    
    first = [name]
    others = [c for c in df.columns if c not in first]
    df = df[first + others]
    return df



def polar_pixel_encoding(row, r_bins=20, theta_bins=36, max_radius=400):
    """
    Convert 8 (x, y) stimulus positions into a polar pixel encoding grid
    relative to the fish's orientation (egocentric reference frame).
    Ignores stimuli with d{i}_stim == 0 (i.e. inactive stimuli).
    """
    required_ori = "f_ori"
    if required_ori not in row:
        warnings.warn(f"⚠️ Missing '{required_ori}' column in DataFrame row — skipping row.")
        return np.zeros((r_bins, theta_bins))

    # Check coordinate columns
    missing_coords = [f"d{i}_{axis}" for i in range(8) for axis in ["x", "y"] if f"d{i}_{axis}" not in row]
    if missing_coords:
        warnings.warn(f"⚠️ Missing coordinate columns: {missing_coords}")
        return np.zeros((r_bins, theta_bins))

    fish_ori = row["f_ori"]
    polar_grid = np.zeros((r_bins, theta_bins))

    # Define bin edges
    r_bin_edges = np.linspace(0, max_radius, r_bins + 1)
    theta_bin_edges = np.linspace(-np.pi, np.pi, theta_bins + 1)

    # Loop over the 8 stimuli
    for i in range(8):
        stim_col = f"d{i}_stim"

        # if stim_col=0 it means no stimulus
        if stim_col in row and row[stim_col] == 0:
            continue

        # Si les coordonnées manquent → warning + on saute
        try:
            x = row[f"d{i}_x"]
            y = row[f"d{i}_y"]
        except KeyError:
            warnings.warn(f"⚠️ Missing coordinate for stimulus {i}. Skipping.")
            continue

        # Conversion en coordonnées polaires
        dx = x - row["f_x"]
        dy = y - row["f_y"]

            # --- Absolute stimulus angle relative to fish ---
        stim_angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
        r = np.sqrt(dx**2 + dy**2)
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi) - fish_ori
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # wrap [-π, π]

        # Binning
        r_idx = np.clip(np.digitize(r, r_bin_edges) - 1, 0, r_bins - 1)
        theta_idx = np.clip(np.digitize(theta, theta_bin_edges) - 1, 0, theta_bins - 1)

        polar_grid[r_idx, theta_idx] = 1
        
    return polar_grid



def visualize_polar_encoding(polar_grid, r_bins=20, theta_bins=36, radius=1.0,
                             color_grid='white', bg_color='black', stim_color='yellow',
                             fish_color='deepskyblue', fish_size=0.05,
                             title="Polar encoding visualization", ax=None):
    """
    Draw polar encoding on an existing axis (used for GIF animation).
    """
    polar_grid = np.array(polar_grid)
    if polar_grid.shape != (r_bins, theta_bins):
        raise ValueError(f"Expected shape {(r_bins, theta_bins)}, got {polar_grid.shape}")

    # Create axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)

    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(bg_color)

    # --- Grid lines (white on black background) ---
    for i in range(1, r_bins + 1):
        r = radius * i / r_bins
        circle = plt.Circle((0, 0), r, color=color_grid, fill=False, lw=0.6)
        ax.add_artist(circle)

    # Rotation offset so that fish faces upward
    angle_offset = np.pi / 2
    angles = np.linspace(0, 2 * np.pi, theta_bins, endpoint=False) + angle_offset
    for angle in angles:
        ax.plot([0, radius * np.cos(angle)], [0, radius * np.sin(angle)],
                color=color_grid, lw=0.6)

    # --- Highlight active bins (stimuli) ---
    r_bin_edges = np.linspace(0, radius, r_bins + 1)
    theta_bin_edges = np.linspace(-np.pi, np.pi, theta_bins + 1) + angle_offset

    for n in range(r_bins):
        for m in range(theta_bins):
            if polar_grid[n, m] == 1:
                r_inner, r_outer = r_bin_edges[n], r_bin_edges[n + 1]
                theta1, theta2 = theta_bin_edges[m], theta_bin_edges[m + 1]
                theta = np.linspace(theta1, theta2, 30)
                x_inner = r_inner * np.cos(theta)
                y_inner = r_inner * np.sin(theta)
                x_outer = r_outer * np.cos(theta)
                y_outer = r_outer * np.sin(theta)
                x = np.concatenate([x_outer, x_inner[::-1]])
                y = np.concatenate([y_outer, y_inner[::-1]])
                ax.fill(x, y, color=stim_color, alpha=0.8)

    # --- Fish (center, facing upward) ---
    ax.plot(0, 0, marker='o', color=fish_color, markersize=8)
    ax.arrow(0, 0, 0, fish_size, head_width=0.05, head_length=0.1,
             fc=fish_color, ec=fish_color, lw=1.2, length_includes_head=True)

    # --- Axes ---
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.axis('off')
    ax.set_title(title, color='white', pad=20)

    return ax



def make_polar_encoding_gif_segments(
    df, 
    segment_starts,          # liste d'indices où commencent les segments
    segment_length=90,       # nombre de frames par segment (≈ 30 s si fps=3)
    fps=3,
    output_path="polar_segments.gif"
):
    """
    Create a GIF showing multiple experimental segments (each 30s) 
    from different parts of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column 'polar_encoding'
    segment_starts : list[int]
        List of starting frame indices (e.g. [0, 10000, 20000])
    segment_length : int
        Number of frames per segment
    fps : int
        Frame rate
    output_path : str
        Path to save the output GIF
    """

    # Collect frames from all chosen segments
    frames_to_use = []
    segment_ids = []
    for i, start in enumerate(segment_starts):
        end = start + segment_length
        for j in range(start, min(end, len(df))):
            frames_to_use.append(j)
            segment_ids.append(i + 1)  # experience number

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    plt.close(fig)

    def update(idx):
        ax.clear()
        frame_idx = frames_to_use[idx]
        segment_id = segment_ids[idx]
        t_rel = (idx % segment_length) / fps  # temps relatif à l’intérieur du bloc

        visualize_polar_encoding(
            df["polar_encoding"].iloc[frame_idx],
            r_bins=20,
            theta_bins=36,
            bg_color='black',
            color_grid='white',
            title=f"Condition {segment_id}  |  t = {t_rel:.1f} s",
            ax=ax
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames_to_use), blit=False)
    ani.save(output_path, writer="pillow", fps=fps)

    return Image(filename=output_path)



def temporal_train_val_test_split(
    df,
    target_cols,
    train_size=0.7,
    val_size=0.15,
    test_size=None,
    verbose=True,
):
    """
    Split a DataFrame into train, validation, and test sets (temporal order preserved).

    Returns directly:
        X_train, X_val, X_test,
        y_train_targets..., y_val_targets..., y_test_targets...

    So one can unpack in one line:
        X_train, X_val, X_test, target_speed_train, target_speed_val, target_speed_test, target_ori_train, target_ori_val, target_ori_test = ...
    """
    import pandas as pd

    if test_size is None:
        test_size = 1 - (train_size + val_size)

    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    n = len(df)
    split_1 = int(n * train_size)
    split_2 = int(n * (train_size + val_size))

    # Split
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_val, X_test = X.iloc[:split_1], X.iloc[split_1:split_2], X.iloc[split_2:]
    y_train, y_val, y_test = y.iloc[:split_1], y.iloc[split_1:split_2], y.iloc[split_2:]

    if verbose:
        print(f"✅ Temporal split complete:")
        print(f"  Train: {len(X_train)} rows ({train_size*100:.1f}%)")
        print(f"  Val:   {len(X_val)} rows ({val_size*100:.1f}%)")
        print(f"  Test:  {len(X_test)} rows ({test_size*100:.1f}%)")

    # Dynamically unpack target columns
    results = [X_train, X_val, X_test]
    for col in target_cols:
        results.extend([y_train[col], y_val[col], y_test[col]])

    return tuple(results)



def extract_polar_features(polar_grid):
    """
    Extracts 10 biologically and informationally meaningful features 
    from a polar pixel encoding grid.
    
    Parameters
    ----------
    polar_grid : np.ndarray
        2D array (r_bins × theta_bins) of 0/1 encoding stimulus positions.

    Returns
    -------
    dict
        Dictionary with 10 scalar features.
    """
    polar_grid = np.array(polar_grid)
    r_bins, theta_bins = polar_grid.shape
    total_bins = r_bins * theta_bins
    eps = 1e-9  # to avoid log(0)

    # 1. Number of active bins
    stim_count = polar_grid.sum()

    # 2. Average radial distance of active bins
    radii = np.repeat(np.arange(r_bins), theta_bins)
    avg_radius = (polar_grid.flatten() * radii).sum() / (stim_count + eps)

    # 3. Minimum radial distance (closest active stimulus)
    if stim_count > 0:
        min_radius = radii[polar_grid.flatten() == 1].min()
    else:
        min_radius = r_bins  # max possible

    # 4. Fraction of active bins in the front sector (e.g., -60° to +60°)
    front_bins = np.arange(theta_bins//3, 2*theta_bins//3)
    front_ratio = polar_grid[:, front_bins].sum() / (stim_count + eps)

    # 5. Left-right balance (difference between left/right hemisphere)
    left_half = polar_grid[:, :theta_bins//2].sum()
    right_half = polar_grid[:, theta_bins//2:].sum()
    left_right_balance = (right_half - left_half) / (stim_count + eps)

    # 6. Angular entropy (spread of active bins along angles)
    p_theta = polar_grid.sum(axis=0) / (stim_count + eps)
    angular_entropy = -np.sum(p_theta * np.log(p_theta + eps))

    # 7. Radial entropy (spread of active bins along radius)
    p_r = polar_grid.sum(axis=1) / (stim_count + eps)
    radial_entropy = -np.sum(p_r * np.log(p_r + eps))

    # 8. Center of mass in angular space (weighted mean angle)
    theta_vals = np.linspace(-np.pi, np.pi, theta_bins, endpoint=False)
    weights = polar_grid.sum(axis=0)
    if weights.sum() > 0:
        center_of_mass_theta = np.arctan2(
            (np.sin(theta_vals) * weights).sum(),
            (np.cos(theta_vals) * weights).sum()
        )
    else:
        center_of_mass_theta = 0.0

    # 9. Number of active rings (radii containing ≥1 active bin)
    n_rings_active = (polar_grid.sum(axis=1) > 0).sum()

    # 10. Number of active angular sectors (angles containing ≥1 active bin)
    n_angles_active = (polar_grid.sum(axis=0) > 0).sum()

    return {
        "stim_count": stim_count,
        "avg_radius": avg_radius,
        "min_radius": min_radius,
        "front_ratio": front_ratio,
        "left_right_balance": left_right_balance,
        "angular_entropy": angular_entropy,
        "radial_entropy": radial_entropy,
        "center_of_mass_theta": center_of_mass_theta,
        "n_rings_active": n_rings_active,
        "n_angles_active": n_angles_active,
    }
    
    
    
def compute_orientations_and_radii(df, x_cols, y_cols, ori_names=None, r_names=None):
    """
    Compute orientations (angles in radians) and radii for several (x, y) coordinate pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing x and y coordinates.
    x_cols : list of str
        List of column names for x coordinates (e.g., ['f0_x', 'd0_x', ...]).
    y_cols : list of str
        List of corresponding y coordinate column names (same length as x_cols).
    ori_names : list of str, optional
        List of names for orientation columns. If None, uses x_col.replace('_x', '_ori').
    r_names : list of str, optional
        List of names for radius columns. If None, uses x_col.replace('_x', '_r').

    Returns
    -------
    pd.DataFrame
        DataFrame with orientation and radius columns (same index as input df).
    """

    if len(x_cols) != len(y_cols):
        raise ValueError("x_cols and y_cols must have the same length.")

    if ori_names is None:
        ori_names = [c.replace("_x", "_ori") for c in x_cols]
    if r_names is None:
        r_names = [c.replace("_x", "_r") for c in x_cols]

    result = pd.DataFrame(index=df.index)

    for x_col, y_col, ori_col, r_col in zip(x_cols, y_cols, ori_names, r_names):
        # Orientation relative to horizontal
        result[ori_col] = np.arctan2(df[y_col], df[x_col])

        # Distance to origin (radius)
        result[r_col] = np.sqrt(df[x_col]**2 + df[y_col]**2)

    return result