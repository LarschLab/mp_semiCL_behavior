import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from pathlib import Path
import seaborn as sns
import matplotlib.gridspec as gridspec
plt.style.use('default')
import random

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(repo_root)
from src import analyse as an




def generate_transformed_datasets(stimuli_df, df, roi_df, output="both", experiment ="GsizeMotionSingle", verbose=True):
    """
    Build transformed datasets (absolute and/or polar) from the three raw experiment files.

    Parameters
    ----------
    stimuli_df : pd.DataFrame
        Trajectory file (e.g. `*_trajectory_*.csv`):
        dot positions in fish-centered millimeter coordinates at stimulus onset.
    df : pd.DataFrame
        PositionTxt file: raw fish positions and orientations (pixels, clockwise).
    roi_df : pd.DataFrame
        ROI file: dish geometry and screen offsets (pixels).
    output : {"absolute", "polar", "both"}, default "both"
        Which representation to return:
        - "absolute": fish + dots in absolute screen coordinates
        - "polar":   fish-centered polar coordinates of the dots
        - "both":    both, in a dict.
    experiment : {"GsizeMotionSingle", "GsizeMotionChoice"}, default "GsizeMotionSingle"
        Experiment configuration used to compute dot centers (left/right vs single cloud).
        Be carefull this pipeline is designed for GsizeMotionSingle experiment. for other experiments additional tests and modifications might be needed.

    Returns
    -------
    dict or dict[str, dict[str, pd.DataFrame]]
        If output == "absolute":
            {"f0": df_f0_abs, ..., "f14": df_f14_abs}
        If output == "polar":
            {"f0": df_f0_polar, ..., "f14": df_f14_polar}
        If output == "both":
            {
                "absolute": {"f0": df_f0_abs, ...},
                "polar":    {"f0": df_f0_polar, ...},
            }
        Each DataFrame combines the fish trajectory with the corresponding
        transformed stimulus positions for all frames of that fish.
    """
    
    stimuli_df, df, roi_df = stimuli_df.copy(), df.copy(), roi_df.copy()
    
    
    #----------------- Give proper column names to the dataframes -----------------#
    stimuli_df_col_names = ["name", "x", "y", "size", "color", "backg"]
    n_stimuli = len(stimuli_df.columns)//len(stimuli_df_col_names)
    stimuli_df.columns = stimuli_df_col_names * n_stimuli
    
    n_fish = len(df.columns)//3 - n_stimuli
    fish_columns = [f"f{i}_{var}" for i in range(n_fish) for var in ["x", "y", "ori"]]
    stimuli_columns = [f"d{i}_{var}" for i in range(n_stimuli) for var in ["x", "y", "stim"]]
    df.columns = fish_columns + stimuli_columns
    
    roi_columns = ["xoff", "yoff", "diameter", "x", "y", "radius"]
    roi_df.columns = roi_columns
    
    
    
    #------------------ Sanity check that everything matches -------------------#
                   
    #Check that stimuli df and df_fish have the same number of rows
    #Check that stimuli df and df_fish[stimuli_columns] have the same values in name/stim

    # Select only columns containing x, y, and name
    stimuli_df = stimuli_df.drop(columns=['size', 'color', 'backg'])

    # Rename columns to match the desired format
    new_column_names = [col for i in range(n_stimuli) for col in (f'd{i}_stim', f'd{i}_x', f'd{i}_y')]
    stimuli_df.columns = new_column_names

    # Reorder columns to match the desired structure
    ordered_columns = [col for i in range(n_stimuli) for col in (f'd{i}_x', f'd{i}_y', f'd{i}_stim')]
    stimuli_df = stimuli_df[ordered_columns]    
    
    stim_columns = [f"d{i}_stim" for i in range(n_stimuli)]
    all_true = (df[stim_columns] == stimuli_df[stim_columns]).all().all()

    if verbose :
        if all_true:
            print("All stimuli values between PositionTxt and trajectory files match")
        else:
            print("Some stimuli values between PositionTxt and trajectory files do not match. \n"
                "ATTENTION!!!")
        
        
        
        
    #----------------------- Extract parameters of the experiment -----------------------#
    
    # 1. Calculate the duration of the initial pause
    framerate = 30
    PX_TO_MM = 3.67
    avgROIRadius = round(roi_df['radius'].mean())

    # Identify key indices
    start_interpause_idx = (df["d0_stim"] == "none").idxmax()
    CLstimuli_mask = df["d0_stim"].str.startswith("CLsemi", na=False)
    CLstimuli_start_idx = CLstimuli_mask.idxmax()
    CLstimuli_end_idx = CLstimuli_mask[CLstimuli_start_idx:].idxmin()

    # Compute durations
    CLstimuli_duration = (CLstimuli_end_idx - CLstimuli_start_idx) // framerate
    post_stim_pause_start = (df.loc[CLstimuli_end_idx:]["d0_stim"] == "none").idxmin()
    inter_pause_duration = (post_stim_pause_start - CLstimuli_end_idx) / framerate

    initial_pause = (df["d0_stim"] == "start_none").idxmin() / framerate
    bout_duration = len(df.loc[df["d0_stim"] == "bout"])//framerate

    n_trials_tot = len(df.index[(df['d0_stim'] != "none") & (df['d0_stim'].shift(1) == "none")])
    experimental_conditions = [val for val in df["d0_stim"].unique() if val.startswith("CLsemi")]
    n_trials_per_condition = n_trials_tot // len(experimental_conditions)

    if verbose :
        # Make a nice print that describe and summarize the experiment
        print(f"Experiment type: {experiment}")
        print(f"Initial pause: {initial_pause/60} min")
        print(f"Interstimulus pause: {inter_pause_duration} s")
        print(f"CLstimuli duration: {CLstimuli_duration} s")
        print(f"Bout duration: {bout_duration/60} min")
        print(f"Number of trials: {n_trials_tot}")
        print(f"Number of trials per condition: {n_trials_per_condition}")
        print(f"{len(experimental_conditions)} experimental conditions:{experimental_conditions}")
        
    
    
    
    #------------------------- extract bout and fish dataframes -------------------------#
    
    #remove initial pause
    df["condition"] = np.where(df["d0_stim"] == "none", "inter_stim_pause", df["d0_stim"])

    #save bout data separately and remove from main data
    bout_df = df.loc[df["d0_stim"] == "bout"].copy()
    bout_df = bout_df.drop(columns=stimuli_columns)
    stimuli_bout_df = stimuli_df.loc[stimuli_df["d0_stim"] == "bout"].copy()
    stimuli_bout_df = stimuli_bout_df[['d0_x', 'd0_y', 'd0_stim']]
    stimuli_bout_df = stimuli_bout_df.round(3)
    stimuli_bout_df["condition"] = bout_df["condition"]

    df_selected = df.loc[(CLstimuli_start_idx - int(inter_pause_duration*framerate)):].copy()
    fish_df = df_selected.loc[df["d0_stim"] != "bout"].copy()
    fish_df = fish_df.drop(columns=stimuli_columns)
    stimuli_fish_df = stimuli_df.loc[(CLstimuli_start_idx - int(inter_pause_duration*framerate)):].copy()
    stimuli_fish_df = stimuli_fish_df.loc[stimuli_fish_df["d0_stim"] != "bout"]
    stimuli_fish_df["condition"] = fish_df["condition"]
    
    
    #----------------- Extract information about trial and stimulus block start -----------------#
    
    # fish_df['trial_start'] = (fish_df['condition'] != "inter_stim_pause") & (fish_df['condition'].shift(1) == "inter_stim_pause")

    fish_df['trial_start'] = (fish_df['condition'] == "inter_stim_pause") & (fish_df['condition'].shift(1) != "inter_stim_pause")
    fish_df['trial_id'] = fish_df['trial_start'].cumsum()

    fish_df = fish_df.copy()
    fish_df['stimulus_block_id'] = (fish_df['condition'] != fish_df['condition'].shift(1)).cumsum()
    fish_df['frame_count_stimulus'] = fish_df.groupby('stimulus_block_id').cumcount()
    fish_df['frame'] = np.arange(0, len(fish_df))

    stimuli_fish_df['trial_id'] = fish_df['trial_id']
    stimuli_fish_df['stimulus_block_id'] = fish_df['stimulus_block_id']
    stimuli_fish_df['frame_count_stimulus'] = fish_df['frame_count_stimulus']
    stimuli_fish_df['trial_start'] = fish_df['trial_start']
    stimuli_fish_df['frame'] = np.arange(0, len(stimuli_fish_df))
    
    
    
    
    #----------------- Apply pause-copy logic to each trial independently -----------------#
    #   Replace dot coordinates during inter_stim_pause by copying the first frames of the corresponding stimulus block
    #   Ensures that pauses contain consistent dot positions before stimulus onset
    
    dot_xy_cols = [col for col in stimuli_fish_df.columns if col.startswith('d') and ('_x' in col or '_y' in col)]
    dot_stim_cols = [col for col in stimuli_fish_df.columns if col.startswith('d') and '_stim' in col]

    stimuli_fish_df  = stimuli_fish_df.groupby('trial_id', group_keys=False)[stimuli_fish_df.columns].apply(an.pause_copy_trajectory,
                                                                                        dot_xy_cols=dot_xy_cols,
                                                                                           dot_stim_cols=dot_stim_cols)
    
    
    
    
    #---------- Computes left/right dot counts, converts coordinates to millimeters, and generates absolute stimulus trajectories for each fish across all trials -----#
    
    mask_left = stimuli_fish_df.filter(like='x').gt(0)
    mask_right = stimuli_fish_df.filter(like='x').lt(0)
    stimuli_fish_df['n_dots_left'] = mask_left.sum(axis=1)
    stimuli_fish_df['n_dots_right'] = mask_right.sum(axis=1)

    stimuli_fish_df[stimuli_fish_df.filter(like='x').columns] *= PX_TO_MM
    stimuli_fish_df[stimuli_fish_df.filter(like='y').columns] *= PX_TO_MM

    stimuli_allfish_dict = {}
    for f in range(n_fish):
        abs_trials = []
        for block_id, stim_group in stimuli_fish_df.groupby('stimulus_block_id'):
            # Use the corresponding fish row at trial start (without transformation).
            # (Here, frames_before is 0 as in your code.)
            fish_row = fish_df.loc[stim_group.index[0]]
            trial_abs = an.compute_stimuli_absolute(f, stim_group, fish_row, n_stimuli)
            abs_trials.append(trial_abs)

        stimuli_abs_df = pd.concat(abs_trials)
        extra_cols = stimuli_fish_df.columns[n_stimuli*len(["x", "y", "stim"]):]
        stimuli_abs_df = pd.concat([stimuli_abs_df, stimuli_fish_df[extra_cols]], axis=1)

        stimuli_allfish_dict[f'f{f}'] = stimuli_abs_df
        
        
        
        
    # ---------------------- Invert y axis and fish orientation to ccw ----------------------------#
    fish_df, stimuli_allfish_dict = an.transform_coordinates(fish_df, stimuli_allfish_dict, roi_df)



    # -------------------------------- Calculate dots center --------------------------------------#
    stimuli_allfish_dict = an.compute_stimuli_centers(stimuli_allfish_dict, n_stimuli, mask_left, mask_right, experiment)
    
    
    
    # ------------------------ Calculate how many dots are out of rois -----------------------------#
    stimuli_allfish_dict = an.calculate_dots_out(stimuli_allfish_dict, n_stimuli, roi_df, experiment, mask_left, mask_right)


    # --------------------- Compute absolute positions of fish and stimuli --------------------------#
    # For each fish, create a datafame with fish x, y, ori and all dots x, y stim from stimuli_allfish_dict and frame number and condition. Save as a csv file each fish dataframe.
    abs_dict = {}
    
    for f in range(n_fish):
        fish_key = f'f{f}'
        fish_data = fish_df[[f'f{f}_x', f'f{f}_y', f'f{f}_ori', 'condition', 'frame']].copy()
        stimuli_data = stimuli_allfish_dict[fish_key].copy()
        stimuli_data = stimuli_data.drop(columns=['trial_id', 'condition', 'stimulus_block_id', 'frame_count_stimulus', 'trial_start', 'frame', 'n_dots_left', 'n_dots_right'])
        combined_df = pd.concat([fish_data.reset_index(drop=True), stimuli_data.reset_index(drop=True)], axis=1)
        #remove rowa where dot0_stim is 'none'
        combined_df = combined_df[combined_df['d0_stim'] != 'none']
        combined_df = combined_df.rename(columns={'condition': 'stimulus'})
        #reorder columns by frame, condition, fish x, fish y, fish ori, dot1 x, dot1 y, dot1 stim, dot2 x, dot2 y, dot2 stim, ...
        ordered_columns = ['frame', 'stimulus', f'f{f}_x', f'f{f}_y', f'f{f}_ori']

        for i in range(n_stimuli):
            ordered_columns.extend([f'd{i}_x', f'd{i}_y', f'd{i}_stim'])
        combined_df = combined_df[ordered_columns]
        abs_dict[fish_key] = combined_df
        
        
    # --------------------- Compute polar coordinates of stimuli for each fish --------------------------#
    # For each fish, create a datafame with all dots x, y stim from abs_dict and frame number and condition. Save as a csv file each fish dataframe.    
    polar_dict = {}
        
    n_stimuli = 8    
        
    for fish_key in abs_dict:
        # Copy of the absolute DF for this fish
        df_polar = abs_dict[fish_key].copy()

        # Columns of the fish corresponding to this fish_key (f0, f1, ...)
        fish_x_col = f"{fish_key}_x"
        fish_y_col = f"{fish_key}_y"
        fish_ori_col = f"{fish_key}_ori"

        # Initialize new columns angle / radius for each dot
        for k in range(n_stimuli):
            df_polar[f"d{k}_angle"] = np.nan
            df_polar[f"d{k}_radius"] = np.nan

        # loop on rows
        for idx, row in df_polar.iterrows():
            f_ori = row[fish_ori_col]
            fx = row[fish_x_col]
            fy = row[fish_y_col]

            for k in range(n_stimuli):
                stim_col = f"d{k}_stim"
                x_col = f"d{k}_x"
                y_col = f"d{k}_y"
                # If the columns do not exist (for safety)
                if x_col not in df_polar.columns or y_col not in df_polar.columns:
                    continue
                #Compute coordinates of stimulus in a system centered on the fish
                d_x = row[x_col] - fx
                d_y = row[y_col] - fy
                d_ori = (np.arctan2(d_y, d_x) + 2 * np.pi) % (2 * np.pi)
                if d_ori > f_ori:
                    angle_diff = d_ori - f_ori
                else:
                    angle_diff = 2 * np.pi - (f_ori - d_ori)
                radius = np.sqrt(d_x**2 + d_y**2)
                # Write in the new columns
                df_polar.at[idx, f"d{k}_angle"] = angle_diff
                df_polar.at[idx, f"d{k}_radius"] = radius

        # Drop columns d{k}_x and d{k}_y, but keep d{k}_stim
        drop_cols = []
        for k in range(n_stimuli):
            drop_cols.append(f"d{k}_x")
            drop_cols.append(f"d{k}_y")

        df_polar = df_polar.drop(columns=drop_cols, errors="ignore")
        
        polar_dict[fish_key] = df_polar
        
    
    if output == "absolute":
        return abs_dict
    elif output == "polar":
        return polar_dict          
    elif output == "both":
        return {
            "absolute": abs_dict,
            "polar": polar_dict,
        }
    else:
        raise ValueError("output must be 'absolute', 'polar', or 'both'")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def run_diagnostics(abs_dict=None, polar_dict=None, n_examples=5):
    """
    Run sanity checks on transformed datasets.

    This diagnostic routine can take:
        - abs_dict  : dictionary of absolute-coordinate DataFrames
        - polar_dict: dictionary of polar-coordinate DataFrames
        - n_examples: number of random frames to visually compare

    The function performs three possible checks:
        1. If abs_dict is provided:
            - plots the density of dot angles at trial onset
              (should cluster around ±90° for lateral stimuli)

        2. If polar_dict is provided:
            - plots the density of polar angles at trial onset
              (same expected structure as absolute)

        3. If both are provided:
            - randomly selects example frames and compares:
                * ABSOLUTE view → fish plotted as an oriented arrow
                * POLAR view     → fish at center, facing 0 rad
              The two views must show consistent dot positions.

    Returns
    -------
    None
        The function only displays diagnostic plots.
    """
    
    if abs_dict is None and polar_dict is None:
        raise ValueError("run_diagnostics: you must provide at least abs_dict or polar_dict.")

    print("\n===== RUNNING DIAGNOSTICS =====\n")
    
    fish_key = f'f{random.randint(0, 14)}'
    n_stimuli=8
    
    
    if abs_dict is not None : 
        print("[ABSOLUTE COORDINATES DIAGNOSTIC]")
        print(
            "- We look at the distribution of dot ANGLES in the fish-centered frame\n"
            "  at stimulus onset (frame_count_stimulus == 0).\n"
            "- What you should see:\n"
            "    * For lateral stimuli, the angle density should concentrate\n"
            "      around ±90° (≈ π/2 and 3π/2), i.e. on the sides of the fish.\n"
            "    * You should NOT see most of the density at 0° or 180°\n"
            "      (directly in front of or behind the fish).\n"
        )
        
        df_abs = abs_dict[fish_key]

        # Detect when a trial starts
        df_abs["condition"] = np.where(df_abs["d0_stim"] == "none",
                           "inter_stim_pause",
                           df_abs["d0_stim"])
        df_abs['trial_start'] = (df_abs['condition'] == "inter_stim_pause") & (df_abs['condition'].shift(1) != "inter_stim_pause")
        df_abs['trial_id'] = df_abs['trial_start'].cumsum()
        df_abs['stimulus_block_id'] = (df_abs['condition'] != df_abs['condition'].shift(1)).cumsum()
        df_abs['frame_count_stimulus'] = df_abs.groupby('stimulus_block_id').cumcount()
        df_abs['frame'] = np.arange(len(df_abs))
        
        angle_list = []
        for index, row in df_abs.iterrows():
            f_ori = row[f"{fish_key}_ori"]
            f_x = row[f"{fish_key}_x"]
            f_y = row[f"{fish_key}_y"]
            
            # We are interested only on the beginning of trials
            if row['frame_count_stimulus'] != 0:
                continue
            
            for n_stim in range(8) :
                if row[f"d{n_stim}_stim"] in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue
                d_x = row[f"d{n_stim}_x"]
                d_y = row[f"d{n_stim}_y"]

                d_x_centered = d_x - f_x
                d_y_centered = d_y - f_y

                d_ori = (np.arctan2(d_y_centered, d_x_centered) + 2 * np.pi) % (2 * np.pi)
                if d_ori>f_ori : angle_diff = d_ori - f_ori
                else: angle_diff = 2*np.pi - (f_ori - d_ori)

                angle_list.append(angle_diff)
              

        angle_series = pd.Series(angle_list, name="angle_diff")

        ax = angle_series.plot(kind="density")  # KDE density plot
        ax.set_xlabel("Angle difference (rad)")
        ax.set_ylabel("Density")
        ax.set_title("Circular density of angle between stimulus position and fish orientation for the ABSOLUTE dataframe")
        plt.show()
    
    
    
    
    
    
    if polar_dict is not None : 
        print("[POLAR COORDINATES DIAGNOSTIC]")
        print(
            "- We look at the distribution of dot ANGLES in the fish-centered frame\n"
            "  at stimulus onset (frame_count_stimulus == 0).\n"
            "- What you should see:\n"
            "    * For lateral stimuli, the angle density should concentrate\n"
            "      around ±90° (≈ π/2 and 3π/2), i.e. on the sides of the fish.\n"
            "    * You should NOT see most of the density at 0° or 180°\n"
            "      (directly in front of or behind the fish).\n"
        )
        
        df_polar = polar_dict[fish_key]

        # Detect when a trial starts
        df_polar["condition"] = np.where(df_polar["d0_stim"] == "none",
                           "inter_stim_pause",
                           df_polar["d0_stim"])
        df_polar['trial_start'] = (df_polar['condition'] == "inter_stim_pause") & (df_polar['condition'].shift(1) != "inter_stim_pause")
        df_polar['trial_id'] = df_polar['trial_start'].cumsum()
        df_polar['stimulus_block_id'] = (df_polar['condition'] != df_polar['condition'].shift(1)).cumsum()
        df_polar['frame_count_stimulus'] = df_polar.groupby('stimulus_block_id').cumcount()
        df_polar['frame'] = np.arange(len(df_polar))
        
        angle_list = []
        for index, row in df_polar.iterrows():
            # We are interested only on the beginning of trials
            if row['frame_count_stimulus'] != 0:
                continue
            
            for n_stim in range(8) :
                if row[f"d{n_stim}_stim"] in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue
                d_angle = row[f"d{n_stim}_angle"]
                angle_list.append(d_angle)
              
        angle_series = pd.Series(angle_list, name="angle_diff")

        ax = angle_series.plot(kind="density")  # KDE density plot
        ax.set_xlabel("Angle difference (rad)")
        ax.set_ylabel("Density")
        ax.set_title("Circular density of angle between stimulus position and fish orientation for the POLAR dataframe")
        plt.show()
        
        
        
        
    
    
    if abs_dict is not None and polar_dict is not None :
        print("[ABSOLUTE vs POLAR VISUAL DIAGNOSTIC]")
        print(
            "- We compare the SAME randomly chosen trial frame under two coordinate systems:\n"
            "    1) The ABSOLUTE view  → fish shown as a blue arrow in the arena\n"
            "    2) The POLAR view      → fish at the center, facing angle 0 rad (to the right)\n"
            "\n"
            "- What you should see:\n"
            "    * The dots should appear in CONSISTENT relative positions across both plots.\n"
            "      For example, if in the ABSOLUTE view the dots are on the left of the fish,\n"
            "      then in the POLAR view all dot angles should cluster around ±90°.\n"
            "\n"
            "    * A mismatch indicates a transformation error:\n"
            "         - If ABSOLUTE shows dots in front/behind the fish but POLAR shows them left/right → wrong centering/orientation.\n"
            "         - If ABSOLUTE shows left dots and POLAR shows angles near 0° or 180° → wrong angle computation.\n"
            "\n"
            "- In short:\n"
            "      >>> The two plots MUST tell the same story.\n"
            "      If the geometry doesn't match, there is an issue in the coordinate transform.\n"
        )   
        df_abs = abs_dict[fish_key]
        df_pol = polar_dict[fish_key]  
        # We are looking for frames with activated stimulus
        #    et qui ont au moins un stimulus actif
        possible_frames = []

        for idx_abs, row_abs in df_abs.iterrows():
            frame_val = row_abs.get("frame", idx_abs)

            # ligne correspondante dans df_pol (même frame)
            candidates_pol = df_pol[df_pol.get("frame", df_pol.index) == frame_val]
            if candidates_pol.empty:
                continue

            # vérifier s'il y a au moins un stimulus actif
            has_stim = False
            for k in range(n_stimuli):
                stim_col = f"d{k}_stim"
                if stim_col in df_abs.columns:
                    val = row_abs[stim_col]
                    if val not in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                        has_stim = True
                        break

            if has_stim:
                possible_frames.append((idx_abs, candidates_pol.index[0]))

        if not possible_frames:
            print(f"No frames with active stimuli found for {fish_key}.")
            return

        n_examples = min(n_examples, len(possible_frames))
        chosen_pairs = random.sample(possible_frames, n_examples)

        # Plot for eaxh xhosen frame
        for idx_abs, idx_pol in chosen_pairs:
            row_abs = df_abs.loc[idx_abs]
            row_pol = df_pol.loc[idx_pol]

            fish_x_col = f"{fish_key}_x"
            fish_y_col = f"{fish_key}_y"
            fish_ori_col = f"{fish_key}_ori"

            f_x = row_abs[fish_x_col]
            f_y = row_abs[fish_y_col]
            f_ori = row_abs[fish_ori_col]  # CCW, 0 = vers la droite

            fig = plt.figure(figsize=(10, 4))
            ax_abs = fig.add_subplot(1, 2, 1)
            ax_pol = fig.add_subplot(1, 2, 2, projection="polar")

            # ---------- SUBPLOT 1 : ABSOLUTE VIEW ----------
            arrow_len = 20
            ux = np.cos(f_ori)
            uy = np.sin(f_ori)

            ax_abs.scatter(f_x, f_y, s=50, color="tab:blue")
            ax_abs.arrow(
                f_x, f_y,
                arrow_len * ux, arrow_len * uy,
                head_width=5,
                head_length=8,
                length_includes_head=True,
                color="tab:blue",
                linewidth=1.5
            )

            xs_abs = [f_x]
            ys_abs = [f_y]

            for k in range(n_stimuli):
                stim_col = f"d{k}_stim"
                x_col = f"d{k}_x"
                y_col = f"d{k}_y"

                if stim_col not in df_abs.columns:
                    continue

                stim_val = row_abs[stim_col]
                if stim_val in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue

                if x_col not in df_abs.columns or y_col not in df_abs.columns:
                    continue

                dx = row_abs[x_col]
                dy = row_abs[y_col]
                if np.isnan(dx) or np.isnan(dy):
                    continue

                ax_abs.scatter(dx, dy, s=40, color="tab:orange")
                ax_abs.text(dx, dy, str(k), fontsize=7, ha="center", va="center")

                xs_abs.append(dx)
                ys_abs.append(dy)

            if len(xs_abs) > 1:
                margin = 20
                xmin, xmax = min(xs_abs) - margin, max(xs_abs) + margin
                ymin, ymax = min(ys_abs) - margin, max(ys_abs) + margin
                ax_abs.set_xlim(xmin, xmax)
                ax_abs.set_ylim(ymin, ymax)

            ax_abs.set_aspect("equal", adjustable="box")
            ax_abs.set_title(f"ABS – {fish_key}, frame {int(row_abs.get('frame', idx_abs))}")
            ax_abs.grid(alpha=0.3, linestyle="--", linewidth=0.5)

            # ---------- SUBPLOT 2 : POLAR VIEW ----------
            # fish at center, looking at angle 0 (right)
            # --- draw fish orientation arrow in polar view ---
            head_r = 3.0  # arrow length
            arrow_theta = 0  # fish looks toward angle 0 rad (to the right)

            # arrow: from (angle 0, radius 0) to (angle 0, radius head_r)
            ax_pol.annotate(
                "", 
                xy=(arrow_theta, head_r),     # end of arrow
                xytext=(arrow_theta, 0),      # start of arrow
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2)
            )

            radii_pol = []
            for k in range(n_stimuli):
                stim_col = f"d{k}_stim"
                r_col = f"d{k}_radius"
                a_col = f"d{k}_angle"

                if stim_col not in df_pol.columns or r_col not in df_pol.columns or a_col not in df_pol.columns:
                    continue

                stim_val = row_pol[stim_col]
                if stim_val in ["none", None, np.nan, "bout", "inter_stim_pause"]:
                    continue

                r = row_pol[r_col]
                a = row_pol[a_col]
                if np.isnan(r) or np.isnan(a):
                    continue

                radii_pol.append(r)
                ax_pol.scatter(a, r, s=40, color="tab:orange")
                ax_pol.text(a, r, str(k), fontsize=7, ha="center", va="center")

            if radii_pol:
                Rpol = max(radii_pol) * 1.2
            else:
                Rpol = 1.0
            ax_pol.set_ylim(0, Rpol)

            ax_pol.set_title(f"POLAR – {fish_key}, frame {int(row_pol.get('frame', idx_pol))}")
            ax_pol.set_theta_zero_location("E")  # 0 rad = à droite
            ax_pol.set_theta_direction(1)        # CCW

            ax_pol.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        
        

        