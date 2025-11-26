# Experiment Data Overview

**Recommended dataset for analysis**  
We recommend that you always start from the three original experiment files and run the Python pipeline:

- `Trajectory` file (stimuli trajectories, fish-centered, in mm)  
- `PositionTxt` file (fish positions and orientations, in pixels)  
- `ROI` file (dish geometry and screen offsets, in pixels)

These raw files strictly reflect the experimental setup, but they live in different coordinate systems (pixels vs millimeters, top-left vs bottom-left origin, clockwise vs counterclockwise orientation, etc.).  
The function `generate_transformed_datasets()` in `src/transformed_data_pipeline.py` applies all the necessary conversions and bookkeeping (trial structure, pauses, onset frames, coordinate transforms) to produce a clean, analysis-ready representation.

Usage example:

```python
from src import transformed_data_pipeline as tdp

# Load the three raw files (Trajectory, PositionTxt, ROI)
stimuli_df = ...
position_df = ...
roi_df = ...

results = tdp.generate_transformed_datasets(stimuli_df, df, roi_df)

abs_dict = results["absolute"]  # absolute (screen) coordinates
polar_dict = results["polar"]    # fish-centered polar coordinates

```

Both abs_dict and polar_dict are dictionaries indexed by fish ID:
 - abs_dict["f0"] → DataFrame for fish 0
 - abs_dict["f10"] → DataFrame for fish 10
 - 	same structure for polar_dict

In abs_dict, fish and dots are in a common, absolute screen coordinate system (pixels, CCW orientation).
In polar_dict, the fish is the origin and looks toward angle 0; each dot is described by (dX_angle, dX_radius) around the fish.
Further details on the columns of these DataFrames are given in the next section.

---

## 1. Experiment Description

This experiment examines how zebrafish respond to visual stimuli composed of groups of dots—either static or moving.  
Each fish swims freely inside a **10 cm diameter** dish while being tracked at **30 frames per second**.

At specific time points defined by the **Stimuli Trajectory File**, dots appear either on the left or the right of each fish.  
Dots move inside a **2.5 × 2.5 cm** region and always maintain at least **0.5 cm** distance from the fish’s center.

Stimulus properties:

- **Group size:** 2, 4, 6, or 8 dots  
- **Speed:** 0 cm/s (static) or 3.5 cm/s (moving)

Each experiment includes:
- **15 fish recorded simultaneously**  
- **20 minutes of habituation** 

Each stimulus lasts **60 seconds**, followed by a **50-second interstimulus pause** with a white background. Every stimulus condition (e.g. 3 dots static left) are repeated across multiple trials, but the specific dot trajectories are newly generated each time.

---

## 2. ABSOLUTE Transformed Fish and Stimuli dataframe 
**`contained in abs_dict, output of pipeline`**

It contains the **actual, absolute positions of all dots** for every frame and every fish, fully transformed in the same coordinate system. The trajectory file provides dot positions *relative to each fish at stimulus onset*.  
Since each fish moves differently across a trial, these coordinates must be transformed into absolute coordinates using each fish’s position and orientation at each stimulus starting frame.

Content:
- `frame` — frame index (30 fps)
- `source_frame` — original frame index from the raw PositionTxt/Trajectory files (before any trial-based realignment)
- `stimulus` - stimulus diplayed at that frame
- `fX_x`, `fX_y`, `fX_ori` — fish centroid and orientation
- `dX_x`, `dX_y` `dX_stim`— absolute position of each dot and stimulus name.  

The maximum number of dots that can be displayed can be 8, but smaller group sizes are also showed. When dotX_stim has value 'none', that dot is not being shown. E.g. if the are two dots on the right, then dot0 and dot1 will have useful values, all the other stimuli columns values should be ignored.

Stimulus naming convention example: `CLsemi1800_g0-2_s3.5-3.5`

- `CLsemi` — semi–closed-loop (stimuli positioned relative to fish only at onset)  
- `g0-2` — 0 dots left, 2 dots right  
- `s3.5-3.5` — right-side stimulus moves at 3.5 cm/s

During pauses, the name is `inter_stim_pause`.

Coordinates:
- Fish and dots coordinates are in **pixels**. Fish move in a dish of around 350-80 px (see ROI file).
- Origin at **bottom-left**
- Orientation measured in **counterclockwise radians**

---

## 3. POLAR Transformed Fish and Stimuli dataframe
**`contained in polar_dict, output of pipeline`**

This dataframe mirrors the structure of the absolute-coordinate dataframe (`abs_dict`).  
It contains the same rows, the same indexing (`frame`, `source_frame`), the same stimulus labels, and the same fish metadata (`fX_x`, `fX_y`, `fX_ori`).  

The only difference is how dot positions are represented.

Dot coordinates are expressed in *egocentric polar form*, relative to the fish’s orientation at stimulus onset:

- `dX_radius` — distance from the fish to dot X  
- `dX_angle` — angular position of dot X in the fish-centered coordinate frame (0 radians corresponds to straight ahead; values increase counterclockwise)

Stimulus identity is stored in:

- `dX_stim` — stimulus name (same conventions as in the absolute dataframe)

All other fields keep the same meaning as in `abs_dict`.

Purpose of the polar dataframe :
This representation describes the visual scene *from the fish’s own perspective*.  
It is particularly useful for analyses based on:

- angular response distributions  
- left/right asymmetries  
- egocentric tuning curves  
- behaviour models that depend on relative rather than absolute positions

Because radius and angle correspond directly to what the fish "sees", this version is recommended for perceptual or behaviourally aligned analyses.

---

## 4. Fish Position File
**`PositionTxt_allROI2025-02-20T15_37_59.csv`**

This file contains the **tracked positions of all 15 fish** across the full experiment. It is the output of Bonsai tracking during the experiment.

Content:
- For each frame:  
  - `x`, `y`, and `orientation` for each fish (45 columns total)
  - `x`, `y`, and `name` for each dot. (24 columns total as there are 8 max dots across the experiment) 

  ⚠️ The stimulus columns in this file should be ignored.  
These values are *not* meaningful for this experiment and should be ignored.  
Use the **trajectory file** or the **transformed file** for valid stimulus positions.

Coordinates:
- Fish coordinates are in **pixels**. Fish move in a dish of around 350-80 px.
- Origin at **top-left**
- Orientation measured in **clockwise radians**

---

## 5. Stimuli Trajectory File  
**`2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv`**

This file defines the desired **stimulus trajectories** and is directly read by the **Bonsai pipeline** to generate visual stimuli in our semi–closed-loop virtual reality system. These trajectories represent the *intended* dot positions, which are then displayed relative to the fish’s location at specific time points.

Content:
- One row per frame  
- For each dot:  
  `name`, `x`, `y`, `size`, `color`, `background_color`
  There are 8 dots, so 48 columns in total. 

Naming convention example: `CLsemi1800_g0-2_s3.5-3.5`

- `CLsemi` — semi–closed-loop (stimuli positioned relative to fish only at onset)  
- `g0-2` — 0 dots left, 2 dots right  
- `s3.5-3.5` — right-side stimulus moves at 3.5 cm/s

During pauses, the name is `inter_stim_pause`.

Coordinates:
- Expressed in **millimeters**, relative to the fish centroid at stimulus onset  
- Positive x → stimulus appears on the **left** of the fish
- Negative x → stimulus appears on the **right** of the fish
- Positive y → stimulus appears on the **front** of the fish
- Negative y → stimulus appears on the **back** of the fish

---

## 6. ROI File
**`ROIdef2025-02-20T12_09_16.csv`**

Each row corresponds to one of the 15 Petri dishes.  
This file supplies the geometric information needed to project stimuli accurately onto each dish.

Content:
- `xoff`, `yoff` — projector offset 
- `diameter` — dish diameter  
- `x`, `y` — center position on projector  
- `radius` — dish radius

Coordinates:
- Values are in **pixels**
- Origin at **top-left**

These parameters are used when transforming the relative stimulus coordinates into absolute coordinates.