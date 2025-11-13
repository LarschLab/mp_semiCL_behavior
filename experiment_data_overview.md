# Experiment Data Overview

**Recommended dataset for analysis:**  
Use the **Transformed Fish and Stimuli File** as your main reference.  
This file already contains stimuli transformed into the same coordinate system as the fish data, aligned per trial and per fish.  

The remaining files are documented so you can understand how this transformed dataset is produced:  
where each variable originates, how coordinate conversions work, and how different pieces link together.

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

## 2. Transformed Fish and Stimuli File
**`2025.02.20_GsizeMotionSingle_f0_transformed_abs`**

It contains the **actual, absolute positions of all dots** for every frame and every fish, fully transformed in the same coordinate system. The trajectory file provides dot positions *relative to each fish at stimulus onset*.  
Since each fish moves differently across a trial, these coordinates must be transformed into absolute coordinates using each fish’s position and orientation at each stimulus starting frame.

Content:
- `frame` — frame index (30 fps)
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

## 3. Fish Position File
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

## 4. Stimuli Trajectory File  
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
- Positive x → stimulus appears on the **left**  
- Negative x → stimulus appears on the **right**

---

## 5. ROI File
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