# Experiment Data Overview

## Experiment Description

This experiment investigates zebrafish behavior in response to visual stimuli consisting of groups of dots.  
Each fish swims freely inside a Petri dish of **10 cm diameter** while being recorded at **30 frames per second**.

At specific time points defined by the *stimuli trajectory file*, groups of dots appear either on the left or right side of each fish.  
The dots move within a **2.5 × 2.5 cm** area and maintain a **minimum distance of 0.5 cm** from the fish center.

Stimuli vary in:
- **Group size:** 2, 4, 6, or 8 dots  
- **Speed:** 0 cm/s (static) or 3.5 cm/s (moving)  

Each trial presents **one stimulus**. The total experiment duration depends on the length of the trajectory file.  
Each experiment includes **15 fish** recorded simultaneously.

Before stimulus onset, fish had an habituation period of **20 minutes** in the setup without visual stimulation.  
Each stimulus presentation lasts **60 seconds**, followed by a **50-second pause** with no visual input.  

---

**Files referenced in this dataset:**
- **ROI file** — `ROIdef2025-02-20T12_09_16.csv` loaded as 'roi_df' in the AllFish_GsizeMotionSingle notebook.  
  Contains the geometric information for each Petri dish (dish center, radius, projector offsets).  
  This file is used to correctly project visual stimuli onto each dish.
- **Fish position file** `PositionTxt_allROI2025-02-20T15_37_59.csv` loaded as 'df' in the AllFish_GsizeMotionSingle notebook.  
  This file is the output of the tracking system. It contains x,y and orientation of each fish across the entire experiment duration. 
- **Stimuli trajectory file** `2025.02.20-0957_trajectory_GsizeMotionSingle_180min.csv` loaded as 'stimuli_df' in the AllFish_GsizeMotionSingle notebook.  
  This file defines the stimuli trajectory, with dot positions expressed in millimeters relative to the fish’s centroid at stimulus onset.  
  To obtain the actual stimulus position seen by each fish, these relative coordinates must be transformed using the corresponding fish position and orientation from the PositionTxt file. Because each fish has a different trajectory, each one ends up with a different “real” stimulus path.

#### ⚠️ Warning  
Do not use the stimulus-related columns in the PositionTxt File. These values DO NOT correspond to the real stimulus position and must be ignore. **Stimuli positions** should be retrieved only from **stimuli trajectory file**!

---

## PositionTxt File

This file contains **fish positions**. 
Each row represents one frame. Columns include the positions and orientations of all fish, as well as stimulus information.

| Column Content | Description |
|-----------------|--------------|
| Fish positions  | `x`, `y`, and `orientation` for each fish (15 fish × 3 columns = 45 columns total) |
| Stimulus data   | `x`, `y`, and `stimulus_name` for each dot (8 dots) |

For analysis, **only the first 45 columns** (fish data) are relevant.  
Stimulus positions should be retrieved from the stimuli trajectory file and transformed relative to the fish position at the start of each trial. 

Coordinate system details:
- **Fish positions (`x`, `y`)** are in *absolute pixel coordinates*.  
- The origin (0,0) is at the **top-left corner**.  
- **Orientation** values are in *radians*, measured *clockwise*.

---

## Trajectory File
This file defines the stimulus positions and is the one read by our Bonsai pipeline to generate the visual stimuli in the virtual reality setup using a semi-closed loop configuration.

Each row corresponds to one frame.  
Columns include stimulus properties:

| Column | Description |
|---------|--------------|
| `name` | Stimulus name (e.g., `CLsemi1800_g0-2_s3.5-3.5`) |
| `x`, `y` | Dot coordinates in **millimeters**, relative to the fish centroid |
| `size` | Dot size |
| `color` | Dot color |
| `background_color` | Background color |

Coordinate conventions:
- Left side → **positive x values**  
- Right side → **negative x values**  
  (This convention originates from the Bonsai stimulus generation code.)


#### Stimulus Naming Convention

Example: `CLsemi1800_g0-2_s3.5-3.5`

| Component | Meaning |
|------------|----------|
| `CLsemi` | Semiclosed-loop experiment — dots appear relative to the fish position at onset and remain fixed thereafter |
| `g0-2` | Group configuration — 0 dots on the left, 2 dots on the right |
| `s3.5-3.5` | Speed configuration — 3.5 cm/s for the right-side stimulus (values can be 0 or 3.5) |

When no stimulus is presented, the `name` column contains the value **`none`**.

---

## ROI File

Each row corresponds to a Petri dish (one fish).  
Columns include spatial and geometric parameters:

| Column | Description |
|---------|-------------|
| `xoff`, `yoff` | Offset coordinates in projector screen pixels (origin at top-left) |
| `diameter` | Diameter of the Petri dish |
| `x`, `y` | Dish center coordinates |
| `radius` | Radius of the dish |

These parameters define the position of each dish on the projection screen and are used to align the visual stimulus with the corresponding fish.
