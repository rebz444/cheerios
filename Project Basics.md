# Project Basics

> **What this page is.** Stable, slow-changing description of the project — task, animals, recording setup, theoretical framing, and chapter-level status. Analytical methods live on the Methods page; specific results live in the Findings database; unresolved questions live on Open Questions.
> 

## Project overview

The thesis investigates temporal reward prediction and time investment decision making in mice using a novel probabilistic task. Unlike prior deterministic timing tasks (Shuler & Bear; Hamilos et al.), the task uses a probabilistic reward function.

**Thesis claims under investigation (revised May 2026).** The project is organized around three nested claims:

1. **Reward history modulates clock speed.** The previous trial's outcome shifts a neural "clock speed" signal on the current trial, in a way that explains — or at minimum covaries with — the trial-to-trial wait-time changes observed behaviorally.
2. **Different brain regions support this modulation differently.** Across striatum, motor cortex, visual cortex, thalamus, and hippocampus, the strength and form of (a) elapsed-time encoding, (b) clock-speed extractability, and (c) reward-history modulation differs. "Some regions do it, some don't" is a valid outcome; "every region does it identically" is not.
3. **The cortico-striatal-thalamic circuit enables these computations.** Inter-regional spike-time relationships (CCG, Granger, communication subspace) constrain which regions lead, lag, and read out the timing / clock-speed signal.

Dopamine remains the proposed upstream modulator of clock speed (Chapter 4), but is **paused** until the ephys side of the story stabilizes. The earlier DMS-and-DA-centric narrative (reward → DA → DMS clock speed → wait time) is preserved as the Chapter 4 working hypothesis but is no longer the headline framing.

**Experimental scope.** All results to date are purely observational: no optogenetic, pharmacological, or behavioral manipulations. Transgenic lines are limited to sensor expression for photometry (GRAB-DA, GCaMP, rCaMP).

## The behavior task

```
cue_on                    cue_off                         decision
   │                         │                                │
   │◄─────── BACKGROUND ────►│◄─────────── WAIT ─────────────►│◄── CONSUMPTION
   │       (light on)        │         (light off)            │    (reward)
   │                         │                                │
   │   [licks CAN happen]    │    [NO licks until decision]   │   [licking]
   │           ↓             │                                │
   │  BACKGROUND restarts    │                                │
   │                         │                                │
   ▼                         ▼                                ▼
```

### Three Trial Periods

| Period | Start | End | What happens | Group difference |
| --- | --- | --- | --- | --- |
| **Background** | cue_on | cue_off |   • Visual cue (light) on. 
  • Mouse can lick, but licks trigger the background period to restart. 
  • If licked and BG restarts, 
      ◦ the same originally drawn bg duration is used. 
      ◦ the cue remains on
      ◦ mouse is not informed of the restart
  • the experienced BG duration may exceed the drawn duration due to restarts.  | Two groups:
  • **Long BG**: mean 5s, ±20% (uniform)
  • **Short BG**: mean 1s, ±20% (uniform) |
| **Wait** | cue_off | decision |   • Visual cue off indicates wait period start.
  • Mouse waits. Longer the wait, larger the reward probability. 
  • First lick = decision. Time waited = decision lick time - wait start time.
  • No other licks can occur between cue_off and decision by definition.
  • reward probability functions: $p(t) = 0.9 \left(1 - e^{-t/3} \right)$
  • max wait time is 60s; miss trial if no decision lick within 60s. | both groups share the same reward probability function |
| **Consumption** | decision | trial end |   • Starts immediately after decision lick regardless of reward delivered or not. 
  • Fixed duration of 3 sec. 
  • If rewarded, reward delivered
      ◦ reward calculated based on the reward probability function
      ◦ asymptotically approaches 0.9
  • Next trial’s bg starts immediately after
  • no consumption if no decision lick during the wait period | no difference |

### Optimal behavior

Expected reward rate per trial:

$\text{rate}(t_w) = \frac{p(t_w) \cdot R}{t_{bg} + t_w + t_c}$

With $t_c = 3\,\text{s}$ fixed and $t_{bg}$ fixed per group (5s for Long BG, 1s for Short BG), the only free variable is $t_w$. Optimizing yields:

- **Short BG optimal** $t_w$**: 3.86s**
- **Long BG optimal** $t_w$**: 5.03s**

A shorter background raises the opportunity cost of each additional second of waiting, so the optimal strategy commits earlier.

Empirical group means:

- **Short BG wait: 2.08s** (undershoots optimal by ~1.8s)
- **Long BG wait: 4.29s** (undershoots optimal by ~0.7s)

Both groups undershoot the reward-rate-optimal wait, with Short BG undershooting more. This is the behavioral starting point, and one thing the RL / predictive model needs to reproduce.

### The Three Anchors

We analyze DA dynamics relative to three possible reference points (anchors):

```
                 last_lick                                      decision
cue_on               │               cue_off                        │
   │                 │                  │                           │
   │◄────────────────┼──────────────────┼───────────────────────────┤
   │            ANCHOR 1: cue_on → decision                         │
   │                 │                  │                           │
   │                 │                  │◄──────────────────────────┤
   │                 │                  │  ANCHOR 2: cue_off        │
   │                 │                  │     → decision            │
   │                 │                  │                           │
   │                 │◄─────────────────┼───────────────────────────┤
   │                 │       ANCHOR 3: last_lick → decision         │
   │                 │                  │                           │
   │◄─ BACKGROUND ──►│                  │◄──────── WAIT ───────────►│
                                          (no licks until decision)
```

| Anchor | Window | Column | Interpretation |
| --- | --- | --- | --- |
| **cue_on** | cue_on → decision | `time_waited_since_cue_on` | External reference. Includes background period. Long BG mice may use this. |
| **cue_off** | cue_off → decision | `time_waited` | Wait period only. Pure timing from cue offset (light off). |
| **last_lick** | last_lick → decision | `time_waited_since_last_lick` | Self-generated reference. Can span trials if no background licks. Short BG mice may use this. |

#### Note on last_lick Spanning Trials

The `last_lick` anchor is special: if the mouse didn't lick during the background period, the "last lick" is from the **previous trial** (during consumption). This means `time_waited_since_last_lick` can be longer than the current trial duration.

This is behaviorally meaningful — it captures timing from the mouse's last self-generated action, regardless of trial boundaries.

### Training

Mice require **≥20 days of daily training** before reaching expert performance. Because this task design has no prior published implementation, expert-performance criteria were not established a priori — training is therefore extended well past the minimum to verify that performance has plateaued.

Recording is performed on one mouse at a time; remaining animals continue training until their recording slot. This is why cleaned behavior sessions (1,566) substantially exceed ephys sessions (93): most sessions are training sessions from mice awaiting recording, or from mice that were never ultimately recorded.

### Task rig & data pipeline

- **Rig controller:** Raspberry Pi.
- **Task code:** [krave](https://github.com/rebz444/krave.git) — defines trial logic and writes raw event data (lick times, cue transitions, reward delivery, etc.).
- **Preprocessing:** `luckycharm` (first pipeline scripts) — parses raw events into trial-level dataframes used throughout behavior analysis.

---

## Dataset scale

### Behavior (all cleaned sessions)

| Group | Mice | Sessions | Trials |
| --- | --- | --- | --- |
| Short BG | 14 | 682 | 290,400 |
| Long BG | 17 | 884 | 303,321 |
| **Total** | **31** | **1,566** | **593,721** |

### Neuropixels ephys

| Group | Mice | Sessions | Trials | Units |
| --- | --- | --- | --- | --- |
| Short BG | 10 | 49 | 20,991 | 1,728 |
| Long BG | 10 | 44 | 12,550 | 1,320 |
| **Total** | **20** | **93** | **33,541** | **3,048** |

Units by region: STR 2,373 · V1 675.

**Mouse IDs.**

- **Short BG:** RZ034, RZ036, RZ047, RZ049, RZ050, RZ051, RZ061, RZ062, RZ063, RZ070
- **Long BG:** RZ037, RZ038, RZ039, RZ052, RZ053, RZ055, RZ057, RZ058, RZ059, RZ065

Histology status for these mice is detailed in the Histology section below — only a subset has completed clearing + registration and currently enters downstream analyses.

### Fiber photometry (pilot)

| Group | Mice | Sessions | Trials |
| --- | --- | --- | --- |
| Short BG | 2 | 28 | 19,455 |
| Long BG | 4 | 44 | 15,929 |
| **Total** | **6** | **72** | **35,384** |

---

## Recording modalities

### **Electrophysiology**

recording log: 

https://docs.google.com/spreadsheets/d/1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo/edit?usp=sharing

Acute simultaneous V1 + DMS recordings with two Neuropixels 1.0 probes in well-trained mice.

**Surgery (day before recording).** Stereotaxic craniotomies opened over DMS and V1.

| Site | AP | ML | DV |
| --- | --- | --- | --- |
| DMS | +0.9 mm (bregma) | ±1.5 mm | 4.0 mm from pia |
| V1 | lambda | ±3.0 mm (lambda) | 2.0 mm from pia |

**Probes.** Neuropixels 1.0, 1 shank, 384 channels selectable from 960 sites. Bottom 384 channels selected for both probes. Shanks coated with Vybrant DiI or CM-DiI prior to insertion for post-hoc trajectory reconstruction.

**Insertion geometry.** Both probes inserted on the same side of the brain.

| Probe | Depth | Angle |
| --- | --- | --- |
| DMS | 4000 µm | 15° backward (anterior → posterior) |
| V1 | 2000 µm | 15° forward (posterior → anterior) |

**Recording.** Acute session while the mouse performs the task. Recording log (Google Sheets): [recording log](https://docs.google.com/spreadsheets/d/1xpb52EO7BII-_5zIcB2HaOpZqepFcYTvBYgxA4zLyNo/edit?usp=sharing).

**Spike sorting.** [jhu_shuler-lab](https://github.com/rebz444/jhu_shuler-lab.git) pipeline. Runs Kilosort 2.5 and Kilosort 3 in parallel and keeps consensus units across the two sorters.

**Curation.** Consensus units manually curated in Phy and labeled as `good`, `mua`, or `noise`.

**Output for analysis.** All labeled units downloaded; region-assignment (see Histology below) and waveform-based quality checks determine which subset enters each downstream analysis.

![image.png](Project%20Basics/image.png)

### **Fiber photometry**

Pilot dataset — simultaneous recording of three indicators in 6 mice (4 Long BG, 2 Short BG). See Dataset scale above for session / trial counts.

**Signals.**

| Indicator | Measures | Location |
| --- | --- | --- |
| GRAB-DA | Dopamine release | Left DMS |
| GCaMP | Calcium (neuronal activity) | Right DMS |
| rCaMP | Calcium (neuronal activity) | Bilateral V1 |

The rCaMP-in-striatal-fiber measurement (picking up V1 axons terminating in DMS via the left-DMS fiber) is a uniquely novel opportunity — not available in prior V1 timing studies.

**Processing & analysis.** [raisinbran](https://github.com/rebz444/raisinbran.git) — both signal processing and downstream analyses.

---

## Histology

**Pipeline.**

1. **Clearing.** SHIELD → Delipid → RIMS.
2. **Imaging.** Lightsheet microscopy.
3. **Stitching.** Custom pipeline on lab PC (not yet on GitHub).
4. **Registration.** Align stitched volume to Allen Brain Atlas.
5. **Probe tracing.** Manual tracing using [brainglobe](https://brainglobe.info).
6. **Unit assignment.** Traces fed into `cheerios`, which assigns each unit to a brain region by peak channel. Waveform used as a secondary check.
7. **Analysis subset.** Region- and waveform-confirmed units enter downstream neural analyses.

**Status (ephys cohort of 20 mice).**

- **Fully through clearing + registration — used downstream (20 mice):**
    - RZ052 has probe tracks but is currently excluded.

---

## Key Hypotheses and Correlations

> **Updated framing (May 2026).** The causal chain remains the working organization, but the link content has been generalized beyond DMS-specific and DA-specific claims. The DA-related instantiation of Link 1 (reward → DA → clock speed) is held in reserve until the ephys-only version of the chain is resolved (see Chapter 4 status).
> 

### The Causal Chain

```
Trial n−1                    Trial n                        Trial n
─────────                    ───────                        ───────
 OUTCOME      ─ Link 1 ─►    DA / NEURAL DYNAMICS   ─ Link 2 ─►   WAIT TIME
(rewarded?)                  (baseline, slope,                    (decision
                              neural clock)                        time)
```

- **Link 1** — How does the previous trial's outcome modulate the current trial's neural clock speed in each region of interest? Which regions show this modulation, in what direction, and at what magnitude?
- **Link 2** — Does the trial's clock speed correlate with (and ideally precede within-trial) the trial's wait time? Within a trial, does early-window clock speed predict the eventual decision time?

---

## Column Reference

| Column | Meaning |
| --- | --- |
| `time_waited` | cue_off → decision (wait period duration) |
| `time_waited_since_cue_on` | cue_on → decision (includes background) |
| `time_waited_since_last_lick` | last_lick → decision (can span trials) |
| `reward` | Current trial reward (0.0 or 5.0) |
| `previous_trial_reward` | Previous trial reward (0.0 or 5.0) |
| `miss_trial` | True if no decision within 60s |
| `bg_length` | Background period duration (cue_on → cue_off) includes bg restarts |
| `trial_time` | Time within trial (relative to cue_on) |
| `decision_time` | When decision occurred (relative to cue_on) |
| `dff_zscored` | Z-scored DA signal |

## Theoretical framework

- **Pacemaker-accumulator framing.** A neural clock-speed signal — operationalized for now as the slope of decoded time vs true time per trial — is hypothesized to be modulated by reward history. Faster clock after reward → reaches criterion sooner → shorter wait. Caveat: decoder slope is correlated with the wait it predicts, so the confound has to be controlled (within-quartile, and ideally within-trial early-window precedence).
- **Distributed timing code.** Multiple regions of the cortico-striatal-thalamic-hippocampal network are known to carry temporal signals (Buhusi & Meck 2005; Mello 2015; Wang & Jazayeri 2018). Whether they all carry the clock-speed-modulated-by-reward component, or whether this is restricted to specific regions / cell types, is an open question this thesis addresses.
- **Encoding ambiguity.** In a probabilistic task, "timing cells" could encode elapsed time, hazard rate, cumulative reward probability, or subjective value — correlated but dissociable. GLM is the tool that lets us decompose this per unit.
- **Edge-cell caveat.** Many units that appear to rescale may in fact be firing only at the interval edges (cue-locked or pre-motor / lick-locked). These confound the population timing claim. Cell typology decomposition is the planned correction.

## Chapter status

| Chapter | Status |
| --- | --- |
| 1. Behavior | Basics completed (ΔT result robust, R2M-corrected, group anchors validated). Predictive model pending. Behavioral strategy classification (reactive vs timing) open — GMM is not a reliable classifier. |
| 2. Cross-region neural timing codes | Rescaling and decoder run across 11 cell sets in 5 regions (STR, MC, V1, Thal, HPF). Two confound rounds in progress: (a) rescaling edge-cell contamination (resolution: τ-stratification → crop-and-resort → full-PSTH similarity); (b) decoder single-session bottleneck (resolution: pseudo-population in parallel with simultaneous track + mixed-effects pooling). Cell typology decomposition planned as the prerequisite for regional-differentiation claims. GLM in progress. |
| 3. Cortico-striatal-thalamic circuit | V1→DMS CCG complete (V1 leads, ~46.5 ms — polysynaptic / network-level rather than monosynaptic). Region focus to be decided after Chapter 2 settles. Candidate next analyses: MC→DMS CCG (anatomical positive control), VAL→MC and VAL→DMS (thalamic loop), Granger / communication subspace on simultaneous V1↔DMS sessions conditioned on clock speed. |
| 4. Dopamine modulation | <strong>Paused</strong> pending ephys-side resolution. Pilot photometry complete; DA / decoder direction tension (Short BG DA ramp shallows after reward; decoder clock speed faster after reward) remains the unresolved theoretical issue. rCaMP-in-striatal-fiber analysis is a uniquely novel opportunity to revisit when photometry resumes. |
| 5. Histology | Pipeline working, ~11/20 brains through clearing + registration. Continuing on remaining mice. Gates ephys interpretation for any new analyses on uncleared mice. |

## Repositories

| Purpose | Repo / Path |
| --- | --- |
| Behavior task code (runs on Pi) | GitHub: https://github.com/rebz444/krave.git |
| Behavior analysis | Local: `/Users/rebekahzhang/Documents/code/luckycharm` |
| Spike sorting pipeline | GitHub: https://github.com/rebz444/jhu_shuler-lab.git |
| Ephys analysis | GitHub:  · Local: `/Users/rebekahzhang/Documents/code/cheerios` |
| Photometry (processing + analysis) | GitHub:  · Local: `/Users/rebekahzhang/Documents/code/raisinbran` |
| Probe reconstruction (stitching + brainglobe tracing) | Github: https://github.com/rebz444/honeycomb.git |

## Publication target

Neuron, Current Biology, or eLife. Strongest novelty claims: novel probabilistic time-investment task; cross-region comparison of clock-speed modulation by reward history; cortico-striatal-thalamic circuit signatures during self-timed waiting (V1 leading DMS plus complementary CCGs).