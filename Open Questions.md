# Open Questions

> **What this page is.** Living list of what's still unresolved, organized by chapter. Each item: one line describing the question, a line on why it matters, and the proposed resolution path. Close out items as Findings entries are added.
> 

## Framing (revised May 2026)

The thesis is organized around three nested claims (see Project Basics):

1. Reward history modulates clock speed
2. Different brain regions support this modulation differently
3. The cortico-striatal-thalamic circuit enables these computations

Dopamine / photometry questions are paused pending ephys-side resolution.

## Chapter 1 — Behavior

- **Trial-type clustering (open).** Bimodality in wait-time distributions is a confirmed observation, but the two-strategy interpretation is **not validated**. GMM classification is NOT reliable and should not be treated as established.
    - *Resolution path:* Hartigan's dip test, Silverman's bandwidth test, BIC-based mixture model comparison; bg_length correlation diagnostic on zero-restart trials.
- **Per-mouse Wilcoxon tests** for delta-T analysis — currently tested on pooled trials, inflating significance. Rerun per-mouse.
- **Behavioral strategy × neural activity (new).** If reactive vs timing trials can be identified behaviorally (not via GMM, which is unreliable), does neural activity differ on those trials? Particularly: is clock-speed modulation by reward history present only on "timing" trials? Requires a defensible behavioral strategy classifier first.

## Chapter 2 — Cross-region neural timing codes

### Rescaling confounds (active, this-week priority)

- **Edge-cell contamination.** Many units contributing to the rescaling r appear to fire near the interval edges (cue-locked transients or pre-motor / lick activity). These can inflate r without reflecting genuine tile-the-interval timing.
    - *Resolution path, in order:*
        1. τ-stratified reporting (early/mid/late peak bins, r per stratum, fraction per region). Fast — uses existing caches.
        2. Crop-and-resort: crop the first/last X% of each unit's PSTH, re-find the peak in the cropped portion, re-run rescaling. Units that fire at both edges and the middle keep their middle peak; pure-edge units get exposed. Requires regenerate of caches.
        3. Full-PSTH similarity (Procrustes distance + zero-lag normalized cross-correlation) between Q3 and Q4 PSTHs, run as a 2×2 of alignment × cropping. **Alignment:** (a) anchor-aligned — tests whether the whole code from the anchor rescales (population-level claim, but confounded by edge cells until cropped); (b) peak-aligned in normalized time — removes peak position to isolate response shape/width, complements the τ-correlation rather than replacing it (peak-alignment is not itself a rescaling test). The informative contrast for peak-aligned is normalized-time vs absolute-time profile match. **Cropping:** before and after edge cropping. The anchor-aligned-cropped panel is the key population test. Report each as a per-unit distribution per region.
- **Best-quartile-pair reporting.** Q3→Q4 is not always the strongest contrast; some sessions show better Q2→Q3. Best-(anchor × quartile-pair) sweep already exists in `a_rescaling.py` — propagate into the cross-region table.

### Decoder bottleneck (active)

- **Single-session simultaneity requirement** is fatal for Long-BG MSN (n=1 session at MIN_UNITS=5, already lowered).
    - *Resolution path:*
        - Pseudo-population decoder: pool units across sessions (within group × region), trial-match on wait quartile or BG length, decode at population level. Loses per-trial clock speed but gains sample. Parallel track to simultaneous decoder.
        - Mixed-effects pooling of per-session estimates (current decoder, MIN_UNITS=5, random effect for session/mouse). Reframes single-session results as noisy observations of a population-level effect.

### Cell typology (new, prerequisite for regional-differentiation claim)

- **PSTH decomposition.** Cluster or factor-analyze unit PSTHs across all regions to identify canonical response types (ramping, mid-interval peak, onset transient, offset transient, tonic). Per-region composition of types becomes the regional differentiator. Until this exists, "regions do it differently" can't be defended.

### Clock-speed claim (active)

- **Reference frame of the clock-speed signal.** cue_off-null at all regions: the clock-speed history effect doesn't appear when the decoder is anchored to cue_off. Working interpretation: the clock-speed signal lives in the animal's behavioral reference frame (cue_on for Long BG, last_lick for Short BG); cue_off (which neither group times from internally) doesn't expose it. Not a falsifier of the reward effect — a constraint on its reference frame. Sanity check: inspect the cue_off-decoded trace to confirm it's not flat noise.
- **Within-trial precedence (new).** Does early-window clock speed (first 1–2 s post-anchor) predict the trial's full wait time? Stronger correlation argument than full-trial slope vs full-trial wait (which is tautologically related).
- **Wait-time confound within quartile.** The time-matched quartile control reduces but doesn't eliminate the wait–slope correlation within quartiles. Test: regress out current-trial wait from clock speed before applying the history test.

### GLM (in progress)

- Per-unit GLM with explicit confounds: firing rate ~ time-since-anchor + lick rate + reward(prev) + BG length + interactions. Provides the unconfounded "elapsed time encoding" measure that backstops the decoder.

### Probe reconstruction

- Still gating ephys interpretation. ~11/20 mice cleared. Any ephys finding that has not passed this check is provisional.

### Publishable figures

- Need cross-validated sort order (no circular diagonals).

## Chapter 3 — Cortico-striatal-thalamic circuit

- **Region focus decision (open).** Which regions form the empirical circuit claim? After cell typology + GLM, the region(s) with the strongest clock-speed-by-reward effect anchor the circuit story. Candidates: MC, MSN, VAL.
- **V1→DMS latency interpretation.** ~46.5 ms exceeds monosynaptic expectation. Polysynaptic / network-level coupling is the working interpretation. Re-examine after the MC→DMS positive control.
- **MC→DMS CCG (planned).** Anatomical positive control for the CCG method. Should show short / monosynaptic latency.
- **Thalamic involvement (planned).** VAL has dense relay projections to MC and DMS. VAL→MC and VAL→DMS CCGs would complete the loop signature.
- **Granger / communication subspace (planned).** On simultaneous V1↔DMS sessions, condition shared subspace activity on clock speed. Tests circuit-functional linkage.

## Chapter 4 — Dopamine modulation (paused)

- All photometry analysis phases on hold pending ephys-side resolution.
- DA / decoder direction tension (Short BG DA ramp shallows after reward; decoder clock speed faster after reward) remains the unresolved theoretical issue. Cross-regression on shared trials planned once ephys story stabilizes.
- rCaMP-in-striatal-fiber is a uniquely novel opportunity — analysis plan needed when photometry resumes.

## Meta / paper-writing

- Scope decision: working assumption: paper = behavior + Chapter 2 (cross-region timing codes + cell typology) + Chapter 3 (V1→DMS plus one additional CCG, e.g. MC→DMS). DA chapter as follow-up.
- Final figure list for the paper.
- Committee-ready vs publication-ready triage (use Committee-ready flag in Findings).