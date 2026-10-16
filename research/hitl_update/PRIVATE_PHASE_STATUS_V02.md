# Research status v0.2 — private hypothesis screening

**2026-10-16. Public summary only; candidate-specific implementation intentionally not published.**

The frozen delivery branches remain separate from the supervised-update research branch.

A separate local, private exploratory experiment now tests a simulated version of the stated failure mode: an operator-confirmed **normal operating-mode change** may resemble the telemetry signatures of historically labeled equipment faults. Baselines compare no updating, standard supervised retraining and a task-directed candidate mechanism.

**Verified high-level findings:**

- The unchanged detector fails under the deliberately severe simulated operating-mode shift.
- Ordinary supervised retraining with labeled examples resolves most of the simulated shift; therefore the research must compare against this strong baseline.
- A first candidate did **not** produce stable, repeatable evidence of an improvement over ordinary supervised retraining. It must not be presented as a successful patented algorithm.
- Source annotations distinguish permitted, held, denied, and temporally invalid examples. A label file alone does not authenticate the reviewer or prove that raw sensor files are available.
- Neither authentic new operator approvals nor physical three-state fault labels were provided for this experiment.
- No runtime model self-training, automatic deployment, or modification to the frozen delivery branch occurred.

**Next research gate:** evaluate cross-component probability compatibility when the anomaly detection model changes under trusted supervision. This requires independent state and anomaly labels as well as compatible temporal observation records. A later candidate must outperform ordinary supervised retraining on an identical, frozen labeled dataset without using its final audit split for tuning.

The experiment program, unpublished mathematical procedures, and patent-sensitive candidate source are held privately pending an intellectual-property filing review. They are intentionally **not** present in this public GitHub branch.
