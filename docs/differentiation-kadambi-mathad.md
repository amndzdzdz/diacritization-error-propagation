# Differentiating from Mathad (2021) and Kadambi (2024)

**Status:** draft, written per plan §7 item 1 while the reading is fresh.
Destined for the introduction and related work; the rebuttal can quote it
directly if a reviewer raises the objection below. Source material: plan
[§1](mdd-paper-project-plan-v2.md#1-the-claim) and
[§5](mdd-paper-project-plan-v2.md#5-getting-it-accepted), and
[novelty-check.md §4](novelty-check.md), whose second subsection is the
working draft this expands from.

## The precedent

Mathad et al. (Interspeech 2021) and Kadambi et al. (Interspeech 2024)
already ask the structurally identical question about a different upstream
component: an automatic front-end process feeds a downstream pronunciation
assessment score, and both papers attempt to separate the front-end's
contamination from the learner's true acoustic deviation. Kadambi's own
words state the obligation directly — "care must be taken to distinguish
between the impact of alignment error (a spurious signal) and true acoustic
deviation on the automated score" — and that sentence is this project's
thesis with "alignment error" replaced by "diacritization error." Both
papers should be cited early and described accurately: this establishes the
question type as legitimate at this venue rather than inventing it, and a
reviewer who finds them uncited first will conclude we did not read the
field.

Both papers also report an adverse prior worth stating honestly rather than
burying: the upstream component's downstream impact is minor for typically
developing children and moderate (in ΔPLLR) for a clinical group — not
dominant. A reviewer who has read Kadambi will arrive expecting a small
effect, and a large one will read as suspicious unless the difference in
mechanism is argued up front, which is what the rest of this note does.

## Why alignment error and diacritization error are not the same failure mode

The reference in a pronunciation-assessment pipeline plays two roles at
once: it can be the model's input (in text-dependent systems) and it is
always the scoring key — whether a phoneme counts as mispronounced is
decided by comparing the learner's production against it. Forced alignment
and automatic diacritization corrupt that reference through categorically
different mechanisms, not merely different pipeline stages.

**Alignment error moves the measurement window.** The reference phoneme
sequence stays correct; a misaligned boundary only changes which stretch of
audio is read against which phoneme. The score can degrade because the
acoustics are pulled from the wrong interval, but a phoneme the learner
produced correctly cannot, by construction, be relabelled as an error —
the target itself never moved.

**A wrong diacritic moves the target label.** Corrupting the canonical
sequence changes what the "correct" phoneme was asserted to be at that
position. A learner who produced the true phoneme correctly is now scored
against a different, incorrect target and can be flagged as having made an
error they did not make. The ground truth shifts, not just the window used
to read it. This is why label-path corruption reaches every architecture,
including prompt-free systems that never consume the reference as model
input: it corrupts the key, not the input, and no architecture is scored
without a key.

Text-dependent systems add a second channel unavailable to the alignment
literature at all: the reference is a *detachable* text object, so it is
possible to hand a model one version of it while scoring against another.
An alignment cannot be split this way — there is no experiment where the
model sees one alignment and the scorer uses a different one, because
alignment is not an input a model consumes independently of the audio it
is aligned to. That detachability is what lets this work separate label-path
corruption from input-path corruption into two independently measurable
quantities (the paper's 2×2), which is precisely the decomposition Kadambi's
combined ΔPLLR effect cannot express. The decoupling is the contribution,
and it has no analogue in the forced-alignment literature by construction of
what alignment is.

## What this means for the adverse prior

Mathad and Kadambi's "minor to moderate" finding is evidence about
window-shifting error; it does not obviously transfer to label-shifting
error, and the paper should not assume it does either way. The empirical
test is the case-ending result: case endings are diacritics that are
grammatically, not perceptually, determined, so they are exactly the
condition where the *label* can move while the *pronunciation target* does
not. If the case-ending class produces a large, DER-disproportionate
downstream effect, that is direct evidence the two failure modes behave
differently and the alignment-literature prior does not transfer. If it
does not, the paper still has the protocol, the decomposition, and the
miscalibrated-DER result, which is why the pre-committed effect-size
position (`docs/effect-size-position.md`) treats a small effect as a
reportable outcome rather than a setback.

## One-paragraph version, for the rebuttal

> Mathad (2021) and Kadambi (2024) measure how forced-alignment error
> contaminates automatic pronunciation scoring, and we adopt their framing
> of separating spurious upstream signal from true acoustic deviation. The
> difference is categorical, not incremental: alignment error moves the
> measurement window while the reference phoneme sequence stays correct, so
> a correctly pronounced phoneme cannot be mislabelled as an error by
> construction. A corrupted diacritic moves the target label itself, so it
> can. Because the corrupted object is a detachable text reference rather
> than an alignment, we can additionally give a model one version of it and
> the scorer another — an experiment with no alignment analogue — which is
> what lets this work decompose reference corruption into separately
> measurable label-path and input-path effects rather than reporting the
> combined quantity Kadambi's design measures.
