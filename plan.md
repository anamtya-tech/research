Experiments To Do
Data Prep
Collect N background samples (Makesh have already collected - these are our negative samples). Split these into short segments
Collect M positive samples of classes we are interested in - Elephant, Langur etc. 
Listen to some data and ensure its right - noise is noise and positive samples sound correct. Can do some SNR / RMS analysis to identify plausible candidates to listen to.

Must use our simulator to simulate environments and pick sounds from the places we added GT sounds - do with or without background/ambient sounds

Goal: Understand exactly what data you have before touching a model

Tasks:
Create 50 positive clips per target class
After smart splitting, appropriate label based on what we created - if an elephant and lion were together then you can label the probability based upon the distance from centre - a nearer sound gets higher prob than further and all probs should sum to 1.
Setup 200 background/negative clips from Makesh’s sampling
Listen to several positive clip manually
Is the target sound actually present?
How much of the clip is target vs background?
What is the recording quality / SNR?

 Build a simple spreadsheet:
    clip_id | class | duration | SNR_estimate | quality(1-3) | notes

Deliverable: Annotated clip inventory
Pass criteria: You can describe your data distribution confidently


Yamnet Baseline Scores
Goal: See what raw YAMNet already thinks about your sounds

Tasks:
Run YAMNet on all positive clips
Look at top-5 AudioSet classes predicted for each
Record: does YAMNet's top prediction make ANY sense? e.g. Elephant clip → does it predict "Animal" or "Rumble"? Plot histogram of scores for your target vs random clips

Deliverable: YAMNet baseline score report
Pass criteria: You understand what YAMNet already knows
              about your target classes (may be very little)
Key insight: If YAMNet already scores elephant clips high on
             "Animal/Elephant" class — your job is easier.
             If it scores them high on "Music" — you know the embedding space needs work.


Embedding Visualization
Goal: Visually confirm that target sounds cluster in YAMNet space

Tasks:
Extract 1024-d embeddings for all clips (mean pooled)
Run UMAP or t-SNE to reduce to 2D
Plot: color by class (elephant, langur, drone, background)
Examine: do target classes form distinct clusters? or are they scattered through background?

Deliverable: UMAP plot, written observations
Pass criteria: At least partial separation visible
If clusters heavily overlap with background -> Level 3 fine-tuning will be needed (flag this early)

Simple Classifier: Linear Probe
Goal: What's the ceiling for a frozen backbone with zero complexity?

Model:
  YAMNet (frozen) → mean pool → Logistic Regression

Why logistic regression first:
  - No hyperparameter tuning needed
  - Tells you directly how linearly separable embeddings are
  - Takes 2 minutes to train

Tasks:
One binary classifier per target class (vs background)
80/20 train/test split, stratified
Report: Accuracy, Precision, Recall, F1, AUC-ROC
Plot ROC curve — save it, every future experiment gets overlaid on this same plot

Deliverable: Baseline numbers
Pass criteria: AUC > 0.70 (if not, embedding space is wrong)

Small MLP Head
Goal: Does non-linearity help over logistic regression?
MLP = Multi Label Predictor

Model:
  YAMNet (frozen) → mean pool → MLP(256→128→1)

Tasks:
Train with Adam, lr=1e-3, 50 epochs. Try SGD as well.
Use early stopping on val loss (patience=10)
Plot train vs val loss curves — look for overfitting
Compare AUC to Experiment 1.1
Loss function depends upon number of classes

Elephant-only head:  sigmoid + binary CE
Wildlife multiclass where only one animal expected: softmax + categorical CE
Wildlife where co-occurrence is possible: sigmoid + binary CE


Deliverable: MLP vs logistic regression comparison
Pass criteria: AUC improves by >2% over 1.1
If not: the gain isn't worth the complexity, the problem is the features not the head

