## Section A: Business Context

In the US alone, over $262 billion is lost annually in administrative overhead tied 
to claim denials. A wrong denial has real consequences — providers lose revenue 
when appeal windows close, and patients face delays in care while disputes are 
resolved. A model that predicts likely denials before submission gives medical 
billers and revenue cycle managers the ability to fix issues proactively, reducing 
appeals, protecting revenue, and getting patients the care they need faster.

## Section B: Problem Framing — Three AI Tasks

### Task 1: Claim Denial Prediction
The input is a set of claim features including ICD codes, CPT codes, provider type, billed amount, and patient age. The output is a probability of denial between 0.0 and 1.0. This is a binary classification problem — the model predicts whether a claim will be approved or denied.

We will consider this task successful when recall on the denied class reaches 0.80 or 
above. A missed denial (false negative) is more costly than a false alarm (false 
positive) — a biller who is warned about a likely denial can fix the claim before 
submission, but a missed denial results in lost revenue and appeal overhead.

### Task 2: Anomaly Detection
The input is a batch of claims. The output is an anomaly score per claim, with the 
highest-scoring claims flagged for human review. This is an unsupervised anomaly 
detection problem — there are no pre-labeled fraud cases to train on, so the model 
learns what "normal" looks like and flags deviations from it.

We will consider this task successful when at least 15 of the top 20 flagged claims 
(precision-at-k, k=20) appear statistically unusual upon manual review — for example, 
extreme billing amounts, duplicate submissions, or clinically inconsistent ICD-CPT 
pairings.

### Task 3: ICD Code Classification
The input is a plain-English diagnosis description such as "patient presents with chest 
tightness and shortness of breath." The output is the top 3 predicted ICD-10 codes 
with confidence scores. This is a multi-class text classification problem.

We will consider this task successful when top-3 accuracy reaches 0.75 or above. If 
the correct ICD code appears anywhere in the model's top 3 predictions, a human 
coder can select it — this mirrors how ICD assist tools work in real clinical settings.

## Section D: Constraints and Assumptions

- All data used in this project is publicly available and de-identified. The primary 
dataset is the CMS Medicare Part B Physician and Other Practitioners dataset, 
published openly by the Centers for Medicare & Medicaid Services.

- No real patient data will be handled under any circumstances. All patient-level 
fields are either excluded or synthesized. This project is not subject to HIPAA 
but acknowledges its importance in any real-world deployment.

- All models must be explainable. Black-box approaches such as deep neural networks 
are not appropriate for this use case — healthcare AI requires that predictions 
can be traced back to specific features and explained to a clinician or biller.

- The application must run on free-tier infrastructure. The Streamlit app will be 
deployed on Streamlit Community Cloud at no cost, meaning model complexity and 
file sizes must remain within those constraints.

- Because real claim-level denial outcomes are not publicly available, denial labels 
are synthetically generated using realistic probability rules based on known denial 
patterns. This is a known limitation of the project and is documented as such.