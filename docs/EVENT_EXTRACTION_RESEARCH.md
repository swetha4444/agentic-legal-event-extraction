# Event Extraction: Research Summary (Methods, Evaluation, No Ground Truth)

This doc summarizes **how event extraction is done** across domains (including legal), **how it is evaluated**, and **practical options when you have no ground truth** (e.g. you are building a dataset from scratch by scraping CourtListener).

---

## 1. How event extraction is done

### 1.1 Task definition

Event extraction (EE) usually has two subtasks:

- **Event detection (ED):** Identify event triggers (words/phrases that denote an event) and assign event types (e.g. *Marry*, *Sue*, *Arrest*, *File*).
- **Event argument extraction (EAE):** For each event, identify arguments (participants, time, place, etc.) and their **roles** (e.g. *Agent*, *Patient*, *Time*, *Place*).

Your project is “role-aware” legal event extraction: you care about **who** did what (plaintiff, defendant, judge, etc.), so argument roles are central.

### 1.2 Paradigms (how models do it)

- **Classification / tagging:** Sentence or span classification; sequence tagging (BIO/BIOES) for triggers and arguments. Common in older and non-generative work.
- **Structured prediction / span-based:** Predict (trigger span, event type) and (argument span, role) from the text. Evaluated with exact or relaxed span match (see below).
- **Generative:** LLMs generate event structures (e.g. JSON or natural language) given the text. Outputs are often semantically correct but **do not match gold spans exactly** → exact-match evaluation underestimates performance (see REGen, BEMEAE).
- **Distant supervision:** Align text with knowledge bases (e.g. Freebase, Wikipedia infoboxes) to create **automatic (noisy) labels**; train models on this. Used when you don’t have hand-annotated events (relevant for “no ground truth” setting).
- **Self-training / silver labels:** Use a teacher model (or heuristics) to label unlabeled text; train or refine a student model. Can bootstrap from a small seed set or from another modality (e.g. AMR).

**Legal domain:** “Events Matter: Extraction of Events from Court Decisions” (Filtz et al., JURIX 2020) compares **deep learning, CRF, and rule-based** methods on ECHR decisions; they build a **manually annotated ECHR corpus** as gold. So in legal, same paradigms (classification, CRF, rules) plus the option of generative/LLM-based extraction.

### 1.3 Standard benchmarks (other domains)

- **ACE 2005, ERE:** Classic English event extraction; trigger + argument spans and roles.
- **TAC-KBP:** Template-based; events and relations.
- **MAVEN, WikiEvents, CASIE, etc.:** Various domains and event ontologies.
- **TextEE (2023–2024):** Standardized benchmark over **16 datasets, 8 domains**, with fixed preprocessing and splits to make numbers comparable across papers.

Legal-specific:

- **ECHR event corpus** (Events Matter): Gold events from court decisions; event type, parties, time.
- **LexTime:** Temporal ordering of legal events (e.g. Federal Complaints); not full EE but related.
- **CaseSumm, CourtReasoner:** Summarization/reasoning; can inform how you use “facts” for downstream event/role tasks.

---

## 2. How event extraction is evaluated (and pitfalls)

### 2.1 Standard metrics (when you have gold)

- **Precision / Recall / F1** at **event level** (trigger + type) and **argument level** (argument span + role).
- **Matching rules:**  
  - **Strict (exact match):** Predicted span must equal gold span (and role).  
  - **Relaxed:** Overlap (e.g. token overlap, or “argument head” match).  
  - **Type match only:** Ignore span; only check event type and role correctness.

Classic issue: **argument head vs full span.** Different guidelines (ACE vs ERE) and papers use different conventions (e.g. “John” vs “John Smith”); preprocessing (tokenization, span normalization) is often **not** documented, so **scores across papers are not directly comparable** (Peng et al., “The Devil is in the Details”, Findings ACL 2023).

### 2.2 Three main evaluation pitfalls (Peng et al., 2023)

1. **Data preprocessing discrepancy:** Different tokenization, span handling, train/dev/test splits → same dataset yields different numbers; often not specified.
2. **Output space discrepancy:** Tagging vs generative models produce different output formats; mapping “generated argument” to “gold span” is ambiguous; exact match is unfair to generators.
3. **Missing pipeline evaluation:** Many papers only evaluate EAE given gold triggers (or only ED); **end-to-end** (ED → EAE on raw text) is what matters for real use and is often missing.

**Implication for you:** If you later add event extraction, document preprocessing and output format clearly and report **pipeline** (full EE) metrics, not only argument-only under gold triggers.

### 2.3 Exact match vs semantic match

- **Exact match (EM):** Predicted argument span must equal gold span.  
  - **Problem:** LLMs (and humans) often produce **semantically correct but lexically different** arguments (paraphrases, different granularity). EM then **underestimates** performance (e.g. REGen reports ~+24 F1 when moving from EM to relaxed/LLM-based matching).
- **Relaxed matching:** Overlap (e.g. overlap ratio, or head word); or normalize spans (e.g. to syntactic head).
- **LLM-based matching:** Use an LLM to judge whether the predicted argument “expresses the same meaning” as the gold (or as a reference). REGen (Findings EMNLP 2025) combines exact, relaxed, and LLM-based matching and gets ~87% alignment with human judgment.

So: **evaluation metric choice strongly affects reported performance**, especially for generative/LLM extractors.

---

## 3. Evaluation when you do NOT have ground truth (your case)

You are **scraping/building data from scratch** (e.g. CourtListener → facts). You do **not** have gold event/argument annotations. Options below.

### 3.1 Create a small gold set (recommended)

- Manually annotate **events + argument roles** for a **small** set of documents (e.g. 50–100 opinions or 100–200 events).
- Use this as **test set only** (don’t train on it if you want an unbiased eval).
- Report **precision, recall, F1** (with a defined matching rule: exact vs relaxed; document it).
- Optionally report **inter-annotator agreement** (IAA) on a subset to show that the task is well-defined; event IAA is often only ~0.78–0.87 even for experts (ambiguity of event boundaries and roles).

**Cost:** Non-trivial but one-time; gives a clear, interpretable metric.

### 3.2 LLM-as-judge (no human gold)

- Take your system’s event/argument output and ask an LLM:  
  - “Does this extraction contain only events that appear in the source?” (faithfulness)  
  - “Are argument roles (who did what) correct?”  
  - “Rate 1–5: completeness / correctness of events and roles.”
- Or **pairwise:** “Given the source text, which of extraction A vs B is better (events + roles)?” then compare systems (e.g. LLM vs rule-based).
- Studies report **high correlation with human judgment** (e.g. ~0.85) when the judge is calibrated; but the judge can be biased or noisy, so use fixed prompts and seeds.

**Pros:** No human annotation; scalable.  
**Cons:** Judge quality and bias; not a “true” gold metric.  
**Use:** For model selection and ablation when you can’t afford a large gold set; complement with a small human gold set if possible.

### 3.3 Downstream task as proxy (your docs already mention this)

- Use **extracted facts** (or **events + roles**) as input to a downstream task: e.g. timeline construction, “find similar cases,” or a simple QA task.
- Compare: **gold facts** (if you have a little) vs **LLM facts** vs **heuristic facts** → run the same event extractor on each → compare **downstream performance** (e.g. accuracy of temporal order, or retrieval relevance).
- **Pros:** Measures “does this matter in practice?”  
- **Cons:** Confounds fact quality with event-extraction quality; need a defined downstream metric.

See **docs/NON_LLM_BASELINES_AND_EVAL.md** (§2.6) for the same idea (facts → events → SRL/roles; compare event/role F1 when feeding different fact extracts).

### 3.4 Nugget-based or checklist (minimal “gold”)

- Define a **list of nuggets** (atomic propositions) per document, e.g. “Plaintiff filed in state court,” “Injury on June 2, 2023,” “Three claims: premises, vicarious, negligent hiring.”
- Either **human** write nuggets for a sample, or **LLM** generate candidate nuggets and human approve/filter.
- **Metric:** For each system output (events or facts), check **recall**: how many nuggets are “covered” (exact substring or NLI: “Does output entail this nugget?”). Optionally precision if you define “system nuggets” (e.g. by clause split or LLM).
- **Pros:** Lighter than full event/role annotation; focuses on content.  
**Cons:** Designing nuggets takes effort; precision side needs a convention for “system nuggets.”

### 3.5 Human evaluation on a sample (no formal gold)

- Take a **random sample** of system outputs (e.g. 50–100 cases).
- Ask annotators (or yourself):  
  - Over-inclusion: “Does this contain non-events / wrong roles?” (count violations)  
  - Under-inclusion: “From a fixed checklist per case, how many key events/roles are missing?”  
  - Factual errors: wrong date, wrong party, wrong role.
- Report **% of outputs with no over-inclusion**, **average # missing facts/events**, **% with ≥1 factual error** (as in NON_LLM_BASELINES_AND_EVAL.md §2.5).
- **Pros:** Direct, interpretable, no need for span-level gold.  
**Cons:** Labor-intensive; not a single F1 number.

### 3.6 Silver / automatic labels (for training and weak eval)

- **Distant supervision:** Align your text with a KB (e.g. if you had a legal KB of “case X → events E”); auto-generate labels; train and optionally use a **held-out human-annotated** set for eval.
- **Self-training:** Train an initial model on a small seed (or on data from another domain); run it on your scraped data to get “silver” events; use for training or as a **pseudo-gold** for tuning (with the caveat that silver is noisy).
- **Heuristics as silver:** e.g. “sentences with a verb from list V and a person entity → event candidate”; use for ablations or as a baseline. Not ground truth, but a concrete comparison point.

---

## 4. Practical recommendations for your project

| Situation | Recommendation |
|-----------|----------------|
| **No gold at all yet** | (1) Use **LLM-as-judge** (faithfulness, role correctness, or pairwise) to compare fact extractors and, later, event extractors. (2) Add **nugget recall** (or checklist) on a small sample if you can define nuggets. (3) Plan a **small gold set** (50–100 docs) for events+roles when you add EE. |
| **When you add event extraction** | (1) **Document** preprocessing and output format (trigger/argument schema). (2) Prefer **relaxed or LLM-based matching** in addition to exact match (see REGen, BEMEAE) so that LLM-based extractors are not unfairly penalized. (3) Report **pipeline** (end-to-end) metrics, not only argument-only. |
| **Legal-specific** | Reuse ideas from **Events Matter** (ECHR): event types + parties + time; compare rule-based vs CRF vs neural/LLM. If you can’t use ECHR, your CourtListener + small manual annotations can serve as a “in-house” gold. |
| **Reproducibility** | Use or adapt a **standardized framework** (e.g. OmniEvent, TextEE) for preprocessing and scoring if you adopt existing event schemas; otherwise, release your own preprocessing and matching script with the paper/code. |

---

## 5. Papers to look at (full list)

### Evaluation pitfalls and metrics

1. **The Devil is in the Details: On the Pitfalls of Event Extraction Evaluation**  
   Hao Peng, Xiaozhi Wang, Feng Yao, Kaisheng Zeng, Lei Hou, Juanzi Li, Zhiyuan Liu, Weixing Shen. *Findings of ACL*, 2023.  
   Preprocessing discrepancy, output-space mismatch, missing pipeline eval.  
   [ACL](https://aclanthology.org/2023.findings-acl.586/) · [PDF](https://aclanthology.org/2023.findings-acl.586.pdf) · Code: [OmniEvent](https://github.com/THU-KEG/OmniEvent)

2. **REGen: A Reliable Evaluation Framework for Generative Event Argument Extraction**  
   *Findings of EMNLP*, 2025.  
   Exact vs relaxed vs LLM-based matching; +23.93 F1 over exact match; 87.67% alignment with humans.  
   [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.649/)

3. **BEMEAE: Moving Beyond Exact Span Match for Event Argument Extraction**  
   *NAACL*, 2025.  
   [ACL](https://aclanthology.org/2025.naacl-long.295/)

4. **Evaluating Zero-Shot Event Structures: Recommendations for Automatic Content Extraction (ACE) Annotations**  
   *ACL (short)*, 2023.  
   Structural ambiguities (coreference, argument head, modality); 32% / 25% impact on FN.  
   [ACL](https://aclanthology.org/2023.acl-short.142/)

5. **A Comparison of the Events and Relations Across ACE, ERE, TAC-KBP, and FrameNet Annotation Standards**  
   *ACL Workshop*, 2014.  
   [ACL](https://aclanthology.org/W14-2907/)

### Benchmarks and standardized evaluation

6. **TextEE: A Standardized, Fair, and Reproducible Benchmark for Event Extraction**  
   *Findings of ACL*, 2024.  
   16 datasets, 8 domains; 14+ methods; 5 LLMs; standardized preprocessing and splits.  
   [ACL](https://aclanthology.org/2024.findings-acl.760/) · [Project](https://khhuang.me/TextEE) · [GitHub](https://github.com/ej0cl6/textee)

7. **GENEVA: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles**  
   *ACL (long)*, 2023. Newer, larger EAE dataset; event types and roles; still has some data-quality caveats.  
   [ACL](https://aclanthology.org/2023.acl-long.203/) · [PDF](https://aclanthology.org/2023.acl-long.203.pdf)

8. **Other recent EAE dataset** (ACL long 2024) — event types and roles; newer/more fine-grained than ACE; data cleaning quality varies.  
   [ACL](https://aclanthology.org/2024.acl-long.224/) · [PDF](https://aclanthology.org/2024.acl-long.224.pdf)

**Classic EE dataset (known issues):**  
- **ACE** (Automatic Content Extraction) — default evaluation dataset for ~2 decades. [LREC 2004](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf). Critique: [Cai & O’Connor, ACL 2023 short](https://aclanthology.org/2023.acl-short.142.pdf) (coreference, argument head, modality; can inflate false negatives).

### Legal domain event extraction

9. **Events Matter: Extraction of Events from Court Decisions**  
   Filtz, Navas-Loro, Santos, Polleres, Kirrane. *JURIX*, 2020.  
   ECHR corpus; deep learning vs CRF vs rule-based; event type, parties, time.  
   [IOS Press](https://ebooks.iospress.nl/doi/10.3233/FAIA200847) · [PDF (Penni)](https://penni.wu.ac.at/papers/JURIX2020%20Events%20Matter%20Extraction%20of%20Events%20from%20Court%20Decisions.pdf)

10. **LexTime: A Benchmark for Temporal Ordering of Legal Events**  
   *arXiv*, 2025.  
   U.S. Federal Complaints; 512 instances; event pairs and temporal relations.  
   [arXiv:2506.04041](https://arxiv.org/abs/2506.04041)

**Legal knowledge graphs:** Constructing KGs from legal text (e.g. Chinese legal texts; related work may help for legal + structured extraction).  
[MDPI Information 15(11):666](https://www.mdpi.com/2078-2489/15/11/666)

### Methods: generative, distant supervision, self-training

11. **Generative Approaches to Event Extraction: Survey and Outlook**  
    *Workshop on Future of Event Detection (FuturED)*, 2024.  
    [ACL](https://aclanthology.org/2024.futured-1.7/)

12. **Event Extraction Using Distant Supervision**  
    *ACL (short)*, 2014. (L14-1091)  
    [ACL](https://aclanthology.org/L14-1091/)

13. **Scale Up Event Extraction Learning via Automatic Training Data Generation**  
    *AAAI*, 2024.  
    Distant supervision; scale from thousands to hundreds of thousands.  
    [AAAI](https://aaai.org/papers/12030-scale-up-event-extraction-learning-via-automatic-training-data-generation)

14. **Learning from a Friend: Improving Event Extraction via Self-Training with Feedback from Abstract Meaning Representation**  
    *ACL*, 2023.  
    [ACL](https://aclanthology.org/) · [GitHub (PLUM)](https://github.com/PLUM-Lab/Event_Extraction_with_Self_Training)

15. **Semi-Supervised Event Extraction with Paraphrase Clusters**  
    *NAACL*, 2018.  
    Bootstrap from paraphrase clusters; +1.1–1.3 F1 on ACE05/TAC-KBP.  
    [ACL](https://aclanthology.org/N18-2058/)

**LLM-based EE approaches (general):**  
- [ACL 2022 long 466](https://aclanthology.org/2022.acl-long.466.pdf)  
- [CASE 2022](https://aclanthology.org/2022.case-1.5.pdf)  
- [ULTRA, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.487.pdf) (document-level EAE)

### Evaluation without ground truth / LLM-as-judge

16. **Benchmarking LLM-as-a-Judge Models for 5W1H Extraction Evaluation**  
    *MDPI Electronics*, 2025.  
    GPT, Claude, Gemini on 5W1H extraction; 90%+ alignment; faithfulness, completeness.  
    [MDPI](https://www.mdpi.com/2079-9292/15/3/659)

17. **Inter-annotator Agreement for ERE Annotation**  
    *ACL Workshop*, 2014.  
    Comparing annotators without gold standard.  
    [ACL](https://aclanthology.org/W14-2904/)

18. **Validation Methodology for Expert-Annotated Datasets: Event Annotation Case Study**  
    IAA ~0.78–0.87; ambiguity-aware validation.  
    [Core](https://core.ac.uk/download/pdf/200222278.pdf)

### In-repo reference

19. **NON_LLM_BASELINES_AND_EVAL.md** (this repo)  
    Facts extraction baselines; LLM-as-judge; nuggets; downstream task; sentence-level / ROUGE / BERTScore.

---

## 6. Summary

- **Methods:** Event extraction is done via tagging, span-based, or generative (LLM) paradigms; in legal, DL/CRF/rules (and now LLMs) are used; distant supervision and self-training help when you lack labels.
- **Evaluation:** Standard is P/R/F1 with exact or relaxed span match; exact match underestimates generative models; preprocessing and output space must be documented for comparability.
- **No ground truth:** You can still make progress with (1) a **small human gold set**, (2) **LLM-as-judge**, (3) **nugget/checklist recall**, (4) **human audit** (over/under-inclusion, errors), (5) **downstream task** (e.g. events → timeline or retrieval), and (6) **silver labels** (distant supervision or self-training) for training and weak eval. Combining (1) and (2) is a practical path: small gold for a reliable number, LLM-as-judge for scale and model comparison.
