# 🎬 Hybrid Cold-Start & Temporal Drift Aware Recommendation System

> A production-inspired recommendation system that adapts to both user lifecycle stages and evolving user preferences over time.

## TenFlix V4 — live rating-aware recommendation layer

V4 is the current product-facing implementation. It preserves V3 as an offline benchmark
and adds the missing web-application boundary:

* biased explicit matrix factorization with user/item biases;
* immediate regularized user fold-in after a rating;
* typed onboarding, organic, recommendation, imported, and legacy events;
* positive and negative content profiles using genres, year, and decade;
* evidence-based temporal eligibility and continuous recency weighting;
* collaborative, content, quality/popularity, recent, and exploration candidates;
* lifecycle-aware linear reranking, bounded freshness, and MMR diversity;
* repository protocols, an in-process service, and an optional FastAPI adapter;
* schema-4 artifacts and strict promotion gates.

### V4 commands

Install the editable project with development and optional HTTP dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,service]"
```

Place MovieLens `ratings.csv` and `movies.csv` in the repository root, then run:

```powershell
tenflix-v4 prepare-data --config configs/v4.yaml
tenflix-v4 tune --config configs/v4.yaml
tenflix-v4 train --config configs/v4.yaml --mode evaluation --run-id movielens-v4-eval
tenflix-v4 evaluate --run artifacts/movielens-v4-eval
tenflix-v4 recommend --run artifacts/movielens-v4-eval --user-id 1 --top-k 10
tenflix-v4 preview --run artifacts/movielens-v4-eval --ratings ratings-preview.json
tenflix-v4 promote --run artifacts/movielens-v4-eval
```

`tune` is the expensive full methodology path and is required before a serious promotion
attempt. It may be skipped only for a quick local smoke run; `train` then uses the
untuned defaults in `configs/v4.yaml`. When tuning output exists in
`data/processed-v4/v4-tuning.json`, training applies the selected MF parameters,
learned feature statistics, lifecycle weights, and MMR strength automatically.

`promote` refuses any run whose evaluation report does not satisfy every accuracy,
coverage, diversity, temporal, cold-start, and latency gate. `serve` additionally requires
a promoted production artifact.

### Full-stack web app upgrade

The repository now includes a product-facing web layer around the promoted V4 artifact:

* Supabase/Postgres repository adapters and schema migration;
* Supabase Auth JWT verification for authenticated API calls;
* catalog browsing, current ratings, movie detail, recommendation, and watch-action endpoints;
* TMDb enrichment support for posters, IMDb/TMDb IDs, watch providers, and Stremio actions;
* a Next.js App Router frontend under `apps/web` using the Refero brutalist/editorial design system.

See [`docs/web-app-upgrade.md`](docs/web-app-upgrade.md) for the local setup and runbook.

### Latest V4 full-dataset verification

The `movielens-v4-eval` artifact was trained from all 20,000,263 ratings using the
global chronological protocol and evaluated on all 4,898 users with both pre-cutoff
context and relevant, release-eligible test events. It is correctly marked
`validated: false`; no production artifact was promoted.

| System | NDCG@10 | Recall@10 | Catalog coverage |
| --- | ---: | ---: | ---: |
| Popularity-quality | 0.1048 | 0.0306 | 0.0114 |
| Static biased MF | 0.0376 | 0.0118 | 0.2108 |
| Recency-aware biased MF | 0.0511 | 0.0151 | 0.1875 |
| Full V4 | **0.1117** | **0.0354** | 0.1207 |

Full V4 beats popularity and static MF with positive 95% paired-bootstrap lower
bounds. Recency-aware MF also beats static MF for temporally eligible users, and the
genre-aware cold simulation passes. Genre diversity is 0.7814. Promotion is blocked
because coverage is below 0.15 and measured recommendation p95 was 379 ms, above the
250 ms gate. Fold-in plus recommendation p95 was 409 ms and passed its 500 ms gate.
The serving hot path was optimized after this report; rerun `evaluate` before using new
latency numbers for promotion.

The repository-wide logic audit subsequently corrected onboarding tokenization,
single-rating content evidence, item-bias fallback, cache identity, HTTP POST validation,
promotion enforcement, evaluation sampling/coverage semantics, and tuning normalization.
The saved run remains loadable, but its metrics describe the pre-audit artifact. Retrain
and reevaluate before treating the current source tree as benchmarked.

### Live service contract

The engine exposes `RatingRepository` and `CatalogRepository` protocols so the future web
backend can provide PostgreSQL adapters without coupling database code to the ML package.
`RecommendationService.record_rating()` invalidates the cached profile immediately;
`recommend()` folds the current rating history into fixed global item factors without a
global retrain.

The optional HTTP adapter exposes:

```text
POST /v1/ratings
GET  /v1/recommendations/{user_id}
POST /v1/recommendations/preview
GET  /v1/health
GET  /v1/model
```

## TenFlix v3 benchmark implementation

> Historical note: V3 remains the validated methodology baseline, but V4 is now the
> active implementation for future web-app integration.

The executable implementation now lives in `src/tenflix`. `TenFlix_v1.ipynb` and
`TenFlix_v2.ipynb` are retained as historical experiments and their saved temporal
metrics and drift values are **not valid regression targets**. They used independently
fitted temporal SVD spaces, trained recent embeddings on evaluation holdout interactions,
and passed recommendation DataFrames to metric functions expecting movie-ID sequences.

V3 permanently addresses those problems with:

* one shared latent item space and global ID mappings;
* a per-user chronological 30% old / 40% recent-context / 30% future-holdout split;
* training-context-only rating centering, popularity, quality, and drift thresholds;
* sparse query-time TF-IDF scoring without a dense movie-by-movie matrix;
* one lifecycle router and an ordered, typed recommendation result;
* versioned, hash-verified artifacts and segment-level offline evaluation.

### Lifecycle policy

| Stage | Observed interactions | V3 strategy |
| --- | ---: | --- |
| New | 0 | Selected-genre content model |
| Cold | 1–19 | Interaction-derived content model |
| Sparse | 20–49 | Content/collaborative blend |
| Mature | 50+ | Drift-aware temporal collaborative model |

### Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
tenflix prepare-data --config configs/v3.yaml
tenflix train --config configs/v3.yaml --mode evaluation --run-id movielens-v3-eval
tenflix evaluate --run artifacts/movielens-v3-eval
tenflix recommend --run artifacts/movielens-v3-eval --user-id 1 --top-k 10
tenflix recommend --run artifacts/movielens-v3-eval --genres Action Sci-Fi --top-k 10
```

For the Supabase/Postgres web app layer, `.[dev]` now includes the required web
runtime dependencies. If you only need the product API without test/lint tooling,
install `python -m pip install -e ".[web]"`.

`evaluate` exits with code 2 when the statistical acceptance gates are not met. A
completed run is only described as validated when hybrid NDCG@10 and Recall@10 beat
the popularity baseline for mature moderate/volatile users with a positive paired
bootstrap confidence bound, and catalog coverage also improves.

### Latest full-dataset verification

The local `movielens-v3-eval` run trained on all 20,000,263 source ratings and evaluated
a seeded sample of 2,000 eligible users. It correctly finished as `validated: false`:

| System | NDCG@10 | Recall@10 | Catalog coverage |
| --- | ---: | ---: | ---: |
| Popularity-quality | 0.0993 | 0.0564 | 0.0038 |
| Static CF | 0.0856 | 0.0438 | 0.2447 |
| Recent-context CF | 0.0774 | 0.0404 | 0.2740 |
| Hybrid | 0.0765 | 0.0360 | 0.2550 |

The hybrid substantially expands coverage, but neither moderate nor volatile users
showed a statistically positive ranking gain over popularity. A production run should
not be promoted until model iteration passes those gates.

---

## 📌 Overview

Traditional recommendation systems assume that user preferences remain static over time. In reality, users continuously evolve their interests, explore new genres, and shift their consumption patterns.

This project addresses that challenge by building a **Hybrid Recommendation System** that combines:

* Content-based recommendations for cold-start users
* Temporal collaborative filtering for mature users
* Preference drift analysis
* Lifecycle-aware recommendation strategies
* Hybrid decision logic

Rather than focusing solely on maximizing recommendation accuracy, this project emphasizes **adaptability, interpretability, and production-oriented system design**.

---

## 🎯 Problem Statement

Most recommender systems aggregate all historical user interactions and assume that user preferences remain unchanged.

This creates three major challenges:

### 1. Cold Start Problem

New users have little or no interaction history, making collaborative filtering ineffective.

### 2. Preference Drift

User interests evolve over time, causing older interactions to become less representative of current preferences.

### 3. One-Size-Fits-All Recommendations

Different users require different recommendation strategies. A static recommendation pipeline cannot effectively serve all user segments.

This project aims to solve all three challenges through a unified hybrid recommendation framework.

---

## 🧠 Key Concepts

* User Lifecycle Segmentation
* Cold-Start Recommendation
* Temporal Modeling
* Matrix Factorization
* Preference Drift Quantification
* Hybrid Recommendation Systems
* Offline Recommendation Evaluation
* Recommendation System Design

---

## 🏗️ System Architecture

```text
User Interactions
        │
        ▼
Lifecycle Segmentation
        │
        ▼
Temporal Windowing
(Early / Mid / Recent)
        │
        ▼
Collaborative Embeddings
(Matrix Factorization)
        │
        ▼
Preference Drift Analysis
        │
        ▼
Hybrid Decision Layer
        │
        ▼
Personalized Recommendations
```

---

## 📂 Dataset

### MovieLens Dataset

The project uses:

* `ratings.csv`
* `movies.csv`

### Dataset Characteristics

* Millions of user-movie interactions
* Explicit ratings
* Timestamps available
* Movie metadata (genres)

The availability of timestamps enables realistic temporal modeling and drift analysis.

---

# 🚀 Project Pipeline

---

## Stage 1 — Data Ingestion & Sanity Checks

### Tasks Performed

* Loaded ratings and movie metadata
* Converted Unix timestamps into datetime format
* Performed integrity checks
* Analyzed user activity distributions
* Examined interaction time spans

### Objective

Validate the dataset and justify the need for temporal recommendation strategies.

---

## Stage 2 — User Lifecycle Segmentation

Users were categorized into lifecycle stages:

| Category | Criteria          |
| -------- | ----------------- |
| New      | 0 interactions    |
| Sparse   | < 20 interactions |
| Mature   | ≥ 20 interactions |

### Why This Matters

Different user groups require different recommendation strategies.

This segmentation forms the foundation of the hybrid recommendation system.

---

## Stage 3 — Cold-Start Recommendation System

For new and sparse users:

### Approach

* Genre-based content representations
* Content similarity search
* Preference elicitation through genres

### Benefits

* No dependency on historical interactions
* Explainable recommendations
* Immediate usability for new users

---

## Stage 4 — Temporal Windowing

User histories were divided into:

* Early
* Mid
* Recent

### Purpose

* Prevent temporal leakage
* Enable realistic evaluation
* Capture evolving user preferences

### Outcome

Each user is represented by multiple temporal versions of themselves.

---

## Stage 5 — Time-Aware Collaborative Modeling

### Approach

For each temporal window:

1. Construct sparse user-item matrices
2. Apply matrix factorization using Truncated SVD
3. Learn latent user and item representations

### Result

Users are represented dynamically rather than through a single static profile.

---

## Stage 6 — Preference Drift Quantification

### Objective

Measure how much user preferences change over time.

### Method

Cosine distance between user embeddings across temporal windows:

* Early → Mid
* Mid → Recent
* Early → Recent

### User Categories

Users are classified as:

* Stable
* Moderate
* Volatile

### Key Insight

Not all users evolve equally.

Some users maintain consistent preferences while others undergo significant preference shifts.

---

## Stage 7 — Temporal Recommendation Engine

### Recommendation Strategy

Recommendations are generated using:

* Recent user embeddings
* Similarity in latent preference space
* Exclusion of previously interacted items

### Core Principle

Recent behavior provides stronger signals than historical averages.

---

## Stage 8 — Hybrid Decision Logic

The recommendation strategy is selected dynamically based on user characteristics.

| User Type         | Strategy                               |
| ----------------- | -------------------------------------- |
| New               | Content-Based                          |
| Sparse            | Content-Based                          |
| Mature + Stable   | Blended Historical + Recent            |
| Mature + Moderate | Recent-Focused Collaborative Filtering |
| Mature + Volatile | Recent-Only Collaborative Filtering    |

### Why This Matters

Different users require different recommendation strategies.

This decision layer introduces system-level intelligence into the recommendation pipeline.

---

## Stage 9 — Offline Evaluation

### Evaluation Setup

Train:

* Early + Mid interactions

Test:

* Recent interactions

### Metrics

* Precision@K
* Recall@K
* Hit Rate

### Important Observation

Offline recommendation metrics often underestimate personalized recommendation quality because they reward exact next-item prediction rather than semantic relevance.

Evaluation results were therefore interpreted directionally rather than absolutely.

---

## Stage 10 — Business Interpretation

### Business Problems Addressed

* User engagement decline
* Stale recommendations
* Cold-start friction
* Reduced recommendation relevance

### Expected Impact

* Improved recommendation relevance
* Better user engagement
* Higher retention
* Enhanced discovery experience
* Better onboarding for new users

---

## Stage 11 — Final Conclusions

### Key Findings

* User preferences are dynamic rather than static.
* Temporal modeling provides richer user representations.
* Preference drift varies significantly across users.
* Hybrid systems are more robust than single-strategy recommenders.
* Understanding user behavior is as important as optimizing recommendation metrics.

---

# 📊 Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Sparse Matrix Operations

* SciPy

### Machine Learning

* Scikit-Learn
* TruncatedSVD

### Visualization

* Matplotlib

---

# ⚠️ Limitations

Current limitations include:

* Offline metrics underestimate semantic relevance.
* Item freshness is not explicitly modeled.
* Temporal windows are coarse-grained.
* No contextual signals are incorporated.

Examples of contextual signals:

* Time of day
* Device type
* Location
* Session intent

---

# 🔮 Future Improvements

Potential future enhancements include:

### Model Improvements

* ALS Matrix Factorization
* Bias-aware Matrix Factorization
* Content-enriched item embeddings

### Recommendation Improvements

* Diversity-aware re-ranking
* Popularity-aware serving
* Freshness constraints

### Production Improvements

* Online A/B Testing
* Incremental model updates
* Continuous drift monitoring

---

# 🎯 Why This Project Stands Out

Unlike traditional recommendation system projects that focus solely on collaborative filtering accuracy, this project emphasizes:

* User lifecycle awareness
* Preference evolution
* Behavioral analysis
* Recommendation adaptability
* Production-oriented design

The focus is not just on recommending items, but on understanding how users change over time and adapting recommendations accordingly.

---

# 👨‍💻 Author

**Akshit Gangwar**

Computer Science & Engineering (Data Science)

Interests:

* Recommendation Systems
* Machine Learning
* Data Science
* System Design
* Full Stack Development

---

## Final Thought

This project demonstrates that effective recommendation systems require more than accurate models. They require an understanding of user behavior, temporal dynamics, and adaptive decision-making.

By combining lifecycle awareness, temporal modeling, preference drift analysis, and hybrid recommendation logic, this system moves beyond static recommendations toward a more realistic and user-centric recommendation framework.
019ee62e-7361-7323-a9fb-627833dc3fe0
