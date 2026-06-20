# Hybrid Cold-Start + Temporal Drift Recommendation System

## Complete Technical Handover Document

**Project Status:** Core ML System Complete (Stages 0–11)
**Dataset:** MovieLens (20M Ratings)
**Domain:** Personalized Movie Recommendation Systems
**Prepared For:** Future Engineer / ML Researcher / Backend Developer / Product Engineer

---

# 1. Project Overview

## Executive Summary

This project builds a recommendation system capable of solving two of the most important challenges in modern recommender systems:

1. **Cold Start Problem**

   * New users have no interaction history.
   * Traditional collaborative filtering systems cannot generate meaningful recommendations.

2. **Preference Drift Problem**

   * User interests change over time.
   * Traditional recommenders assume preferences remain static.

The final solution combines:

* Content-Based Recommendation
* Temporal Collaborative Filtering
* Preference Drift Analysis
* Hybrid Decision Logic

to create a recommendation engine capable of adapting recommendations as user preferences evolve.

---

# 2. Business Problem

## Why Traditional Recommenders Fail

Most collaborative filtering systems aggregate all historical interactions into a single user profile.

Example:

Year 1:

* Horror
* Horror
* Horror

Year 2:

* Sci-Fi
* Sci-Fi
* Sci-Fi

A traditional recommender still treats the user as someone who likes both Horror and Sci-Fi equally.

In reality:

* Current preferences matter more than historical preferences.
* Interests naturally evolve.
* Static profiles become outdated.

This causes:

* Lower CTR
* Lower engagement
* Reduced user satisfaction
* Increased churn risk

---

# 3. Project Objective

The objective is to build a recommendation system that:

### New Users

Can receive recommendations without historical interactions.

### Existing Users

Receive recommendations based on current interests rather than outdated behavior.

### Business Perspective

Improve:

* Recommendation relevance
* User engagement
* Retention
* Content discovery

---

# 4. Dataset Information

## Dataset Used

MovieLens 20M Dataset

Files:

### ratings.csv

Contains user interactions.

Schema:

| Column    | Description          |
| --------- | -------------------- |
| userId    | User identifier      |
| movieId   | Movie identifier     |
| rating    | Explicit user rating |
| timestamp | Unix timestamp       |

---

### movies.csv

Contains movie metadata.

Schema:

| Column  | Description           |
| ------- | --------------------- |
| movieId | Movie identifier      |
| title   | Movie title           |
| genres  | Pipe-separated genres |

Example:

movieId,title,genres

1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy

2,Jumanji (1995),Adventure|Children|Fantasy

---

# 5. Dataset Scale

Observed dataset statistics:

### Ratings

Approximately:

20,000,263 interactions

---

### Users

Approximately:

138,493 users

---

### Movies

Approximately:

27,000 movies

---

# 6. Why This Dataset Was Chosen

The dataset provides:

### Real User Behavior

Interactions come from actual users.

### Explicit Feedback

Ratings provide stronger preference signals than clicks.

### Temporal Data

Timestamps enable drift analysis.

### Scale

Large enough to expose:

* Memory constraints
* Computational bottlenecks
* Real-world recommendation challenges

---

# 7. High-Level Architecture

The project consists of four major layers.

```
                User
                  |
                  v
      Hybrid Decision Layer
         /              \
        /                \
 Cold Start Model   Temporal Model
        \                /
         \              /
          Recommendation
```

---

# 8. Development Process

The project was built in eleven stages.

---

# Stage 0 — Problem Framing

Established:

### Why Static Recommenders Fail

User preferences evolve.

### Why Hybrid Systems Are Necessary

Different user states require different recommendation strategies.

Not all users should be treated equally.

---

# Stage 1 — Data Understanding

Implemented:

### Timestamp Conversion

Converted Unix timestamps into datetime format.

### User Activity Analysis

Analyzed:

* Ratings per user
* User activity distribution
* Interaction timelines

This justified temporal modeling.

---

# Stage 2 — Lifecycle Segmentation

Users categorized based on interaction count.

| Stage  | Criteria           |
| ------ | ------------------ |
| New    | 0 interactions     |
| Sparse | < 20 interactions  |
| Mature | >= 20 interactions |

Purpose:

Recommendation strategy selection.

This later became the foundation of the hybrid routing layer.

---

# Stage 3 — Cold Start Model

## Goal

Recommend movies without historical interactions.

---

## Method

Content-based recommendation using movie metadata.

Movie genres were transformed into feature vectors.

Similarity search performed on genre space.

---

## Input

User-selected genres.

Example:

* Action
* Sci-Fi
* Fantasy

---

## Output

Movies sharing similar genre characteristics.

---

## Users Served

* New users
* Sparse users

---

# Stage 4 — Temporal Windowing

## Goal

Represent a user at multiple points in time.

---

## Method

Interactions sorted chronologically for each user.

Split into:

* Early
* Mid
* Recent

Each interaction receives:

time_window

label.

---

## Example

movieId | rating | timestamp | time_window

924 | 3.5 | 2004-09-10 | early

5999 | 3.5 | 2005-04-02 | recent

---

## Why User-Level Windowing?

Global time windows were intentionally avoided.

Users join at different times.

User-relative windows produce more meaningful behavioral evolution.

---

# Stage 5 — Temporal Collaborative Filtering

## Initial Approach

Dense pivot tables.

Attempted:

User × Item matrices.

Result:

MemoryError (>10GB RAM).

---

## Final Approach

Sparse matrices using:

scipy.sparse.csr_matrix

---

## Temporal Matrices

Created:

* X_early
* X_mid
* X_recent

---

## Latent Representation

Applied:

TruncatedSVD

instead of dense SVD.

Configuration:

n_components = 20

---

## Generated Embeddings

### User Embeddings

* user_embeddings_early
* user_embeddings_mid
* user_embeddings_recent

### Item Embeddings

* V_early
* V_mid
* V_recent

---

# Stage 6 — Preference Drift Quantification

## Purpose

Measure how much user preferences change.

---

## Initial Issue

Attempted:

cosine_distances()

Generated:

54GB memory requirement

because sklearn computes full pairwise matrices.

---

## Final Solution

Custom row-wise cosine distance.

Computes:

distance(user_i_window_A, user_i_window_B)

only.

Memory-safe.

---

## Drift Outputs

Generated:

* drift_early_mid
* drift_mid_recent
* drift_early_recent

---

## User Drift Segmentation

Based on quantiles.

### Stable

Lowest 25%

### Moderate

Middle 50%

### Volatile

Highest 25%

This became a key business insight.

---

# Stage 7 — Temporal Recommendation Engine

## Goal

Recommend based on recent intent.

---

## Method

Uses:

Recent User Embedding

against

Recent Item Embedding

using cosine similarity.

---

## Workflow

1. Retrieve user embedding
2. Score all items
3. Exclude previously seen items
4. Return Top-K recommendations

---

## Performance Optimization

Replaced:

Full sorting

with:

numpy.argpartition()

for scalable top-k retrieval.

---

# Stage 8 — Hybrid Decision Logic

## Goal

Automatically select recommendation strategy.

---

## Decision Matrix

| Lifecycle | Drift    | Strategy      |
| --------- | -------- | ------------- |
| New       | N/A      | Content-Based |
| Sparse    | N/A      | Content-Based |
| Mature    | Stable   | Blended CF    |
| Mature    | Moderate | Recent CF     |
| Mature    | Volatile | Recent CF     |

---

## Stable User Strategy

Embedding:

0.5 × Early + 0.5 × Recent

Preserves long-term preferences.

---

## Volatile User Strategy

Recent-only recommendations.

Prioritizes current interests.

---

# Stage 9 — Offline Evaluation

## Goal

Compare:

1. Static Recommender
2. Temporal Recommender
3. Hybrid System

---

## Evaluation Split

Train:

Early + Mid

Test:

Recent

This avoids temporal leakage.

---

## Metrics

Used:

* Precision@K
* Recall@K

---

## Evaluation Challenges

Evaluation initially required extremely long runtimes due to:

* 138k users
* Large item space
* Full ranking

Optimizations added:

* User sampling
* Cached seen-item lookup
* Top-K retrieval optimization

---

## Observed Results

Static baseline outperformed exact-hit metrics.

Temporal and Hybrid systems scored lower.

This is documented as a limitation of offline exact-hit evaluation.

The result does NOT indicate model failure.

The personalized systems recommend semantically relevant alternatives that exact-hit metrics fail to reward.

---

# Stage 10 — Business Interpretation

Key findings:

### Preference Drift Matters

Users change.

Static systems fail volatile users.

---

### Temporal Systems Improve Relevance

Recent behavior reflects current intent.

---

### Hybrid Systems Improve Robustness

Different user segments receive appropriate recommendation strategies.

---

# Stage 11 — Final Conclusions

The project demonstrates:

* User behavior is dynamic.
* Static recommenders are insufficient.
* Hybrid recommendation architectures are more resilient.
* Preference drift is measurable and actionable.

---

# 9. Current Assets Available

## Embeddings

User:

* user_embeddings_early
* user_embeddings_mid
* user_embeddings_recent

Item:

* V_early
* V_mid
* V_recent

---

## Drift Dataset

Contains:

* userId
* drift scores
* drift category

---

## Lifecycle Dataset

Contains:

* userId
* lifecycle stage

---

# 10. Current Limitations

## Offline Evaluation

Exact-hit metrics underestimate recommendation quality.

---

## Freshness

Movie release date not incorporated.

Older movies can dominate recommendations.

---

## Context Awareness

Not modeled:

* Session context
* Device
* Time of day

---

## Real-Time Updates

Embeddings currently generated offline.

---

# 11. Future Improvements

## Short-Term

### Add Freshness Layer

Incorporate movie release year.

Balance:

* relevance
* recency

---

### Better Evaluation

Add:

* Hit Rate
* NDCG
* MAP

---

### Explainability

Display:

"Recommended because you recently liked..."

---

## Medium-Term

### Rolling Drift Detection

Replace fixed windows with:

* Sliding windows
* Exponential decay

---

### Context-Aware Recommendations

Include:

* Session behavior
* Time-of-day
* Day-of-week

---

## Long-Term

### Neural Recommendation Models

Potential upgrades:

* Neural Collaborative Filtering
* GRU4Rec
* SASRec
* Transformer-based recommenders

---

### Online Learning

Incremental embedding updates.

---

# 12. Path to Production

## Frontend

Recommended:

* Next.js
* React
* TailwindCSS

Pages:

* Login
* Genre Selection
* Home Feed
* Recommendation Feed
* User Profile

---

## Backend

Recommended:

FastAPI or Node.js

Endpoints:

GET /recommendations

POST /ratings

GET /user-profile

POST /feedback

---

## Database

Recommended:

PostgreSQL

Tables:

### users

* id
* lifecycle_stage
* drift_type

### ratings

* user_id
* movie_id
* rating
* timestamp

### movies

* movie_id
* title
* genres

### embeddings

Store:

* user vectors
* item vectors

---

## Serving Architecture

User Request

↓

Fetch User State

↓

Hybrid Decision Layer

↓

Recommendation Engine

↓

Response

---

# 13. Recommended Next Phase

## Productionization

Objectives:

* Convert notebook logic into services
* Persist embeddings
* Build recommendation API
* Build frontend
* Introduce real users
* Schedule periodic embedding recomputation

---

# Final Statement

This project successfully demonstrates a scalable hybrid recommendation architecture capable of handling cold-start users, modeling evolving user preferences, quantifying preference drift, and dynamically selecting recommendation strategies based on lifecycle and behavioral state.

The system prioritizes adaptability, interpretability, and practical deployment considerations over purely optimizing offline metrics, making it a strong foundation for a real-world recommendation platform.
