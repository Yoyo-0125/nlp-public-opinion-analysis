# Gemini CLI Instructions (GEMINI.md)

## Project Overview
This repository is an **NLP-based Public Opinion Monitoring (舆情监测) project**.

The goal of the project is to:
- Periodically collect social media / online text data
- Perform NLP processing such as cleaning, embedding, sentiment analysis, topic detection, and keyword extraction
- Track **hot events, emerging topics, and sentiment trends over time**
- Provide results that can later be visualized on a personal website (e.g. GitHub Pages)

The project is **research / learning oriented**, not a production-scale commercial system.

---

## Core Tech Stack

- **Language**: Python (primary)
- **NLP / ML**:
  - Transformers (HuggingFace)
  - Chinese pretrained models (e.g. RoBERTa / BERT variants)
  - Sentence Embedding models (e.g. text2vec)
  - Clustering models (e.g. HDBSCAN)
- **Training**:
  - PyTorch
  - Fine-tuning pretrained models for sentiment / topic tasks
- **Data**:
  - Text scraped from social platforms or public web pages
  - Stored locally or as simple structured files (CSV / JSON)

---

## Typical Workflow

1. **Data Collection**
   - Run a Python crawler / scraper
   - Collect text data within a specific time range
   - Deduplicate and clean raw text

2. **Preprocessing**
   - Text normalization
   - Tokenization
   - Filtering noise and irrelevant content

3. **Model Inference / Training**
   - Sentence embedding generation
   - Sentiment classification (binary or multi-class)
   - Topic detection / clustering
   - Keyword extraction

4. **Daily Update**
   - The full pipeline is expected to run **about once per day**
   - Results are saved for later visualization

---

## How Gemini CLI Should Help

Gemini CLI is mainly used as a **coding and reasoning assistant**, not as an autonomous agent.

Preferred uses:
- Help write or refactor **Python code**
- Explain ML / NLP concepts when implementing models
- Assist with:
  - Loss function selection
  - Model architecture design
  - Training and evaluation logic
- Debugging errors in PyTorch / Transformers code
- Helping design clean project structure

Non-goals:
- Do NOT generate fake data
- Do NOT invent experimental results
- Do NOT assume access to paid APIs or cloud services

---

## Constraints & Assumptions

- Hardware is limited (local laptop + occasional cloud GPU)
- Models should be **reasonably efficient**, not extremely large by default
- Code clarity and learning value are prioritized over extreme optimization
- The project evolves iteratively; incomplete modules are acceptable

---

## Coding Style Preferences

- Prefer **clear, explicit Python code** over overly clever abstractions
- Add comments for non-trivial logic
- Avoid unnecessary frameworks
- When suggesting code:
  - Prefer HuggingFace + PyTorch standard patterns
  - Keep functions modular and reusable

---

## Output Expectations

When helping in this repository, Gemini should:
- Be precise and technical
- Explain *why* a method is chosen, not just *what*
- Clearly separate assumptions from facts
- Prefer practical, implementable suggestions

---

## Target Audience

- The repository owner (a student learning NLP & ML)
- Future reviewers (teachers, interviewers, or collaborators)

The code and explanations should be understandable to someone with:
- Basic Python knowledge
- Intro-level ML / DL background

---

## Notes

This project is expected to grow gradually.
Design decisions should leave room for:
- Model replacement
- Data source expansion
- Visualization integration

Focus on **correctness, learning value, and maintainability**.

