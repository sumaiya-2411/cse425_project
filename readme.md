<h1><b>Supervised Neural Network Project: GNN-Based BERT for Understanding Context from Music </b></h1>

This repository contains the source code for our Neural Networks CSE425 coursework project.

Our team built a hybrid machine learning system combining BERT (for text) and Graph Neural Networks (for audio structure) to deeply understand musical context. Unlike standard generative models, this project focuses on understanding and prediction, including multi-label tagging, emotion regression, and cross-modal alignment between audio and text.

<h1><b>Project By </b></h1>

Sumaiya Tasnim Khan

<h1><b>Project Roadmap & Tasks </b></h1>

This project is divided into four progressive tasks:

**Task 1:** BERT Baseline for Music Tag Understanding
Implemented a BERT-based multi-label classifier on textual music context without graph structure.

**Task 2:** GNN on Music Structure Graphs
Built a GraphSAGE encoder on segment graphs using audio-only node features to predict genres and tags.

**Task 3:** GNN-BERT Fusion for Multi-Context Understanding
Fused structural (GNN) and semantic text (BERT) representations using cross-attention to predict multi-label context.

**Task 4:** Cross-Modal MusicCaps Alignment
Learned a shared embedding space between audio graphs and natural-language descriptions (MusicCaps) using a dual-encoder and InfoNCE contrastive loss.

<h1><b>Datasets Used </b></h1>

**FMA (small/medium):** Used for audio, genre (8-16 classes), and top tags.

**GTZAN:** Used for standard easy baselines and genre classification (10 genres, 1,000 tracks).

**MusicCaps:** 5,521 clips with expert natural-language captions provided by Google, used for advanced cross-modal retrieval.

<h1><b>Repository Structure </b></h1>

The repository is structured according to the project specifications:

gnn-bert-music-context/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/                  # FMA, GTZAN, MusicCaps downloads
│   ├── processed/            # .pt graphs, mel-spec, BERT caches
│   └── splits/               # train/val/test JSON
│
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb    # End-to-end inference example
│
├── src/
│   ├── audio_features.py     # mel, chroma, segmentation
│   ├── graph_builder.py      # chord + segment graphs
│   ├── bert_encoder.py
│   ├── gnn_model.py          # GraphSAGE
│   ├── fusion_model.py       # cross-attention GNN-BERT
│   ├── contrastive.py        # Task 4 InfoNCE
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── metrics.json
│   ├── plots/                # t-SNE, AUC-PR, F1 curves
│   └── retrieval_examples/
│
└── report/
    └── final_report.pdf
