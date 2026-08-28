# Models

- `lgbm_model.pkl` and `tfidf_vectorizer.pkl` at repo root are DVC-tracked pipeline outputs (ignored by git, versioned via `dvc.yaml` + `dvc.lock`).
- `best_model_*.h5` are deep-learning checkpoints from notebooks 12-13 (BERT / RNN variants, ~30MB each). They are **gitignored** (`*.h5` in `.gitignore`) to avoid GitHub 100MB limit.

To share them:
```bash
# Option 1: DVC
dvc add models/best_model_lstm.h5
dvc push

# Option 2: Git LFS
git lfs track "*.h5"
git add .gitattributes models/*.h5
```

Do not `git add` large `.h5` directly.
