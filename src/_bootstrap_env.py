# must run before `transformers` / HF tokenizers import
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")