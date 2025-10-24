# Mini GPT Decoder (Учебный проект)

Полноценный минимальный **GPT-decoder**: эмбединги, маскированное многоголовое внимание, FFN, residual+LayerNorm, стек блоков и LM-голова.

Основано на лекциях по Attention и FFN. Источники заданий: лекция про Attention и Multi-Head (HeadAttention/MHA) и лекция про Feed-Forward (FFN). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## Структура
- `models/embeddings.py` — TokenEmbeddings, PositionalEmbeddings
- `models/head_attention.py` — HeadAttention (causal self-attention)
- `models/multihead_attention.py` — MultiHeadAttention
- `models/feed_forward.py` — FFN
- `models/decoder_block.py` — DecoderBlock (Pre-LN + residual)
- `models/model.py` — GPTLanguageModel (стек блоков + LM-голова)
- `examples/train_and_infer.py` — быстрый запуск
- `tests/test_shapes.py` — smoke-тесты форм

## Установка
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
# Если нужно CPU-билд:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
