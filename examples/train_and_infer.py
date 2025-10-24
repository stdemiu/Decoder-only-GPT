import torch
import torch.nn as nn
import torch.optim as optim

from models.model import GPTLanguageModel


def main():

    vocab_size = 100
    max_seq_len = 32
    emb_size = 128
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = GPTLanguageModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    B, S = 4, 16
    x = torch.randint(0, vocab_size, (B, S), device=device)
    y = torch.randint(0, vocab_size, (B, S), device=device)

  
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(10):
        logits = model(x)                   
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 2 == 0:
            print(f"[step {step+1}/10] loss={loss.item():.4f}")


    model.eval()
    start = torch.randint(0, vocab_size, (1, 4), device=device) 
    out_ids = model.generate(start, max_new_tokens=5)
    print("Generated ids:", out_ids.tolist())


if __name__ == "__main__":
    main()
