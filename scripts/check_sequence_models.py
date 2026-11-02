"""Check padding invariance, batch independence and finite backward gradients."""
import json
import torch
from compare_lstm_xlstm import ROOT, Classifier, seed_all


def main():
    cfg = json.loads((ROOT / 'configs/lstm_xlstm_comparison.json').read_text())
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for kind in cfg['models']:
        seed_all(42)
        model = Classifier(kind, 100, cfg).to(device).eval()
        short = torch.tensor([[3, 4, 2]], device=device)
        padded = torch.tensor([[3, 4, 2, 0, 0, 0]], device=device)
        batch = torch.tensor([[3, 4, 2, 0, 0, 0], [8, 9, 5, 6, 7, 2]], device=device)
        with torch.no_grad():
            expected = model(short)
            torch.testing.assert_close(model(padded), expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(model(batch)[:1], expected, atol=2e-5, rtol=2e-5)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(batch), torch.tensor([1, 2], device=device))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.embedding.weight.grad[0].count_nonzero() == 0
        print(f'{kind}: padding / batch invariance, finite gradients, zero PAD gradient passed on {device}')


if __name__ == '__main__':
    main()
