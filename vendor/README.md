# xLSTM source provenance

Source: https://github.com/NX-AI/xlstm/archive/ab22eadbd245f293dd8dec38ed29963d73758a12.zip

The downloaded ZIP is retained as `xlstm-ab22ead.zip`. The source license is
retained in the extracted directory. This experiment uses the NeurIPS 2024
`xLSTMBlockStack`, with mLSTM blocks only, using the native PyTorch parallel backend.

One local compatibility patch in `xlstm/blocks/slstm/cell.py` changes the
module-level `from .src.cuda_init import load` into a lazy wrapper importing
the same function on first use. The upstream module otherwise probes CUDA
Toolkit include paths at import time even when only mLSTM is used. This machine
has CUDA-enabled PyTorch but no standalone CUDA Toolkit/C++ compiler.
No mLSTM computation, parameters, initialization, or gradient code is modified.
The experiment environment metadata records SHA-256 of every Python source file.
