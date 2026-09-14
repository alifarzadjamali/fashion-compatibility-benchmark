# Final efficiency report

All encoders were profiled on the same RTX 5070 Ti 16 GB with CUDA synchronization, warm-up, repeated
steady-state measurement, documented precision, and one PyTorch forward FLOP-count definition.
FLOPs are estimates, not hardware-independent latency guarantees. Active image parameters and whole
checkpoint parameters are reported separately.

Marqo offers the strongest accuracy-efficiency trade-off in the full roster: about 35.4 estimated
GFLOPs/image, 1.91 ms warmed batch latency, and 0.65 GB peak VRAM, while leading IQON CP/FITB and
Polyvore CP. FashionCLIP is similarly light (8.8 GFLOPs, 2.28 ms, 0.40 GB) but less accurate. CLIP
ViT-L/14@336 is the costliest measured encoder at about 381.9 GFLOPs, 9.31 ms, and 2.03 GB peak VRAM.
DINOv3 and GR-Lite require about 125.7 and 289.6 GFLOPs respectively; GR-Lite's accuracy gain over
DINOv3 is accompanied by materially higher image cost.

For the 434,089 IQON items, extraction ranged from roughly 409 s (ResNet50) to 4,778 s (CLIP), while
cached embeddings prevented any repeated encoder work. Dataset extraction used batch 32 for every
representation; steady-state profiling used batch 64. PCA and LR fitting took seconds and are not the
dominant compute cost. Exact checkpoint size, cache size, throughput, VRAM, PCA time, and classifier
time are in `reports/tables/final/efficiency_full.csv`; dominance relationships are in
`reports/tables/final/efficiency_pareto.csv`.

