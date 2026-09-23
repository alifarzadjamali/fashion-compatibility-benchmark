# Final cross-dataset report

Polyvore-D-Clean and IQON3000-Clean agree strongly on representation ordering despite a large
absolute domain shift: CP rho=.964/tau=.905 and FITB rho=.857/tau=.714. Marqo leads IQON CP/FITB;
Polyvore's leading group is Marqo/SigLIP2/CLIP with overlapping CP intervals. This replication
supports a broad VLM-over-CNN/self-supervised conclusion, not a universal total order among the top
three.

A100 transfer is directionally consistent but noisier. Polyvore CP rank versus A100 LAT/AAT gives
rho=.786/.893; IQON CP rank versus A100 LAT/AAT gives rho=.893/.721. Polyvore-trained scorers are
usually stronger on A100 LAT, while IQON-trained SigLIP2/Marqo remain strongest on AAT. Since each
A100 task has only 100 questions, facet and pairwise rank changes are treated descriptively.

LookBench exact-checkpoint overlap contains five models. Retrieval rank correlates weakly with
Polyvore compatibility (rho=.20 for CP, .10 to -.20 for FITB, all non-significant descriptively),
with multiple rank reversals. The defensible conclusion is that fashion retrieval strength is not a
reliable proxy for relational outfit compatibility in this small exact-match subset; it is not proof
of zero association.

The provenance-restricted sensitivity excludes Marqo and GR-Lite. SigLIP2 then leads Polyvore and CLIP
leads IQON, while FashionCLIP trails its generic CLIP counterpart. Hence the broad modern-VLM signal
survives, whereas a general fashion-specialization advantage does not.
