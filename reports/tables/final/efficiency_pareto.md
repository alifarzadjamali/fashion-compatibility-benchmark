| analysis      | cost         | representation      |    score |   cost_value | pareto_dominated   | dominated_by                                     |
|:--------------|:-------------|:--------------------|---------:|-------------:|:-------------------|:-------------------------------------------------|
| polyvore_cp   | latency_ms   | resnet50            | 0.69375  |      1.97587 | True               | marqo_fashionsiglip                              |
| polyvore_cp   | latency_ms   | dinov3_vitl16       | 0.7822   |      3.14497 | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_cp   | latency_ms   | clip_vitl14_336     | 0.888303 |      9.31316 | True               | siglip2_b16_384;marqo_fashionsiglip              |
| polyvore_cp   | latency_ms   | siglip2_b16_384     | 0.889167 |      5.882   | True               | marqo_fashionsiglip                              |
| polyvore_cp   | latency_ms   | fashionclip2        | 0.868957 |      2.28229 | True               | marqo_fashionsiglip                              |
| polyvore_cp   | latency_ms   | marqo_fashionsiglip | 0.889676 |      1.90741 | False              |                                                  |
| polyvore_cp   | latency_ms   | gr_lite             | 0.827677 |      6.82791 | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_cp   | gflops       | resnet50            | 0.69375  |      8.17427 | False              |                                                  |
| polyvore_cp   | gflops       | dinov3_vitl16       | 0.7822   |    125.68    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_cp   | gflops       | clip_vitl14_336     | 0.888303 |    381.92    | True               | siglip2_b16_384;marqo_fashionsiglip              |
| polyvore_cp   | gflops       | siglip2_b16_384     | 0.889167 |    112.127   | True               | marqo_fashionsiglip                              |
| polyvore_cp   | gflops       | fashionclip2        | 0.868957 |      8.81762 | False              |                                                  |
| polyvore_cp   | gflops       | marqo_fashionsiglip | 0.889676 |     35.4166  | False              |                                                  |
| polyvore_cp   | gflops       | gr_lite             | 0.827677 |    289.623   | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_cp   | peak_vram_mb | resnet50            | 0.69375  |    763.662   | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_cp   | peak_vram_mb | dinov3_vitl16       | 0.7822   |    885.224   | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_cp   | peak_vram_mb | clip_vitl14_336     | 0.888303 |   2029.1     | True               | siglip2_b16_384;marqo_fashionsiglip              |
| polyvore_cp   | peak_vram_mb | siglip2_b16_384     | 0.889167 |   1442.93    | True               | marqo_fashionsiglip                              |
| polyvore_cp   | peak_vram_mb | fashionclip2        | 0.868957 |    400.792   | False              |                                                  |
| polyvore_cp   | peak_vram_mb | marqo_fashionsiglip | 0.889676 |    651.19    | False              |                                                  |
| polyvore_cp   | peak_vram_mb | gr_lite             | 0.827677 |   1686.69    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_fitb | latency_ms   | resnet50            | 0.406287 |      1.97587 | True               | marqo_fashionsiglip                              |
| polyvore_fitb | latency_ms   | dinov3_vitl16       | 0.504821 |      3.14497 | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_fitb | latency_ms   | clip_vitl14_336     | 0.64998  |      9.31316 | True               | siglip2_b16_384                                  |
| polyvore_fitb | latency_ms   | siglip2_b16_384     | 0.65282  |      5.882   | False              |                                                  |
| polyvore_fitb | latency_ms   | fashionclip2        | 0.617488 |      2.28229 | True               | marqo_fashionsiglip                              |
| polyvore_fitb | latency_ms   | marqo_fashionsiglip | 0.647933 |      1.90741 | False              |                                                  |
| polyvore_fitb | latency_ms   | gr_lite             | 0.557324 |      6.82791 | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_fitb | gflops       | resnet50            | 0.406287 |      8.17427 | False              |                                                  |
| polyvore_fitb | gflops       | dinov3_vitl16       | 0.504821 |    125.68    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_fitb | gflops       | clip_vitl14_336     | 0.64998  |    381.92    | True               | siglip2_b16_384                                  |
| polyvore_fitb | gflops       | siglip2_b16_384     | 0.65282  |    112.127   | False              |                                                  |
| polyvore_fitb | gflops       | fashionclip2        | 0.617488 |      8.81762 | False              |                                                  |
| polyvore_fitb | gflops       | marqo_fashionsiglip | 0.647933 |     35.4166  | False              |                                                  |
| polyvore_fitb | gflops       | gr_lite             | 0.557324 |    289.623   | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| polyvore_fitb | peak_vram_mb | resnet50            | 0.406287 |    763.662   | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_fitb | peak_vram_mb | dinov3_vitl16       | 0.504821 |    885.224   | True               | fashionclip2;marqo_fashionsiglip                 |
| polyvore_fitb | peak_vram_mb | clip_vitl14_336     | 0.64998  |   2029.1     | True               | siglip2_b16_384                                  |
| polyvore_fitb | peak_vram_mb | siglip2_b16_384     | 0.65282  |   1442.93    | False              |                                                  |
| polyvore_fitb | peak_vram_mb | fashionclip2        | 0.617488 |    400.792   | False              |                                                  |
| polyvore_fitb | peak_vram_mb | marqo_fashionsiglip | 0.647933 |    651.19    | False              |                                                  |
| polyvore_fitb | peak_vram_mb | gr_lite             | 0.557324 |   1686.69    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_cp       | latency_ms   | resnet50            | 0.585234 |      1.97587 | True               | marqo_fashionsiglip                              |
| iqon_cp       | latency_ms   | dinov3_vitl16       | 0.64188  |      3.14497 | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_cp       | latency_ms   | clip_vitl14_336     | 0.704243 |      9.31316 | True               | marqo_fashionsiglip                              |
| iqon_cp       | latency_ms   | siglip2_b16_384     | 0.703107 |      5.882   | True               | marqo_fashionsiglip                              |
| iqon_cp       | latency_ms   | fashionclip2        | 0.698052 |      2.28229 | True               | marqo_fashionsiglip                              |
| iqon_cp       | latency_ms   | marqo_fashionsiglip | 0.722143 |      1.90741 | False              |                                                  |
| iqon_cp       | latency_ms   | gr_lite             | 0.651079 |      6.82791 | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_cp       | gflops       | resnet50            | 0.585234 |      8.17427 | False              |                                                  |
| iqon_cp       | gflops       | dinov3_vitl16       | 0.64188  |    125.68    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_cp       | gflops       | clip_vitl14_336     | 0.704243 |    381.92    | True               | marqo_fashionsiglip                              |
| iqon_cp       | gflops       | siglip2_b16_384     | 0.703107 |    112.127   | True               | marqo_fashionsiglip                              |
| iqon_cp       | gflops       | fashionclip2        | 0.698052 |      8.81762 | False              |                                                  |
| iqon_cp       | gflops       | marqo_fashionsiglip | 0.722143 |     35.4166  | False              |                                                  |
| iqon_cp       | gflops       | gr_lite             | 0.651079 |    289.623   | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_cp       | peak_vram_mb | resnet50            | 0.585234 |    763.662   | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_cp       | peak_vram_mb | dinov3_vitl16       | 0.64188  |    885.224   | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_cp       | peak_vram_mb | clip_vitl14_336     | 0.704243 |   2029.1     | True               | marqo_fashionsiglip                              |
| iqon_cp       | peak_vram_mb | siglip2_b16_384     | 0.703107 |   1442.93    | True               | marqo_fashionsiglip                              |
| iqon_cp       | peak_vram_mb | fashionclip2        | 0.698052 |    400.792   | False              |                                                  |
| iqon_cp       | peak_vram_mb | marqo_fashionsiglip | 0.722143 |    651.19    | False              |                                                  |
| iqon_cp       | peak_vram_mb | gr_lite             | 0.651079 |   1686.69    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_fitb     | latency_ms   | resnet50            | 0.316613 |      1.97587 | True               | marqo_fashionsiglip                              |
| iqon_fitb     | latency_ms   | dinov3_vitl16       | 0.366008 |      3.14497 | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_fitb     | latency_ms   | clip_vitl14_336     | 0.436411 |      9.31316 | True               | marqo_fashionsiglip                              |
| iqon_fitb     | latency_ms   | siglip2_b16_384     | 0.433024 |      5.882   | True               | marqo_fashionsiglip                              |
| iqon_fitb     | latency_ms   | fashionclip2        | 0.4275   |      2.28229 | True               | marqo_fashionsiglip                              |
| iqon_fitb     | latency_ms   | marqo_fashionsiglip | 0.445081 |      1.90741 | False              |                                                  |
| iqon_fitb     | latency_ms   | gr_lite             | 0.383185 |      6.82791 | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_fitb     | gflops       | resnet50            | 0.316613 |      8.17427 | False              |                                                  |
| iqon_fitb     | gflops       | dinov3_vitl16       | 0.366008 |    125.68    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_fitb     | gflops       | clip_vitl14_336     | 0.436411 |    381.92    | True               | marqo_fashionsiglip                              |
| iqon_fitb     | gflops       | siglip2_b16_384     | 0.433024 |    112.127   | True               | marqo_fashionsiglip                              |
| iqon_fitb     | gflops       | fashionclip2        | 0.4275   |      8.81762 | False              |                                                  |
| iqon_fitb     | gflops       | marqo_fashionsiglip | 0.445081 |     35.4166  | False              |                                                  |
| iqon_fitb     | gflops       | gr_lite             | 0.383185 |    289.623   | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |
| iqon_fitb     | peak_vram_mb | resnet50            | 0.316613 |    763.662   | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_fitb     | peak_vram_mb | dinov3_vitl16       | 0.366008 |    885.224   | True               | fashionclip2;marqo_fashionsiglip                 |
| iqon_fitb     | peak_vram_mb | clip_vitl14_336     | 0.436411 |   2029.1     | True               | marqo_fashionsiglip                              |
| iqon_fitb     | peak_vram_mb | siglip2_b16_384     | 0.433024 |   1442.93    | True               | marqo_fashionsiglip                              |
| iqon_fitb     | peak_vram_mb | fashionclip2        | 0.4275   |    400.792   | False              |                                                  |
| iqon_fitb     | peak_vram_mb | marqo_fashionsiglip | 0.445081 |    651.19    | False              |                                                  |
| iqon_fitb     | peak_vram_mb | gr_lite             | 0.383185 |   1686.69    | True               | siglip2_b16_384;fashionclip2;marqo_fashionsiglip |