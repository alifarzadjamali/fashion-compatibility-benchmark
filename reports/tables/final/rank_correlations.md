| comparison                        |   n_models |   spearman_rho |   spearman_p |   kendall_tau |   kendall_p |
|:----------------------------------|-----------:|---------------:|-------------:|--------------:|------------:|
| historical_polyvore_d: CP vs FITB |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| polyvore_d_clean: CP vs FITB      |          7 |       0.892857 |  0.00680719  |      0.809524 | 0.0107143   |
| historical vs clean CP            |          7 |       1        |  0           |      1        | 0.000396825 |
| historical vs clean FITB          |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| historical: CP PCA256 vs pca128   |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| historical: FITB PCA256 vs pca128 |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| historical: CP PCA256 vs pca512   |          7 |       1        |  0           |      1        | 0.000396825 |
| historical: FITB PCA256 vs pca512 |          7 |       1        |  0           |      1        | 0.000396825 |
| historical: CP PCA256 vs native   |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| historical: FITB PCA256 vs native |          7 |       0.928571 |  0.00251947  |      0.809524 | 0.0107143   |
| clean: CP PCA256 vs pca128        |          7 |       0.892857 |  0.00680719  |      0.809524 | 0.0107143   |
| clean: FITB PCA256 vs pca128      |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| clean: CP PCA256 vs pca512        |          7 |       1        |  0           |      1        | 0.000396825 |
| clean: FITB PCA256 vs pca512      |          7 |       0.892857 |  0.00680719  |      0.809524 | 0.0107143   |
| clean: CP PCA256 vs native        |          7 |       0.928571 |  0.00251947  |      0.809524 | 0.0107143   |
| clean: FITB PCA256 vs native      |          7 |       0.821429 |  0.0234488   |      0.714286 | 0.0301587   |
| Polyvore CP vs IQON CP            |          7 |       0.964286 |  0.000454149 |      0.904762 | 0.00277778  |
| Polyvore FITB vs IQON FITB        |          7 |       0.857143 |  0.0136973   |      0.714286 | 0.0301587   |
| IQON CP vs FITB                   |          7 |       1        |  0           |      1        | 0.000396825 |
| polyvore_d_clean CP vs A100 LAT   |          7 |       0.785714 |  0.0362385   |      0.714286 | 0.0301587   |
| polyvore_d_clean CP vs A100 AAT   |          7 |       0.892857 |  0.00680719  |      0.809524 | 0.0107143   |
| iqon3000_clean CP vs A100 LAT     |          7 |       0.892857 |  0.00680719  |      0.809524 | 0.0107143   |
| iqon3000_clean CP vs A100 AAT     |          7 |       0.72075  |  0.067635    |      0.58554  | 0.0683139   |