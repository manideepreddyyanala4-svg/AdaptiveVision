# Anomaly-detection model benchmark

22 methods across 29 dataset configurations and 2 regimes -- 3428 successful runs, 670 failed.

**Headline -- one model, every category.** `dinomaly_vitb14` reaches **0.9771 mean AUROC** across all 27 configurations from a single fitted model per dataset family (worst config 0.8909, 15.8 ms/image).

On the 27 configurations both regimes cover, the best multi-class model (`dinomaly_vitb14`, 0.9771) lands +3.85 points against the best per-category model (`dfm_dinov2_vitb14`, 0.9385) -- which needs one checkpoint per category rather than one per dataset family.

## Regime comparison

Mean AUROC per method under each deployment regime, over the 27 configurations every regime covers. A method whose multi-class column matches its one-class column ships as one artifact per dataset family instead of one per category.

| method                          |   multiclass |   oneclass |
|:--------------------------------|-------------:|-----------:|
| dinomaly_vitb14                 |       0.9771 |   nan      |
| dinomaly_vits14                 |       0.9678 |   nan      |
| dinomaly_vitl14                 |       0.9107 |   nan      |
| patchcore_dense_wide_resnet50_2 |       0.8941 |     0.9161 |
| patchcore_dinov2_vitb14         |       0.8879 |     0.931  |
| patchcore_dinov2_vitl14         |       0.8828 |     0.9314 |
| patchcore_knn3_wide_resnet50_2  |       0.876  |     0.9188 |
| patchcore_convnext_small        |       0.8758 |     0.9131 |
| patchcore_wide_resnet50_2       |       0.8748 |     0.9219 |
| patchcore_efficientnet_b4       |       0.8746 |     0.9216 |
| patchcore_resnet50              |       0.8631 |     0.9247 |
| dfm_dinov2_vitb14               |       0.8398 |     0.9385 |
| patchcore_resnet18              |       0.8387 |     0.9182 |
| dfm_dinov2_vitl14               |       0.8165 |     0.9269 |
| dfm_efficientnet_b4             |       0.7574 |     0.8932 |
| dfm_resnet50                    |       0.7452 |     0.8908 |
| dfm_wide_resnet50_2             |       0.7447 |     0.8928 |
| padim_wide_resnet50_2           |       0.7367 |     0.8488 |
| dfm_resnet18                    |       0.7135 |     0.8661 |
| padim_pooled_resnet18           |       0.6304 |     0.7514 |
| padim_pooled_wide_resnet50_2    |       0.5939 |     0.7138 |
| dfm_convnext_small              |       0.4992 |     0.8845 |

## Ranking -- multiclass

`scrap_at_95` is the share of good parts rejected when tuned to catch 95% of defects; `escape_at_1fpr` is the share of defects missed within a 1% false-alarm budget. Methods that did not complete every configuration are listed last and are not comparable to a complete row.

|   rank | regime     | method                          | family    | backend   |   configs |   min_auroc |   peak_vram_gb | mean_auroc        | mean_ap           | mean_f1           | scrap_at_95       | escape_at_1fpr    | mean_pg2          | mean_pb2          | bal_error         | mean_aupro        | mean_aupimo       | mean_pixel_auroc   | ms_per_image   | fit_seconds    |   latency_p50_ms |   latency_p95_ms |   throughput_fps_bs1 |   throughput_fps_bs16 |   model_params_m |   peak_gpu_mb | train_wall_clock_s   | complete   |
|-------:|:-----------|:--------------------------------|:----------|:----------|----------:|------------:|---------------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:-------------------|:---------------|:---------------|-----------------:|-----------------:|---------------------:|----------------------:|-----------------:|--------------:|:---------------------|:-----------|
|      1 | multiclass | dinomaly_vitb14                 | dinomaly  | native    |        27 |      0.8909 |           9.39 | 0.9771 +/- 0.0013 | 0.9824 +/- 0.0008 | 0.9591 +/- 0.0024 | 0.0937 +/- 0.0113 | 0.2448 +/- 0.0135 | 0.8223 +/- 0.0210 | 0.8352 +/- 0.0118 | 0.0560 +/- 0.0045 | 0.8975 +/- 0.0229 | 0.9104 +/- 0.0026 | 0.9784 +/- 0.0007  | 15.8 +/- 0.1   | 3103.4 +/- 3.6 |              nan |              nan |                  nan |                   nan |              nan |           nan | 3103.4 +/- 3.6       | True       |
|      2 | multiclass | dinomaly_vits14                 | dinomaly  | native    |        27 |      0.8256 |           4.43 | 0.9678 +/- 0.0019 | 0.9736 +/- 0.0013 | 0.9486 +/- 0.0034 | 0.1334 +/- 0.0154 | 0.2835 +/- 0.0189 | 0.7852 +/- 0.0204 | 0.7750 +/- 0.0204 | 0.0680 +/- 0.0077 | 0.8795 +/- 0.0270 | 0.8991 +/- 0.0025 | 0.9741 +/- 0.0011  | 8.3 +/- 0.1    | 1006.9 +/- 0.4 |              nan |              nan |                  nan |                   nan |              nan |           nan | 1006.9 +/- 0.4       | True       |
|      3 | multiclass | dinomaly_vitl14                 | dinomaly  | native    |        27 |      0.4691 |           8.59 | 0.9107 +/- 0.0048 | 0.9374 +/- 0.0037 | 0.9071 +/- 0.0054 | 0.3108 +/- 0.0168 | 0.4376 +/- 0.0243 | 0.6218 +/- 0.0402 | 0.6381 +/- 0.0255 | 0.1531 +/- 0.0131 | 0.8092 +/- 0.0239 | 0.8283 +/- 0.0046 | 0.9224 +/- 0.0027  | 33.2 +/- 0.1   | 3472.0 +/- 1.5 |              nan |              nan |                  nan |                   nan |              nan |           nan | 3472.0 +/- 1.5       | True       |
|      4 | multiclass | patchcore_dense_wide_resnet50_2 | patchcore | native    |        27 |      0.6695 |           1.56 | 0.8941 +/- 0.0122 | 0.9178 +/- 0.0092 | 0.8937 +/- 0.0089 | 0.3508 +/- 0.0547 | 0.5362 +/- 0.0567 | 0.5426 +/- 0.0654 | 0.5240 +/- 0.0475 | 0.1717 +/- 0.0253 | 0.7980 +/- 0.0297 | 0.8340 +/- 0.0065 | 0.9668 +/- 0.0030  | 7.5 +/- 0.1    | 88.6 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      5 | multiclass | patchcore_dinov2_vitb14         | patchcore | native    |        27 |      0.6008 |           1.52 | 0.8879 +/- 0.0215 | 0.9167 +/- 0.0156 | 0.8848 +/- 0.0145 | 0.4101 +/- 0.0729 | 0.4003 +/- 0.0773 | 0.4925 +/- 0.0829 | 0.6454 +/- 0.0582 | 0.1785 +/- 0.0391 | 0.6921 +/- 0.0350 | 0.7114 +/- 0.0190 | 0.9051 +/- 0.0075  | 18.2 +/- 0.1   | 95.4 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      6 | multiclass | patchcore_dinov2_vitl14         | patchcore | native    |        27 |      0.5593 |           2.39 | 0.8828 +/- 0.0184 | 0.9123 +/- 0.0122 | 0.8842 +/- 0.0134 | 0.3980 +/- 0.0770 | 0.4271 +/- 0.0692 | 0.5011 +/- 0.0917 | 0.6256 +/- 0.0528 | 0.1811 +/- 0.0357 | 0.6884 +/- 0.0305 | 0.7005 +/- 0.0173 | 0.9012 +/- 0.0096  | 48.5 +/- 0.1   | 271.6 +/- 0.1  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      7 | multiclass | patchcore_knn3_wide_resnet50_2  | patchcore | native    |        27 |      0.6564 |           1.29 | 0.8760 +/- 0.0151 | 0.9113 +/- 0.0107 | 0.8798 +/- 0.0112 | 0.4359 +/- 0.0637 | 0.5423 +/- 0.0764 | 0.4588 +/- 0.0917 | 0.5179 +/- 0.0648 | 0.2038 +/- 0.0380 | 0.7825 +/- 0.0285 | 0.8158 +/- 0.0152 | 0.9571 +/- 0.0045  | 6.5 +/- 0.2    | 20.1 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      8 | multiclass | patchcore_convnext_small        | patchcore | native    |        27 |      0.5783 |           1.32 | 0.8758 +/- 0.0116 | 0.9077 +/- 0.0077 | 0.8782 +/- 0.0086 | 0.4559 +/- 0.0601 | 0.5458 +/- 0.0544 | 0.4304 +/- 0.0597 | 0.5312 +/- 0.0514 | 0.1955 +/- 0.0239 | 0.7016 +/- 0.0361 | 0.7406 +/- 0.0179 | 0.9240 +/- 0.0075  | 6.4 +/- 0.1    | 20.7 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      9 | multiclass | patchcore_wide_resnet50_2       | patchcore | native    |        27 |      0.6457 |           1.29 | 0.8748 +/- 0.0153 | 0.9136 +/- 0.0093 | 0.8774 +/- 0.0120 | 0.4560 +/- 0.0656 | 0.5248 +/- 0.0807 | 0.4273 +/- 0.0812 | 0.5354 +/- 0.0666 | 0.2075 +/- 0.0390 | 0.6960 +/- 0.0349 | 0.7308 +/- 0.0195 | 0.9221 +/- 0.0097  | 6.5 +/- 0.1    | 20.5 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     10 | multiclass | patchcore_efficientnet_b4       | patchcore | native    |        27 |      0.5739 |           1.98 | 0.8746 +/- 0.0233 | 0.9001 +/- 0.0139 | 0.8785 +/- 0.0178 | 0.4311 +/- 0.0712 | 0.5896 +/- 0.0662 | 0.4845 +/- 0.0841 | 0.4879 +/- 0.0608 | 0.2006 +/- 0.0414 | 0.6745 +/- 0.0371 | 0.7140 +/- 0.0175 | 0.9130 +/- 0.0084  | 6.6 +/- 0.1    | 15.5 +/- 0.3   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     11 | multiclass | patchcore_resnet50              | patchcore | native    |        27 |      0.658  |           1.22 | 0.8631 +/- 0.0202 | 0.9087 +/- 0.0125 | 0.8688 +/- 0.0121 | 0.5030 +/- 0.0872 | 0.5245 +/- 0.0582 | 0.3816 +/- 0.0950 | 0.5355 +/- 0.0508 | 0.2235 +/- 0.0361 | 0.6673 +/- 0.0324 | 0.7090 +/- 0.0195 | 0.9171 +/- 0.0107  | 6.2 +/- 0.1    | 20.0 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     12 | multiclass | dfm_dinov2_vitb14               | dfm       | native    |        27 |      0.5506 |           0.61 | 0.8398 +/- 0.0000 | 0.8841 +/- 0.0000 | 0.8577 +/- 0.0000 | 0.5173 +/- 0.0000 | 0.6365 +/- 0.0000 | 0.4142 +/- 0.0000 | 0.4212 +/- 0.0000 | 0.2574 +/- 0.0000 | nan               | nan               | nan                | 16.4 +/- 0.1   | 84.9 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     13 | multiclass | patchcore_resnet18              | patchcore | native    |        27 |      0.6664 |           1.2  | 0.8387 +/- 0.0238 | 0.9001 +/- 0.0142 | 0.8499 +/- 0.0134 | 0.6013 +/- 0.0721 | 0.5424 +/- 0.0585 | 0.2894 +/- 0.0822 | 0.5029 +/- 0.0525 | 0.2718 +/- 0.0428 | 0.6345 +/- 0.0431 | 0.6689 +/- 0.0303 | 0.8953 +/- 0.0162  | 5.8 +/- 0.1    | 19.4 +/- 0.2   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     14 | multiclass | dfm_dinov2_vitl14               | dfm       | native    |        27 |      0.5295 |           1.56 | 0.8165 +/- 0.0000 | 0.8706 +/- 0.0000 | 0.8556 +/- 0.0000 | 0.5305 +/- 0.0000 | 0.6662 +/- 0.0000 | 0.3808 +/- 0.0000 | 0.3864 +/- 0.0000 | 0.2665 +/- 0.0000 | nan               | nan               | nan                | 46.6 +/- 0.1   | 261.8 +/- 0.6  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     15 | multiclass | dfm_efficientnet_b4             | dfm       | native    |        27 |      0.5188 |           1.71 | 0.7574 +/- 0.0000 | 0.8358 +/- 0.0000 | 0.8398 +/- 0.0000 | 0.6724 +/- 0.0000 | 0.7802 +/- 0.0000 | 0.2468 +/- 0.0000 | 0.2711 +/- 0.0000 | 0.3227 +/- 0.0000 | nan               | nan               | nan                | 4.7 +/- 0.1    | 10.2 +/- 2.5   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     16 | multiclass | dfm_resnet50                    | dfm       | native    |        27 |      0.3617 |           0.39 | 0.7452 +/- 0.0000 | 0.8341 +/- 0.0000 | 0.8370 +/- 0.0000 | 0.6804 +/- 0.0000 | 0.7727 +/- 0.0000 | 0.2232 +/- 0.0000 | 0.2985 +/- 0.0000 | 0.3295 +/- 0.0000 | nan               | nan               | nan                | 4.9 +/- 0.4    | 9.4 +/- 0.4    |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     17 | multiclass | dfm_wide_resnet50_2             | dfm       | native    |        27 |      0.3367 |           0.58 | 0.7447 +/- 0.0000 | 0.8332 +/- 0.0000 | 0.8401 +/- 0.0000 | 0.6816 +/- 0.0000 | 0.7549 +/- 0.0000 | 0.2331 +/- 0.0000 | 0.3055 +/- 0.0000 | 0.3184 +/- 0.0000 | nan               | nan               | nan                | 5.2 +/- 0.5    | 10.1 +/- 0.2   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     18 | multiclass | padim_wide_resnet50_2           | padim     | native    |        27 |      0.557  |           1.54 | 0.7367 +/- 0.0493 | 0.8064 +/- 0.0394 | 0.8280 +/- 0.0166 | 0.6764 +/- 0.0955 | 0.8244 +/- 0.0940 | 0.2401 +/- 0.0854 | 0.2120 +/- 0.1053 | 0.3424 +/- 0.0518 | 0.6400 +/- 0.0697 | 0.6960 +/- 0.0340 | 0.9147 +/- 0.0157  | 6.0 +/- 0.1    | 11.9 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     19 | multiclass | dfm_resnet18                    | dfm       | native    |        27 |      0.3759 |           0.55 | 0.7135 +/- 0.0000 | 0.8121 +/- 0.0000 | 0.8321 +/- 0.0000 | 0.7267 +/- 0.0000 | 0.8223 +/- 0.0000 | 0.2007 +/- 0.0000 | 0.2269 +/- 0.0000 | 0.3495 +/- 0.0000 | nan               | nan               | nan                | 4.4 +/- 0.2    | 8.9 +/- 0.8    |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     20 | multiclass | padim_pooled_resnet18           | padim     | native    |        27 |      0.4352 |           0.38 | 0.6304 +/- 0.0487 | 0.7544 +/- 0.0325 | 0.7981 +/- 0.0072 | 0.8704 +/- 0.0583 | 0.8625 +/- 0.0499 | 0.0865 +/- 0.0506 | 0.1737 +/- 0.0581 | 0.4329 +/- 0.0343 | 0.5871 +/- 0.0316 | 0.6138 +/- 0.0240 | 0.8387 +/- 0.0138  | 5.0 +/- 0.1    | 9.2 +/- 0.1    |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     21 | multiclass | padim_pooled_wide_resnet50_2    | padim     | native    |        27 |      0.4573 |           1.21 | 0.5939 +/- 0.0861 | 0.7253 +/- 0.0596 | 0.7941 +/- 0.0101 | 0.8760 +/- 0.0803 | 0.9035 +/- 0.0768 | 0.0753 +/- 0.0680 | 0.1215 +/- 0.0862 | 0.4441 +/- 0.0522 | 0.5580 +/- 0.0623 | 0.5858 +/- 0.0496 | 0.8286 +/- 0.0305  | 6.0 +/- 0.1    | 14.5 +/- 0.0   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     22 | multiclass | dfm_convnext_small              | dfm       | native    |        27 |      0.1334 |           0.49 | 0.4992 +/- 0.0000 | 0.6815 +/- 0.0000 | 0.7916 +/- 0.0000 | 0.9220 +/- 0.0000 | 0.9006 +/- 0.0000 | 0.0638 +/- 0.0000 | 0.1204 +/- 0.0000 | 0.4714 +/- 0.0000 | nan               | nan               | nan                | 5.3 +/- 1.4    | 10.5 +/- 0.3   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |

### Mean AUROC by dataset family -- multiclass

| method                          |   mvtec |   visa |
|:--------------------------------|--------:|-------:|
| dfm_convnext_small              |  0.5169 | 0.477  |
| dfm_dinov2_vitb14               |  0.868  | 0.8045 |
| dfm_dinov2_vitl14               |  0.8357 | 0.7925 |
| dfm_efficientnet_b4             |  0.7512 | 0.7651 |
| dfm_resnet18                    |  0.6624 | 0.7773 |
| dfm_resnet50                    |  0.7102 | 0.7889 |
| dfm_wide_resnet50_2             |  0.7063 | 0.7928 |
| dinomaly_vitb14                 |  0.9898 | 0.9611 |
| dinomaly_vitl14                 |  0.9046 | 0.9184 |
| dinomaly_vits14                 |  0.9816 | 0.9504 |
| padim_pooled_resnet18           |  0.6633 | 0.5894 |
| padim_pooled_wide_resnet50_2    |  0.6327 | 0.5455 |
| padim_wide_resnet50_2           |  0.7918 | 0.6679 |
| patchcore_convnext_small        |  0.9145 | 0.8275 |
| patchcore_dense_wide_resnet50_2 |  0.9255 | 0.8549 |
| patchcore_dinov2_vitb14         |  0.955  | 0.8039 |
| patchcore_dinov2_vitl14         |  0.9509 | 0.7976 |
| patchcore_efficientnet_b4       |  0.9161 | 0.8228 |
| patchcore_knn3_wide_resnet50_2  |  0.9015 | 0.8442 |
| patchcore_resnet18              |  0.865  | 0.8058 |
| patchcore_resnet50              |  0.9027 | 0.8136 |
| patchcore_wide_resnet50_2       |  0.9065 | 0.8351 |

### Best method per configuration -- multiclass

| config           | method                         |   auroc |   average_precision |   f1_max |   ms_per_image |
|:-----------------|:-------------------------------|--------:|--------------------:|---------:|---------------:|
| mvtec/bottle     | dinomaly_vitb14                |  1      |              1      |   1      |           17.5 |
| mvtec/cable      | dinomaly_vits14                |  0.9875 |              0.9925 |   0.9533 |            8.4 |
| mvtec/capsule    | dinomaly_vitb14                |  0.9581 |              0.9909 |   0.9604 |           16.6 |
| mvtec/carpet     | dinomaly_vitb14                |  1      |              1      |   1      |           17.4 |
| mvtec/grid       | dinomaly_vitb14                |  1      |              1      |   1      |           17.7 |
| mvtec/hazelnut   | dinomaly_vitb14                |  1      |              1      |   1      |           17.5 |
| mvtec/leather    | dinomaly_vitb14                |  1      |              1      |   1      |           16.6 |
| mvtec/metal_nut  | dinomaly_vitb14                |  1      |              1      |   1      |           15.6 |
| mvtec/pill       | dinomaly_vitb14                |  0.9776 |              0.9959 |   0.9747 |           14.8 |
| mvtec/screw      | dinomaly_vitb14                |  0.9443 |              0.9815 |   0.9295 |           14.6 |
| mvtec/tile       | dinomaly_vitb14                |  1      |              1      |   1      |           16.2 |
| mvtec/toothbrush | dinomaly_vitb14                |  1      |              1      |   1      |           26.2 |
| mvtec/transistor | dinomaly_vitb14                |  0.9853 |              0.9727 |   0.9593 |           18.2 |
| mvtec/wood       | dinomaly_vitl14                |  1      |              1      |   1      |           36   |
| mvtec/zipper     | dinomaly_vitb14                |  0.9996 |              0.9999 |   0.9958 |           14.7 |
| visa/candle      | dinomaly_vitb14                |  0.9623 |              0.9608 |   0.9036 |           13.7 |
| visa/capsules    | dinomaly_vitb14                |  0.9508 |              0.9714 |   0.9144 |           14.3 |
| visa/cashew      | dfm_wide_resnet50_2            |  0.9794 |              0.9887 |   0.9608 |            3.7 |
| visa/chewinggum  | patchcore_knn3_wide_resnet50_2 |  0.9796 |              0.9913 |   0.9627 |            4.9 |
| visa/fryum       | dinomaly_vits14                |  0.9804 |              0.9907 |   0.9516 |            6.8 |
| visa/macaroni1   | dinomaly_vitb14                |  0.9508 |              0.9476 |   0.905  |           13.8 |
| visa/macaroni2   | dinomaly_vitb14                |  0.8909 |              0.8689 |   0.8525 |           13.8 |
| visa/pcb1        | dinomaly_vitb14                |  0.9788 |              0.9759 |   0.9435 |           13.7 |
| visa/pcb2        | dinomaly_vitb14                |  0.9607 |              0.9661 |   0.9049 |           13.6 |
| visa/pcb3        | dinomaly_vitb14                |  0.9782 |              0.9762 |   0.9376 |           13.6 |
| visa/pcb4        | dinomaly_vitb14                |  0.9941 |              0.9936 |   0.9734 |           13.7 |
| visa/pipe_fryum  | dinomaly_vits14                |  0.9746 |              0.9846 |   0.9732 |            6.8 |

## Ranking -- oneclass

`scrap_at_95` is the share of good parts rejected when tuned to catch 95% of defects; `escape_at_1fpr` is the share of defects missed within a 1% false-alarm budget. Methods that did not complete every configuration are listed last and are not comparable to a complete row.

`mean_ap`/`mean_f1` are omitted from this table: Severstal's test split runs ~92% positive (see the Severstal caveat below), which would make a blended cross-dataset AP/F1 mean incomparable. AUROC and PG2/PB2 are less prevalence-sensitive and stay.

|   rank | regime   | method                          | family    | backend   |   configs |   min_auroc |   peak_vram_gb | mean_auroc        | scrap_at_95       | escape_at_1fpr    | mean_pg2          | mean_pb2          | bal_error         | mean_aupro        | mean_aupimo       | mean_pixel_auroc   | ms_per_image   | fit_seconds   |   latency_p50_ms |   latency_p95_ms |   throughput_fps_bs1 |   throughput_fps_bs16 |   model_params_m |   peak_gpu_mb | train_wall_clock_s   | complete   |
|-------:|:---------|:--------------------------------|:----------|:----------|----------:|------------:|---------------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:-------------------|:---------------|:--------------|-----------------:|-----------------:|---------------------:|----------------------:|-----------------:|--------------:|:---------------------|:-----------|
|      1 | oneclass | dfm_dinov2_vitb14               | dfm       | native    |        29 |      0.6044 |           0.99 | 0.9329 +/- 0.0018 | 0.2460 +/- 0.0101 | 0.4091 +/- 0.0132 | 0.6759 +/- 0.0162 | 0.6658 +/- 0.0158 | 0.1257 +/- 0.0028 | nan               | nan               | nan                | 16.0 +/- 0.1   | 10.8 +/- 0.5  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      2 | oneclass | patchcore_dinov2_vitl14         | patchcore | native    |        29 |      0.613  |           5.28 | 0.9239 +/- 0.0065 | 0.2668 +/- 0.0260 | 0.3416 +/- 0.0357 | 0.6545 +/- 0.0479 | 0.6957 +/- 0.0317 | 0.1252 +/- 0.0153 | 0.7708 +/- 0.0319 | 0.8015 +/- 0.0098 | 0.9522 +/- 0.0041  | 73.8 +/- 5.9   | 72.2 +/- 4.4  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      3 | oneclass | patchcore_dinov2_vitb14         | patchcore | native    |        29 |      0.6211 |           2.23 | 0.9236 +/- 0.0068 | 0.2691 +/- 0.0414 | 0.3614 +/- 0.0413 | 0.6547 +/- 0.0484 | 0.6840 +/- 0.0393 | 0.1300 +/- 0.0132 | 0.7794 +/- 0.0303 | 0.8103 +/- 0.0098 | 0.9561 +/- 0.0028  | 20.9 +/- 1.1   | 33.9 +/- 4.2  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      4 | oneclass | dfm_dinov2_vitl14               | dfm       | native    |        29 |      0.6262 |           6.33 | 0.9222 +/- 0.0013 | 0.2514 +/- 0.0082 | 0.4502 +/- 0.0129 | 0.6544 +/- 0.0136 | 0.6192 +/- 0.0159 | 0.1296 +/- 0.0034 | nan               | nan               | nan                | 101.7 +/- 5.8  | 57.3 +/- 8.8  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      5 | oneclass | patchcore_resnet50              | patchcore | native    |        29 |      0.5573 |           1.24 | 0.9128 +/- 0.0072 | 0.3203 +/- 0.0456 | 0.4485 +/- 0.0380 | 0.5692 +/- 0.0562 | 0.6261 +/- 0.0391 | 0.1472 +/- 0.0210 | 0.7821 +/- 0.0361 | 0.8331 +/- 0.0076 | 0.9657 +/- 0.0027  | 11.1 +/- 1.6   | 19.8 +/- 1.9  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      6 | oneclass | patchcore_wide_resnet50_2       | patchcore | native    |        29 |      0.5686 |           1.35 | 0.9107 +/- 0.0076 | 0.2984 +/- 0.0362 | 0.4635 +/- 0.0426 | 0.6104 +/- 0.0574 | 0.6049 +/- 0.0343 | 0.1454 +/- 0.0221 | 0.7839 +/- 0.0382 | 0.8308 +/- 0.0064 | 0.9635 +/- 0.0022  | 5.7 +/- 0.1    | 12.6 +/- 0.2  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      7 | oneclass | patchcore_efficientnet_b4       | patchcore | native    |        29 |      0.564  |           2.32 | 0.9100 +/- 0.0059 | 0.2886 +/- 0.0372 | 0.4873 +/- 0.0331 | 0.6203 +/- 0.0371 | 0.5958 +/- 0.0277 | 0.1353 +/- 0.0169 | 0.7834 +/- 0.0383 | 0.8318 +/- 0.0074 | 0.9619 +/- 0.0017  | 10.6 +/- 1.2   | 14.5 +/- 1.6  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      8 | oneclass | patchcore_knn3_wide_resnet50_2  | patchcore | native    |        29 |      0.5602 |           1.65 | 0.9076 +/- 0.0069 | 0.3063 +/- 0.0378 | 0.4802 +/- 0.0437 | 0.6151 +/- 0.0423 | 0.5935 +/- 0.0353 | 0.1532 +/- 0.0200 | 0.8141 +/- 0.0344 | 0.8540 +/- 0.0055 | 0.9714 +/- 0.0018  | 7.2 +/- 0.6    | 20.8 +/- 0.6  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|      9 | oneclass | patchcore_resnet18              | patchcore | native    |        29 |      0.5264 |           1.22 | 0.9055 +/- 0.0077 | 0.3495 +/- 0.0517 | 0.4748 +/- 0.0397 | 0.5258 +/- 0.0721 | 0.5978 +/- 0.0335 | 0.1629 +/- 0.0178 | 0.7765 +/- 0.0377 | 0.8282 +/- 0.0073 | 0.9617 +/- 0.0029  | 5.7 +/- 0.4    | 19.0 +/- 0.5  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     10 | oneclass | patchcore_dense_wide_resnet50_2 | patchcore | native    |        29 |      0.5567 |           2.33 | 0.9048 +/- 0.0058 | 0.3123 +/- 0.0321 | 0.4935 +/- 0.0408 | 0.5981 +/- 0.0459 | 0.5736 +/- 0.0334 | 0.1524 +/- 0.0160 | 0.8225 +/- 0.0300 | 0.8599 +/- 0.0042 | 0.9735 +/- 0.0009  | 6.8 +/- 0.2    | 89.5 +/- 0.4  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     11 | oneclass | patchcore_convnext_small        | patchcore | native    |        29 |      0.5416 |           1.42 | 0.9003 +/- 0.0072 | 0.3347 +/- 0.0511 | 0.4796 +/- 0.0444 | 0.5568 +/- 0.0565 | 0.5774 +/- 0.0362 | 0.1518 +/- 0.0228 | 0.7612 +/- 0.0341 | 0.8173 +/- 0.0069 | 0.9586 +/- 0.0016  | 6.2 +/- 1.1    | 13.4 +/- 0.2  |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     12 | oneclass | dfm_wide_resnet50_2             | dfm       | native    |        29 |      0.406  |           0.65 | 0.8883 +/- 0.0018 | 0.3392 +/- 0.0115 | 0.4989 +/- 0.0105 | 0.5733 +/- 0.0099 | 0.6077 +/- 0.0097 | 0.1691 +/- 0.0047 | nan               | nan               | nan                | 4.1 +/- 0.1    | 2.2 +/- 0.2   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     13 | oneclass | dfm_efficientnet_b4             | dfm       | native    |        29 |      0.5623 |           0.76 | 0.8883 +/- 0.0029 | 0.3742 +/- 0.0136 | 0.5057 +/- 0.0117 | 0.4953 +/- 0.0125 | 0.5527 +/- 0.0107 | 0.1705 +/- 0.0054 | nan               | nan               | nan                | 7.9 +/- 0.8    | 3.5 +/- 0.7   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     14 | oneclass | dfm_resnet50                    | dfm       | native    |        29 |      0.4069 |           0.58 | 0.8856 +/- 0.0023 | 0.3722 +/- 0.0112 | 0.4964 +/- 0.0128 | 0.5335 +/- 0.0118 | 0.5800 +/- 0.0083 | 0.1783 +/- 0.0039 | nan               | nan               | nan                | 8.1 +/- 0.6    | 5.5 +/- 1.9   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     15 | oneclass | dfm_convnext_small              | dfm       | native    |        29 |      0.5722 |           1.81 | 0.8724 +/- 0.0021 | 0.4452 +/- 0.0079 | 0.5123 +/- 0.0073 | 0.4602 +/- 0.0070 | 0.5514 +/- 0.0065 | 0.1983 +/- 0.0066 | nan               | nan               | nan                | 6.0 +/- 1.5    | 2.6 +/- 0.3   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     16 | oneclass | dfm_resnet18                    | dfm       | native    |        29 |      0.4085 |           0.41 | 0.8613 +/- 0.0030 | 0.4515 +/- 0.0140 | 0.5598 +/- 0.0146 | 0.4427 +/- 0.0161 | 0.5153 +/- 0.0191 | 0.2184 +/- 0.0059 | nan               | nan               | nan                | 7.7 +/- 0.6    | 4.3 +/- 0.8   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     17 | oneclass | padim_wide_resnet50_2           | padim     | native    |        29 |      0.4415 |           1.82 | 0.8384 +/- 0.0296 | 0.4838 +/- 0.0769 | 0.6372 +/- 0.1112 | 0.4140 +/- 0.0796 | 0.4250 +/- 0.1013 | 0.2393 +/- 0.0487 | 0.6903 +/- 0.0710 | 0.7819 +/- 0.0181 | 0.9555 +/- 0.0056  | 4.9 +/- 0.1    | 3.4 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     18 | oneclass | padim_pooled_resnet18           | padim     | native    |        29 |      0.4225 |           0.57 | 0.7449 +/- 0.0359 | 0.7082 +/- 0.0868 | 0.7173 +/- 0.0953 | 0.1856 +/- 0.0812 | 0.3264 +/- 0.0841 | 0.3292 +/- 0.0499 | 0.7086 +/- 0.0391 | 0.7628 +/- 0.0172 | 0.9264 +/- 0.0067  | 3.9 +/- 0.2    | 2.1 +/- 0.4   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |
|     19 | oneclass | padim_pooled_wide_resnet50_2    | padim     | native    |        29 |      0.4415 |           1.82 | 0.7107 +/- 0.0527 | 0.7264 +/- 0.1080 | 0.8157 +/- 0.1107 | 0.1982 +/- 0.1004 | 0.2428 +/- 0.1129 | 0.3651 +/- 0.0527 | 0.6791 +/- 0.0500 | 0.7317 +/- 0.0252 | 0.9132 +/- 0.0131  | 4.8 +/- 0.1    | 3.4 +/- 0.1   |              nan |              nan |                  nan |                   nan |              nan |           nan | 0.0 +/- 0.0          | True       |

### Mean AUROC by dataset family -- oneclass

| method                          |   kolektor |   mvtec |   severstal |   visa |
|:--------------------------------|-----------:|--------:|------------:|-------:|
| dfm_convnext_small              |     0.8394 |  0.9079 |      0.5776 | 0.8554 |
| dfm_dinov2_vitb14               |     0.9143 |  0.9671 |      0.7994 | 0.9028 |
| dfm_dinov2_vitl14               |     0.9127 |  0.9504 |      0.8048 | 0.8987 |
| dfm_efficientnet_b4             |     0.9173 |  0.9172 |      0.7265 | 0.8633 |
| dfm_resnet18                    |     0.9226 |  0.8578 |      0.6707 | 0.8766 |
| dfm_resnet50                    |     0.9176 |  0.8928 |      0.7124 | 0.8884 |
| dfm_wide_resnet50_2             |     0.9262 |  0.8965 |      0.7291 | 0.8882 |
| padim_pooled_resnet18           |     0.892  |  0.7755 |      0.4225 | 0.7213 |
| padim_pooled_wide_resnet50_2    |     0.895  |  0.7661 |      0.4415 | 0.6484 |
| padim_wide_resnet50_2           |     0.9544 |  0.8939 |      0.4415 | 0.7924 |
| patchcore_convnext_small        |     0.9135 |  0.9517 |      0.5416 | 0.8649 |
| patchcore_dense_wide_resnet50_2 |     0.947  |  0.9511 |      0.5567 | 0.8724 |
| patchcore_dinov2_vitb14         |     0.9521 |  0.9829 |      0.6965 | 0.866  |
| patchcore_dinov2_vitl14         |     0.9479 |  0.9837 |      0.6985 | 0.866  |
| patchcore_efficientnet_b4       |     0.9419 |  0.9647 |      0.564  | 0.8679 |
| patchcore_knn3_wide_resnet50_2  |     0.953  |  0.9527 |      0.5602 | 0.8765 |
| patchcore_resnet18              |     0.9394 |  0.9421 |      0.5264 | 0.8884 |
| patchcore_resnet50              |     0.945  |  0.9582 |      0.5573 | 0.8829 |
| patchcore_wide_resnet50_2       |     0.9513 |  0.9576 |      0.5686 | 0.8772 |

### Best method per configuration -- oneclass

| config           | method                          |   auroc |   average_precision |   f1_max |   ms_per_image |
|:-----------------|:--------------------------------|--------:|--------------------:|---------:|---------------:|
| kolektor         | padim_wide_resnet50_2           |  0.9544 |              0.8315 |   0.785  |            2.9 |
| mvtec/bottle     | dfm_dinov2_vitb14               |  1      |              1      |   1      |           17.4 |
| mvtec/cable      | patchcore_dinov2_vitl14         |  0.9969 |              0.9981 |   0.9856 |           90.2 |
| mvtec/capsule    | patchcore_dinov2_vitl14         |  0.9844 |              0.9961 |   0.9894 |           73   |
| mvtec/carpet     | patchcore_dinov2_vitb14         |  1      |              1      |   1      |           17.9 |
| mvtec/grid       | patchcore_dinov2_vitb14         |  1      |              1      |   1      |           19   |
| mvtec/hazelnut   | patchcore_dense_wide_resnet50_2 |  1      |              1      |   1      |            8.2 |
| mvtec/leather    | dfm_convnext_small              |  1      |              1      |   1      |            8.2 |
| mvtec/metal_nut  | patchcore_dinov2_vitb14         |  1      |              1      |   1      |           17.8 |
| mvtec/pill       | patchcore_dinov2_vitl14         |  0.9974 |              0.9995 |   0.9905 |           74   |
| mvtec/screw      | patchcore_wide_resnet50_2       |  0.9491 |              0.9737 |   0.951  |            4.5 |
| mvtec/tile       | dfm_dinov2_vitl14               |  1      |              1      |   1      |          108.2 |
| mvtec/toothbrush | dfm_dinov2_vitl14               |  0.9833 |              0.9944 |   0.9831 |          114.2 |
| mvtec/transistor | patchcore_dinov2_vitl14         |  0.9938 |              0.9905 |   0.9554 |           76.4 |
| mvtec/wood       | patchcore_convnext_small        |  0.9965 |              0.9989 |   0.9836 |           12   |
| mvtec/zipper     | patchcore_dinov2_vitl14         |  0.9992 |              0.9998 |   0.9944 |           70.7 |
| severstal        | dfm_dinov2_vitl14               |  0.8048 |              0.8373 |   0.8415 |           88   |
| visa/candle      | dfm_convnext_small              |  0.9693 |              0.9704 |   0.9238 |            3.6 |
| visa/capsules    | dfm_dinov2_vitb14               |  0.9118 |              0.9495 |   0.8675 |           15.3 |
| visa/cashew      | dfm_dinov2_vitl14               |  0.9826 |              0.9908 |   0.9557 |          109.4 |
| visa/chewinggum  | dfm_wide_resnet50_2             |  0.9952 |              0.9977 |   0.9796 |            3   |
| visa/fryum       | dfm_resnet18                    |  0.975  |              0.9861 |   0.9561 |            6.7 |
| visa/macaroni1   | patchcore_convnext_small        |  0.8567 |              0.8357 |   0.8045 |            3.6 |
| visa/macaroni2   | patchcore_resnet18              |  0.7625 |              0.7722 |   0.728  |            3.6 |
| visa/pcb1        | dfm_wide_resnet50_2             |  0.9498 |              0.9507 |   0.892  |            2.7 |
| visa/pcb2        | dfm_wide_resnet50_2             |  0.9663 |              0.966  |   0.9126 |            2.6 |
| visa/pcb3        | dfm_dinov2_vitl14               |  0.9208 |              0.9128 |   0.8604 |           82.1 |
| visa/pcb4        | patchcore_resnet50              |  0.9718 |              0.97   |   0.9219 |            9.5 |
| visa/pipe_fryum  | patchcore_convnext_small        |  0.9828 |              0.9917 |   0.9589 |            4.3 |

## Failed runs

Recorded rather than dropped: a model that cannot complete a dataset has told you something about its deployability.

| method                | config           | error                                                                                                                              |
|:----------------------|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| anomalib_anomaly_dino | mvtec/bottle     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/cable      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/capsule    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/carpet     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/grid       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/hazelnut   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/leather    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/metal_nut  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/pill       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/screw      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/tile       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/toothbrush | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/transistor | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/wood       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | mvtec/zipper     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/candle      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/capsules    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/cashew      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/chewinggum  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/fryum       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/macaroni1   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/macaroni2   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/pcb1        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/pcb2        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/pcb3        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/pcb4        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomaly_dino | visa/pipe_fryum  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/bottle     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/cable      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/capsule    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/carpet     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/grid       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/hazelnut   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/leather    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/metal_nut  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/pill       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/screw      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/tile       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/toothbrush | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/transistor | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/wood       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | mvtec/zipper     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/candle      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/capsules    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/cashew      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/chewinggum  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/fryum       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/macaroni1   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/macaroni2   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/pcb1        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/pcb2        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/pcb3        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/pcb4        | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_anomalyvfm   | visa/pipe_fryum  | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/bottle     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/cable      | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/capsule    | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/carpet     | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/grid       | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
| anomalib_cfa          | mvtec/hazelnut   | MisconfigurationException: Trainer was configured with `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list. |
