| adv | xxl-eval-acc | xxl-dev32-acc | xl-eval-acc | xl-dev32-acc | xxl-all-acc |
| :-: | :-: | :-: | :-: | :-: | :-: |
| BoolQ | 77.7+-1.5 | 71.9+-0.6 | 65.9+-0.5 | 66.7+-4.8 | 83.0+-0.9 |
| BoolQ + calibrate | 77.4+-1.1 | 71.9+-0.6 | 66.1+-1.0 | 67.7+-4.8 | 82.6+-1.0 |
| RandomPerm(3+0) | 77.1+-2.2 | 65.6+-3.1 | - | - | - |
| RandomPerm(3+3) | 67.6+-5.8 | 65.6+-3.1 | 62.6+-3.3 | 60.4+-11.8| 77.5+-2.2 |
| HotFlip(train) | 73.2+-4.0 | 65.6+-0.0 | - | - | - |
| HotFlip(eval) | 70.4+-5.9 | 69.7+-6.5 | 60.2+-3.1| 60.9+-2.2 | 78.0+-1.1 |
| InputSpecific(10) | 35.9 | 37.5 | 54.7 | 46.9 | 57.8 |

| adv | eval-acc | dev32-acc | eval-acc-adv | dev32-acc-adv|
| :-: | :-: | :-: | :-: | :-: |
| RandomPerm-xxl | 77.8 | 65.7 | 72.4 | 63.3 |
| Hotflip-xxl | 75.7 | 73.2 | 75.8 | 69.3 |
| RandomPerm-xl | 63.1 | 50.0 | 62.1 | 40.6 |
| HotFlip-xl | 62.7 | 59.4 | 62.9 | 59.4 |


| Triggers | negative accuracy | Triggers | positive accuracy |
| :-: | :-: | :-: | :-: |
| No triggers | 88.79 | No triggers | 85.36 |
| the, the, the | 85.75 | the, the, the | 83.78 |
| well-acted, vibrantly, enjoyable | 10.51 | useless, poorly-constructed, sabotaged | 12.39 |

| adv ratio | eval-acc | eval-acc-adv
| :-: | :-: | :-: |
| 0.0 | 76.95 | 71.48 |
| 0.2 | 77.15 | 74.22 |
| 0.4 | 76.17 | 72.85 |
| 0.6 | 68.36 | 64.84 |
| 0.8 | 76.17 | 70.90 |
| 1.0 | 73.82 | 71.29 |

| prefix length | eval-acc-adv | dev32-acc-adv |
| :-: | :-: | :-: |
| 1 | 73.76+-1.71 | 71.88+-8.27 |
| 2 | 70.57+-5.96 | 69.79+-11.83 |
| 3 | 70.44+-5.88 | 66.66+-4.77 |
| 4 | 72.07+-2.11 | 75.00+-5.41 |