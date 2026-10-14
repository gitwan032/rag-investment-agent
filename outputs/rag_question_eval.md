# 질문 방식 비교 (Agentic RAG, 판정 기준: 팀 원문)

| 분야 | 팀 질문: 통과 문서 | 재작성 | 웹 보완 | 분해 질문: 통과 문서 | 재작성 | 웹 보완 |
|---|---:|---:|:---:|---:|---:|:---:|
| 전력 수요 예측 | 1 | 2 | 예 | 4 | 2 | 예 |
| 발전 예측·VPP | 2 | 0 | 아니오 | 4 | 2 | 예 |
| 배터리(ESS) 운영 | 3 | 0 | 아니오 | 6 | 2 | 예 |
| 전력망 관리 | 4 | 0 | 아니오 | 4 | 2 | 예 |
| **합계** | **10** | | | **18** | | |

## 통과한 문서

- 전력 수요 예측 / team: doe_ai_energy_2024 p.17
- 전력 수요 예측 / decomposed: wef_ai_energy_2021 p.10, wef_ai_energy_2021 p.4, wef_ai_energy_2021 p.8, wef_ai_energy_2021 p.9
- 발전 예측·VPP / team: doe_ai_energy_2024 p.39, wef_ai_energy_2021 p.13
- 발전 예측·VPP / decomposed: doe_ai_energy_2024 p.28, doe_ai_energy_2024 p.39, wef_ai_energy_2021 p.13
- 배터리(ESS) 운영 / team: mdpi_bess_review_2026 p.20, mdpi_bess_review_2026 p.21, mdpi_bess_review_2026 p.44
- 배터리(ESS) 운영 / decomposed: mdpi_bess_review_2026 p.20, mdpi_bess_review_2026 p.21, mdpi_bess_review_2026 p.44, mdpi_bess_review_2026 p.47, mdpi_bess_review_2026 p.54, mdpi_bess_review_2026 p.66
- 전력망 관리 / team: wef_ai_energy_2021 p.12, wef_ai_energy_2021 p.18, wef_ai_energy_2021 p.19, wef_ai_energy_2021 p.4
- 전력망 관리 / decomposed: doe_ai_energy_2024 p.23, doe_ai_energy_2024 p.6, wef_ai_energy_2021 p.12, wef_ai_energy_2021 p.4
