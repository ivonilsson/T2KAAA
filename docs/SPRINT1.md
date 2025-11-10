# Sprint 1
This document details the achievables in the first sprint

## Motivation
Online retail generates high volumes of returns. The National Retail Federation estimates 19.3% of online sales will be returned this year (2025) [[NRF][nrf]]. Each return triggers extra transport, repackaging, and handling. Economically, high return rates provide little to no utility, incurring costs without creating value.

[nrf]: https://nrf.com/research/2025-retail-returns-landscape

## Pre-trained model / method
Use a pre-trained Virtual Try-On (VTON) diffusion model (most likely IDM-VTON on huggingface[🤗][huggingface]). Two inputs: image of person and image of garment to fit. Output: image of fitted garment on person. Evaluate with FID for realism, CLIP similarity for garment and text alignment, and VLM preference tests

[huggingface]: https://huggingface.co/yisol/IDM-VTON

## Dataset
If needed we are going to use the same datasets IDM-VTON was trained on namely VITON-HD and DressCode. While they are not open source, they are available to us.

### Checklist
- [x] Student group of 3 members created and communicated to Teachers
- [x] GitHub: repository created, team members set in .gitconfig and shared with Teachers 
- [x] GitHub: README.md added with project proposal description using proper formats e.g., chapter headings etc
- [x] README.md: Add project motiviation/introduction ( The real-world problem being tackled )
- [x] README.md: Add project pre-trained model/method (Existing AI/ML models or algortithms planned to be used and improved)
- [x] README.md: Add project dataset (dataset planned to be used, links/collection mechanisms)
