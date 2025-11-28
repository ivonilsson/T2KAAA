# Sprint 1
This document details the achievables in the first sprint

## Motivation
Online retail generates high volumes of returns. The National Retail Federation estimates 19.3% of online sales will be returned this year (2025) [[NRF][nrf]]. Each return triggers extra transport, repackaging, and handling. Economically, high return rates provide little to no utility, incurring costs without creating value. Retailers lose profits margins and customers experience inconvenience and frustration. Furthermore, repeated shipping and repackaging contribute to higher co2 emotions and waste products, increasing the industry's environmental footprint. 

One key factor in returns is misleading or inaccurate visualisation of the item being sold. Product images are often displayed by professional models that may not represent the customer's body type or proportions, leading to a mismatch in expectations and a higher chance of return. 

To address this, we can utilize the fast evolving technology in image generation and processing. By using pretrained models for virtual fitting of clothes powered by AI, a customer can “try on” clothes on a picture of themselves. Getting a more fair visualisation and aligning expectations before he/she completes the purchas. thereby reducing returns and addressing the problem. This also allows the customer to get a fair view of the product, without any misleading models. This technology allows for new ways of online shipping that are more fun and fair. Customers get the chance to “try on” multiple clothes and to be more creative than what traditional online shopping allows for. Encouraging experimentation, transparency, more sustainable and enjoyable shopping for consumers and retailers.

This is not a silver bullet for the challenge at hand, but it is a step in the right direction to minimize returns and reduce the industries environmental footprint.

[nrf]: https://nrf.com/research/2025-retail-returns-landscape

## Pre-trained model / method
### Objective
Given:
- a person image P
- a garment image G
- optional text description T

Generate:
- a try-on image Y = f(P, G, T) where the output should preserve the person identity/pose and garment details.

### Model choice
We plan to use IDM-VTON (ECCV 2024), a diffusion-based virtual try-on model. We chose it because it is a recent research model specifically targeting garment fidelity and realistic try-on “in the wild”, and because official code/checkpoints are available for reproducible experimentation.

## Dataset
If needed, we will use datasets commonly used in VTON research, such as VITON-HD and DressCode. These datasets provide paired person/garment images and annotations that support evaluation of try-on realism and garment fidelity. While our project does not really require re-training or fine-tuning any models since IDM-VTON is developed for our specific purpose, we will most likely not use any of this data for that purpose.

### Checklist
- [x] Student group of 3 members created and communicated to Teachers
- [x] GitHub: repository created, team members set in .gitconfig and shared with Teachers 
- [x] GitHub: README.md added with project proposal description using proper formats e.g., chapter headings etc
- [x] README.md: Add project motiviation/introduction ( The real-world problem being tackled )
- [x] README.md: Add project pre-trained model/method (Existing AI/ML models or algortithms planned to be used and improved)
- [x] README.md: Add project dataset (dataset planned to be used, links/collection mechanisms)
