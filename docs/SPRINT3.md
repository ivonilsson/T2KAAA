# Sprint 3


## Model Deployment and Inference Serving
We deploy the IDM-VTON try-on pipeline as a Gradio-driven inference service (app.py) that hosts catalog browsing, photo upload, and generation. Each request logs end-to-end stage timings, prompts, scores, and preview images to MLflow via the new tracking.py helper, giving us latency tracing plus visual QA artifacts. MLflow’s experiment UI now shows these metrics and screenshots per run, so we can monitor reliability and quality.


## Conclusion
Our modular try-on service plus MLflow proved the core idea, trying on garments virtually through IDM-VTON while logging every prompt, latency, and stylist score, works reliably. We would have liked for this to become a web page similar to a real deployment, using IaC and other deployment techniques for full deployment checklist and scalability, instead of a Gradio interface. But we were limited by the compute infrastructure available to us.



## Sprint 3 checklist
- [x] README.md: Project motiviation/introduction (what is the real-world problem being tackled. check this is still correct)
- [ ] README.md: Pre-trained model/method and the AI component: Clear description of the approach used by the selected pre-trained AI/ML models or algortithms, optimization goals, as well as overall system architecture. (check this is still correct)
- [ ] README.md: Experiment and dataset: Clear description and evidence of reproducible experiments conducted in the project. (check this is still correct)
- [x] README.md: Model Deployment and Inference Serving, add details of inference service and monitoring capabilities.
- [x] README.md: Conclusion add reflection on whether your ideas proved to be successful or not also discussing plausible reasons why the technique did or didn't work. Additionally, you can suggest most promising approach to continuing the work if you had time.
- [ ] GitHub: repository includes code for model building (training and evaluation code)
- [ ] GitHub: repository includes code for infrastructure provision/deploy/installation MLFlow e.g., Docker compose scripts
- [ ] GitHub: repository includes test scripts and saved trained model
- [x] GitHub: repository includes model deployment scripts
