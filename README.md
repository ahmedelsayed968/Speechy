# Speechy

## Core Workflow
### Data Pipeline
![image](./assets/data-pip.png)
### Gender Detection Model
#### ECAPA Pretrained Model
- Finetuned on the VoxCeleb dataset, providing a robust starting point for proof-of-concept.
- Available on HuggingFace: [ECAPA Model](https://huggingface.co/JaesungHuh/voice-gender-classifier).
- My contribution: Simplified model integration into the pipeline for seamless use.

#### Classical Approach for Gender Detection
- TODO: Implement classical methods for voice gender detection.
- Reference: [Voice Gender Detection](https://github.com/jim-schwoebel/voice_gender_detection/tree/master).

#### Hybrid Approach
- TODO: Combine the ECAPA pretrained model and classical methods, then compare their performance.