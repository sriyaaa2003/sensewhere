# Literature notes for sensewhere

Short review written before any code. Every claim below is taken from the abstract or listing of the cited work; where I could not confirm a detail it is marked **unverified**.

## 1. Radio sensing and the domain-shift problem

- **Widar 3.0** (Zheng et al., MobiSys 2019, "Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi"). Estimates a body-coordinate velocity profile (BVP) from Wi-Fi channel measurements so that gesture kinetics no longer depend on where the person stands. Reported 92.7% in-domain and 82.6% to 92.4% cross-domain accuracy (environment, location and orientation together). The public data (258K instances, 75 domains) come with BVP features. *Relevance:* BVP is already a location-reduced representation, so the remaining cross-location gap is the hard residual; this is the dataset where "location" is an explicit, labelled domain factor.
- **SenseFi** (Yang et al., Patterns 2023). Library and benchmark of deep models (MLP, CNN, RNN, Transformer) on four Wi-Fi CSI datasets including UT-HAR, Widar and NTU-Fi. Finding in its summary: CNN and GRU backbones are efficient and strong. *Relevance:* justifies small CNN/GRU backbones as the baseline family rather than large Transformers. Per-domain Widar numbers were **unverified**.

## 2. Domain generalisation (no target data at training time)

- **DomainBed** (Gulrajani and Lopez-Paz, ICLR 2021). With careful tuning and a model-selection rule, plain empirical risk minimisation (ERM) was not beaten by more than about a point by the fourteen algorithms tested; an algorithm without a model-selection rule is "incomplete". *Consequence for this project:* ERM is the baseline to beat, model selection must not use target labels, and a negative result for fancy methods is a plausible, publishable outcome.
- **DIVERSIFY** (Lu et al., ICLR 2023). Time-series out-of-distribution representation learning that alternates between finding a worst-case latent distribution and aligning latent distributions; evaluated on gesture, speech, wearable and activity data. *Relevance:* a time-series-specific DG method to include as a comparison if budget allows.

## 3. Domain adaptation and source-free adaptation for time series

- **AdaTime** (Ragab et al., TKDD 2023). Benchmark of domain adaptation on time series with standardised backbones and datasets; stresses that using target labels for model selection breaks the unsupervised assumption and recommends source-risk and DEV-risk selection; found image-domain methods competitive once tuned. *Consequence:* adopt its protocol (fixed backbone, no target labels for selection).
- **SHOT** (Liang et al., ICML 2020). Source-free adaptation: freeze the source classifier, train the target feature extractor with information maximisation and self-training on pseudo-labels.
- **Tent** (Wang et al., ICLR 2021). Test-time adaptation by minimising prediction entropy, updating only normalisation statistics and channel-wise affine parameters; needs one pass, no source data.
- **MAPU** (Ragab et al., KDD 2023). Source-free adaptation for time series with a temporal imputation task in embedding space; extended by E-MAPU with evidential uncertainty (overconfident softmax predictions are the motivation).

*Consequence:* the source-free arm of the study compares (a) BatchNorm statistics re-estimation, (b) Tent, (c) SHOT, and (d) a MAPU-style temporal-imputation regulariser if time allows, all using only unlabelled target data and the source-trained model.

## 4. Modular multi-class design

- **Hierarchical classification** (Silla and Freitas, Data Mining and Knowledge Discovery 2011). Survey that unifies local and global hierarchical approaches and compares them with flat classifiers conceptually and empirically. The survey's own flat-versus-hierarchical numbers were **unverified** here.
- **Sparsely-gated mixture of experts** (Shazeer et al., ICLR 2017). A trainable gate selects a sparse subset of expert sub-networks per input (conditional computation).
- **Class-incremental learning** (surveys by Zhou et al. 2023 and Masana et al.). Directly training on new classes makes performance on old classes drop sharply; evaluation must state the memory budget because replay buffers make comparisons unfair. *Consequence:* the "add classes" experiment fixes a memory budget (no stored old-class samples in the main arm) and reports old-class accuracy, new-class accuracy and the drop.

## 5. Gap this project targets

Cross-location generalisation, source-free adaptation and modular class extension are studied separately, mostly on image data. Few studies put them in one protocol on the same radio-sensing data. The study here fixes one backbone family, one split protocol (leave-one-location-out, no target labels for selection), and asks three questions on the same data:

1. How large is the cross-location gap after BVP preprocessing, and which domain-generalisation method closes it (if any)?
2. How much of the remaining gap does source-free adaptation recover from unlabelled target data, and is it stable or does it collapse (entropy minimisation is known to be fragile)?
3. When classes are added, which design (flat softmax, hierarchical tree, mixture of experts) loses least on old classes and transfers best to a new location?

## Sources

- Widar 3.0: https://ieee-dataport.org/open-access/widar30-dataset-cross-domain-gesture-recognition-wi-fi
- SenseFi: https://arxiv.org/abs/2207.07859
- DomainBed: https://arxiv.org/abs/2007.01434
- DIVERSIFY: https://arxiv.org/abs/2209.07027
- AdaTime: https://arxiv.org/abs/2203.08321
- SHOT: https://arxiv.org/abs/2002.08546
- Tent: https://arxiv.org/abs/2006.10726
- MAPU: https://arxiv.org/abs/2307.07542
- Sparsely-gated MoE: https://arxiv.org/abs/1701.06538
- Hierarchical classification survey: https://doi.org/10.1007/s10618-010-0175-9
- Class-incremental learning survey: https://arxiv.org/abs/2302.03648
