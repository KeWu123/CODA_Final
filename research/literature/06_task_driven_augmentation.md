# Semi-Supervised Task-Driven Data Augmentation

- Venue/year: Medical Image Analysis 2021.
- Core idea: conditional generators learn additive intensity transformations and deformation fields optimized for segmentation, using labeled and unlabeled data.
- Datasets include prostate, cardiac, and pancreas.
- Relevance: prevents claiming that learned intensity/shape augmentation for prostate is new.
- Gap relative to ViSA-MT: learns average transformation generators for the task; it does not perform online hard-but-valid candidate selection in an EMA teacher/student step.
- Paper: https://www.sciencedirect.com/science/article/pii/S136184152030298X
- Code: https://github.com/krishnabits001/task_driven_data_augmentation
