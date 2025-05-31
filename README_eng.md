# Extended Training Procedure with Loss Function Tournament and Intelligent Ensemble

This project extends the traditional training procedure with two main contributions:

1. ### Weak pretraining as a tournament between loss functions
2. ### Final voting through systematic ensemble of best models

## 1. Weak Pretraining with Loss Function Tournament
The pretraining procedure has been modified to test different loss functions in parallel. Each ModelTrainer is instantiated with a different loss among those selected (e.g., cross_entropy_loss, ncod_loss, etc.).
The idea is to treat each loss as a tournament participant, executing training cycles on the entire ABCD dataset. At the end of each training round, models are evaluated and the one with the best validation score (F1 score) is selected for each loss. Subsequently, the loss that achieved the best overall result in that round is selected.

Note: currently only the loss that performed best in the last round is selected, but it would be appropriate to calculate an average or statistic over all previous rounds. This has been left as a TODO.

## 2. Pre-trained Model Management
During training, if pre-trained models are available, they are automatically loaded. The worst performing model (based on F1 score) is selected to attempt improvement.
If a model with better performance is obtained, it replaces the worst among the pre-trained ones. This mechanism allows maintaining a collection of continuously improving models over time, while limiting the number of saved checkpoints. 

## Best Models Ensemble
Once training is completed, a selection of the best models is performed based on F1 score. Specifically:
1. associated with each model is loaded
2. Models are sorted by F1 score
3. The top-k models are selected (default: top_k = 5)
4. An ensemble is constructed using these models for the final voting procedure

This strategy overcomes the previous assumption that all "best" models (one per loss) are actually good. Now a relative ranking is adopted, systematically including only those that are truly more performant, improving the quality of voting.

## Conclusions
These modifications make the training procedure more robust and adaptive:

Loss functions compete with each other, dynamically selecting the most suitable one
Models are continuously updated and improved
Voting is based on an optimized ensemble, no longer on arbitrary or static choices

The system as a whole is designed to automatically and intelligently adapt to dataset complexity and variations in results between different training rounds.

## Teaser
![Teaser_DL_Noisy_Lables_2425.svg](Teaser_DL_Noisy_Lables_2425.svg)