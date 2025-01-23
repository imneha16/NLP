# NLP (Natural language processing)
## 1. Problem Introduction
This Kaggle competition involved the classification of text comments into seven predefined categories like toxicity, severe toxicity, obscene, threat, insult, identity attack and sexual explicit. The task was done without the use of transformers. Thus, the participants needed to create and train models with other NLP techniques that would handle the dataset efficiently.
## 2. Model Architecture
The architecture of the model was designed as a hybrid approach.
Embedding Layer. It changes the tokenized words into their dense vector representations.
BiLSTM Layer. A bidirectional LSTM network to model both forward and backward contextual dependencies in the text. 
GRU Layer. A Gated Recurrent Unit layer that reduces computational overhead while retaining sequence-learning capability. 
1D Convolutional Layer. It is for extracting spatial features, enhancing the model's understanding of sequential data patterns.
Fully Connected Layer. It combines features and produce predictions for each of the seven categories using a sigmoid activation function.
	A few design choices in the architecture were the following.
GRU after LSTM. To further refine learned sequence information with reduced model complexity. Dropout Regularization. After GRU and Convolutional layers, this is to prevent overfitting. 
## 3. Loss Function and Optimization
Loss Function. BCELoss, along with class weights was used for handling the imbalanced dataset. Class weights are calculated as the inverse of target class frequencies.
Optimizer. Adam optimizer was used with a learning rate of 0.0003 which is relatively efficient for gradient-based optimization.
Learning Rate Scheduler. It used StepLR that decays the learning rate by 50% every two epochs, allowing weight fine-tuning.
## 4. Data Preprocessing
Text Preprocessing. Special characters were removed, text was changed into lowercase and stop words were removed using NLTK. 
Tokenization: Tokenized the text into context-conserving words. 
Vocabulary Creation. A vocabulary mapping was created where each word was mapped to an index based on its frequency of appearance. 
Sequence Padding: Padding was used for truncating each sequence to a fixed length of 400 tokens. 
## 5. Experiments and Observations 
Baseline BiLSTM. Only the BiLSTM layer was used. It had poor performance concerning metrics like F1-score because of poor feature representation.
BiLSTM with GRU. The addition of the GRU layer increased accuracy and F1 scores, indicating an improvement in the refinement of sequential features.
Final Model. By adding a convolutional layer, it increased the quality of spatial feature extraction, improving the validation accuracy and generalization. This hybrid model balanced sequential and spatial learning quite well and performed better compared to previous experiments.
## 6. Challenges Faced
Class imbalance. The dataset was highly imbalanced with some categories underrepresented. 
Solution. By calculating and applying class weights. It used a sigmoid activation function to handle multi-label outputs.
Model Overfitting. This was observed in early epochs. 
Solution. Dropout layers and a learning rate scheduler was used.
Training Time. The BiLSTM-GRU-CNN model required quite a lot of computational resources. 
Solution. The batch size was reduced to 16 and training was conducted over 3 epochs to balance performance and resource constraints.
## 7. Results 
Best Validation Accuracy. The hybrid BiLSTM-GRU-CNN model gave the best validation accuracy after several iterations. 
Metrics. The validation F1 score, accuracy, and Hamming loss showed improved metrics compared to previous architectures. 
Submission File. A CSV file with predicted labels was submitted for Kaggle.

