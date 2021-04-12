Natural Language Processing - Assignment 2

Team members: Damien CHAMBON - Theo COSTES - Romain COLLEDANI - Antoine GALLIER

For this assignment, we decided to split the work as follows. First, we transformed the data and processed the review. Then, we worked on the model itself. Finally, we optimized our model by fine-tuning the different parameters.

Transforming the data:

In the dataset, we have the review, a target term and the aspect related to the target term we focus on. Since the whole review may cover many different aspects, we tried to only gather the words of the sentence that were related to the target term. To do so, we used the spacy package and focused on the dependency of each token. We created a set that contains the selected tokens by first adding the target term. Then, we looped multiple times over all tokens to add either words that depended on at least one of the words already present in the set, or add words whose the words already present in the set depended on. We stopped when either the full sentence was included in the set, or when the size of the set exceeded a certain threshold. We then fed the set into the model.

Using that technique did not work as expected. Indeed, we had sentences where the selected tokens had little to do with the target term, or where important words like ®ÿthe bestÿ¯ or ®ÿreally likedÿ¯ were not included. Thus, we decided to only work with the raw sentences (the results were much better), after tokenizing it and encoding them through a BERT Tokenizer, based on the BERT model which was trained on the English Wikipedia and the BookCorpus dataset. We thus are able to transform sentences into word identifiers, and get attention masks as well, that are used for the padding of the sequence which is required for the model. The length of the attention asks was determined based on our different trials.

Adding the aspect category in the model did not improve the performance of the model so we did not include it as well. We encoded the polarity as numbers instead of keeping the labels.


Creating the model:

Our model, which is a classic feedforward network, relies on a pretrained model, the same BERT model that we use for the tokenizer. We use that model since it has been trained on the English language on quite different data. After feeding the BERT model with the word identifiers and the attention masks, we get its output and pass it to linear layers and then a ReLu function. We also use some dropout layers to prevent overfitting on the training set compared to the dev set. The final layer of our neural network has 3 output neurons, one for each class of the polarity. The final prediction of the model is obtained by finding which of the three neurons has the largest score. The index of that neurons tells us which class of the polarity need to be assigned. Our model is created in a class contained in the SentimentClassifier.py.


Training and fine-tuning our model:

The training of our model goes as follows. First, we load the training set, which we transform into a PyTorch dataset. The dataset is created in a class contained in the ReviewDataset.py. Then, the dataset is put in a DataLoader object from the PyTorch class, where the data is shuffled. We chose the batch size depending on the results we got on the development set in our different trials. We pass each batch in the model, compute the loss using the cross-entropy and backpropagate the loss. We run that process for 3 epochs, which we determined to give the optimal performance. After the training is done, we save the model in an object, to be used when we want to make predictions on the dev and test set.

For the dev and test part, we load the model saved earlier and we transform the dev (or test) data through the ReviewDataset class. We apply the same transformation on the reviews, i.e. we transform the words of the sentences into word identifiers and we create attention masks. We then pass those information in the model and transform the output into a prediction of the polarity.

We fine-tuned different parameters such as the depth and width of the layers after the BERT pretrained model, as well as the length of the attention masks. We also fine-tuned the optimizer and the learning rate, as well as the number of epochs. We did the training on Google Collaboratory so that we could use a GPU and make the training faster.

On the dev dataset, we get an accuracy of 81.4 %.
