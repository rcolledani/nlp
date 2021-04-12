import pandas as pd
import transformers 
from transformers import BertModel, BertTokenizer, AdamW
import torch
import torch.nn as nn
import numpy as np
import spacy
import ReviewDataset
import SentimentClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


class Classifier:
    """The Classifier"""

    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        df_train = pd.read_csv(trainfile, sep='\t', header=None)
        df_train.columns = ['polarity', 'aspect_category', 'target_term',
                            'character_offsets', 'review']
        df_train['polarity'] = df_train['polarity'].map(
            {'negative':0, 'neutral':1, 'positive':2}
            )
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        train_dataset = ReviewDataset.ReviewDataset(df_train,
                                                    tokenizer, mode='train')
        BATCH_SIZE = 16
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=BATCH_SIZE, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_name = ['positive', 'negative', 'neutral']
        model = SentimentClassifier.SentimentClassifier(
            len(class_name),
            PRE_TRAINED_MODEL_NAME).to(device)

        epochs = 3
        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(1, epochs+1):
            print('-----')
            print(f'Epoch {epoch}/{epochs}')
            print()

            # TRAINING##
            model.train()
            train_losses = []
            correct_predictions = 0
            n_examples_train = 0

            for i, batch, in enumerate(tqdm(train_dataloader,
                                            position=0, leave=True)):
                # for each batch in the training data loader
                targets = batch['targets'].to(device)
                input_words = batch['input_words'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # forward and backward backpropagation
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_words,
                    attention_mask=attention_mask
                    )
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == targets)
                n_examples_train += list(targets.shape)[0]

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss_mean = np.mean(train_losses)
            train_accuracy = correct_predictions.double() / n_examples_train

            print('Train loss: '+ str(round(train_loss_mean,2))+ ', accuracy: ' + str(round(train_accuracy.item(),2)))
            print()

        torch.save(model.state_dict(), 'model_state.bin')

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        df_to_predict = pd.read_csv(datafile, sep='\t', header=None)
        df_to_predict.columns = ['polarity', 'aspect_category',
                                 'target_term', 'character_offsets', 'review']
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        dataset_to_predict = ReviewDataset.ReviewDataset(
            df_to_predict, tokenizer, mode='test')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 1
        dataloader_to_predict = DataLoader(
            dataset_to_predict, batch_size=BATCH_SIZE, shuffle=False)

        class_name = ['positive', 'negative', 'neutral']
        model = SentimentClassifier.SentimentClassifier(
                len(class_name),
                PRE_TRAINED_MODEL_NAME).to(device)
        model.load_state_dict(torch.load('model_state.bin'))
        model.eval()

        list_pred = []
        for i, batch, in enumerate(tqdm(dataloader_to_predict,
                                        position=0, leave=True)):
            input_words = batch['input_words'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(
                    input_ids=input_words,
                    attention_mask=attention_mask
                    )
            pred = np.argmax(output.detach().numpy()[0])
            if pred == 0:
                list_pred.append('negative')
            elif pred == 1:
                list_pred.append('neutral')
            elif pred == 2:
                list_pred.append('positive')

        return list_pred
