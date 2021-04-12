#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import torch


class ReviewDataset(Dataset):

    def __init__(self, df, tokenizer, mode):
        self.mode = mode
        self.reviews = df['review'].to_numpy()
        if self.mode == 'train':
            self.targets = df['polarity'].to_numpy()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
            )
        if self.mode == 'train':
            target = self.targets[item]
            return {
                'review_text': review,
                'input_words': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)
                }
        else:
            return {
                'review_text': review,
                'input_words': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                }
