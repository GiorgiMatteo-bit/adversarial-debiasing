"""
Data loading utilities for FINDHR dataset.
Dataset is under NDA - this provides the expected interface.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Optional


class ResumeDataset(Dataset):
    """
    Dataset for resume classification with protected attributes.
    
    Expected data format (CSV):
        - id: unique identifier (int or str)
        - full_text_english: resume text after preprocessing (str)
        - gender: binary gender label (0=female, 1=male)
        - ict_label: binary ICT classification (0=non-ICT, 1=ICT)
    
    Example row:
        id,full_text_english,gender,ict_label
        1,"Software engineer with 5 years experience...",1,1
        2,"Administrative assistant role...",0,0
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        """
        Args:
            data_path: path to CSV file with resume data
            tokenizer: ModernBERT tokenizer
            max_length: maximum sequence length (default: 256)
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        required_cols = ['id', 'full_text_english', 'gender', 'ict_label']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter out invalid gender labels (non-binary, no answer)
        valid_genders = [0, 1]
        self.data = self.data[self.data['gender'].isin(valid_genders)].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            job_label: int (0/1)
            gender_label: int (0/1)
        """
        row = self.data.iloc[idx]
        
        # Tokenize resume text
        encoding = self.tokenizer(
            row['full_text_english'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'job_label': torch.tensor(row['ict_label'], dtype=torch.long),
            'gender_label': torch.tensor(row['gender'], dtype=torch.long)
        }


def create_dataloaders(
    train_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0
):
    """
    Create train and test dataloaders.
    
    Args:
        train_path: path to training CSV
        test_path: path to test CSV
        tokenizer: ModernBERT tokenizer
        batch_size: batch size for training
        max_length: maximum sequence length
        num_workers: number of dataloader workers
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    train_dataset = ResumeDataset(train_path, tokenizer, max_length)
    test_dataset = ResumeDataset(test_path, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def calculate_class_weights(data_path: str):
    """
    Calculate class weights for imbalanced dataset.
    Uses inverse frequency weighting: total_samples / (num_classes × class_count)
    
    Args:
        data_path: path to training CSV
        
    Returns:
        job_weights: [weight_non_ict, weight_ict]
        gender_weights: [weight_female, weight_male]
    """
    df = pd.read_csv(data_path)
    
    # Job class weights
    job_counts = df['ict_label'].value_counts().sort_index()
    total = len(df)
    job_weights = torch.tensor([
        total / (2 * job_counts[0]),
        total / (2 * job_counts[1])
    ], dtype=torch.float32)
    
    # Gender class weights
    gender_counts = df['gender'].value_counts().sort_index()
    gender_weights = torch.tensor([
        total / (2 * gender_counts[0]),
        total / (2 * gender_counts[1])
    ], dtype=torch.float32)
    
    return job_weights, gender_weights
