import torch
import torch.nn as nn
from modeling_finetune import NeuralTransformer
import utils
import random
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, embedding_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_discriminator(eeg_encoder, discriminator, train_loader, args):
    device = torch.device(args.device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # Freeze EEG encoder
    eeg_encoder.eval()
    for param in eeg_encoder.parameters():
        param.requires_grad = False
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (eeg_data, _) in enumerate(train_loader):
            batch_size = eeg_data.shape[0]
            
            # Get real EEG embeddings
            with torch.no_grad():
                eeg_data = eeg_data.float().to(device) / 100
                eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=200)
                real_embeddings = eeg_encoder.forward_features(eeg_data)
            
            # Generate fake text embeddings
            fake_embeddings = torch.randn_like(real_embeddings).to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train with real
            discriminator.zero_grad()
            real_output = discriminator(real_embeddings)
            real_loss = criterion(real_output, real_labels)
            
            # Train with fake
            fake_output = discriminator(fake_embeddings)
            fake_loss = criterion(fake_output, fake_labels)
            
            # Total loss
            loss = real_loss + fake_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.cat([real_output, fake_output])
            labels = torch.cat([real_labels, fake_labels])
            correct += ((predictions > 0.5) == labels).sum().item()
            total += len(predictions)
            
        print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}, Accuracy = {100*correct/total:.2f}%')
    
    return discriminator

def main():
    # Get args from run_class_finetuning.py
    args, _ = get_args()
    
    # Load pretrained EEG encoder
    eeg_encoder = NeuralTransformer(
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )
    checkpoint = torch.load('pretrain.pth', map_location='cpu')
    eeg_encoder.load_state_dict(checkpoint['model'])
    eeg_encoder = eeg_encoder.to(args.device)
    
    # Create discriminator
    discriminator = Discriminator().to(args.device)
    
    # Get dataset and dataloader
    dataset_train, _, _, _, _ = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Train discriminator
    discriminator = train_discriminator(eeg_encoder, discriminator, train_loader, args)
    
    # Save discriminator
    torch.save({
        'model': discriminator.state_dict(),
    }, 'classifier.pth')

if __name__ == '__main__':
    main()