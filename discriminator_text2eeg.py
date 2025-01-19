import torch
import torch.nn as nn
from modeling_finetune import NeuralTransformer
import utils
import random
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import random
from einops import rearrange
from tqdm import tqdm
import os
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTextModel

def get_random_text(tokenizer, num_samples=1):
    try:
        # 尝试从本地加载数据集
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="./datasets")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 降级使用基础模板
        templates = [
            "This is a sentence about various topics and subjects.",
            "A random piece of text that contains information.",
            "Some content from a typical document or article.",
            "An interesting discussion about different matters.",
            "The text describes multiple aspects of the topic."
        ]
        selected_texts = [random.choice(templates) for _ in range(num_samples)]
        return selected_texts if num_samples > 1 else selected_texts[0]

    try:
        # 过滤出有效句子
        valid_texts = [
            text for text in dataset['text'] 
            if len(text.strip()) > 20 and len(text.split()) < 50 
            and not text.startswith('=')
        ]
        
        # 随机选择句子
        selected_texts = random.sample(valid_texts, num_samples)
        
        # 确保句子长度适合CLIP
        processed_texts = []
        for text in selected_texts:
            tokens = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=77,
                return_tensors="pt"
            )
            processed_texts.append(text)
            
        # print("Processed text: ", processed_texts[0])
        return processed_texts if num_samples > 1 else processed_texts[0]
        
    except Exception as e:
        print(f"Error processing texts: {e}")
        return ["Default text sample"] * num_samples if num_samples > 1 else "Default text sample"
def get_models(args):
    """Initialize and load EEG encoder model"""
    # Initialize EEG encoder
    eeg_encoder = NeuralTransformer(
        patch_size=200,
        embed_dim=200, 
        depth=12,
        num_heads=10,
        mlp_ratio=4,
        num_classes=7,
        init_values=0.1,
        qkv_bias=True,
        use_mean_pooling=True,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_scale=0.001,
    )

    # Load pretrained weights if specified
    if args.finetune:
        try:
            # First try loading with state dict only
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print(f"Load pretrained weights from {args.finetune}")
            
            # Get model weights
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print(f"Load state_dict by model_key = {model_key}")
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint

            # Clean up state dict
            for k in list(checkpoint_model.keys()):
                if any(n in k for n in ['norm', 'relative_position_index']):
                    checkpoint_model.pop(k)
                    print(f"Removing key {k}")

            # Load cleaned state dict
            msg = eeg_encoder.load_state_dict(checkpoint_model, strict=False)
            print(f"Loading message: {msg}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing without pretrained weights")

    return eeg_encoder


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
    # Initialize CLIP models
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_projection = nn.Sequential(
        nn.Linear(512, 384),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.2),
        nn.Linear(384, 256),
        nn.LeakyReLU(0.2), 
        nn.Dropout(0.2),
        nn.Linear(256, 200),
        nn.LayerNorm(200)
    ).to(device)
    # Freeze text projection
    text_projection.eval()
    for param in text_projection.parameters():
        param.requires_grad = False
    print("Text projection frozen")
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Initialize best tracking variables
    best_loss = float('inf')
    best_accuracy = 0
    best_epoch = 0

    # Freeze EEG encoder
    eeg_encoder.eval()
    for param in eeg_encoder.parameters():
        param.requires_grad = False

    for epoch in tqdm(range(args.epochs), desc='Training Progress', position=0):
        total_loss = 0
        correct = 0
        total = 0
        
        # Use train_loader directly instead of fixed iterations
        pbar = tqdm(train_loader, desc='Processing Batches', position=1, leave=False)

        for eeg_data, _ in pbar:
            batch_size = eeg_data.shape[0]
            
            # Get real (EEG) embeddings
            with torch.no_grad():
                eeg_data = eeg_data.float().to(device)
                eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=200)
                real_embeddings = eeg_encoder.patch_embed(eeg_data)
                real_embeddings = real_embeddings.mean(dim=1)  # [B, 200]

            # Generate fake (text) embeddings using CLIP
            text_prompts = get_random_text(tokenizer, num_samples=batch_size)
            text_tokens = tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_features = text_encoder(**text_tokens).last_hidden_state[:, 0, :]
                fake_embeddings = text_projection(text_features)
            
            # Labels (1 for EEG, 0 for text)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train discriminator
            discriminator.zero_grad()
            
            real_output = discriminator(real_embeddings)
            real_loss = criterion(real_output, real_labels)
            
            fake_output = discriminator(fake_embeddings)
            fake_loss = criterion(fake_output, fake_labels)
            
            loss = real_loss + fake_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.cat([real_output, fake_output])
            labels = torch.cat([real_labels, fake_labels])
            correct += ((predictions > 0.5) == labels).sum().item()
            total += len(predictions)

            pbar.set_description(
                f'Epoch {epoch} - Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%'
            )
        
        # Calculate epoch metrics
        epoch_loss = total_loss/len(train_loader)
        epoch_accuracy = 100 * correct/total
        
        print(f'Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%')
        
        # Save periodic checkpoint
        if epoch % 2 == 0:
            torch.save({
                'model': discriminator.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }, f'discriminator_checkpoints/checkpoint_epoch_{epoch}.pth')
        
        # Update best model if current performance is better
        is_best = False
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_loss = epoch_loss
            best_epoch = epoch
            is_best = True
        elif epoch_accuracy == best_accuracy and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            is_best = True
            
        if is_best:
            print(f'New best model found at epoch {epoch}!')
            torch.save({
                'model': discriminator.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
                'accuracy': best_accuracy,
            }, 'discriminator_checkpoints/checkpoint_best.pth')
            
    print(f'Training completed. Best model was found at epoch {best_epoch} with accuracy: {best_accuracy:.2f}% and loss: {best_loss:.4f}')
            
    return discriminator

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup training args
    class Args:
        def __init__(self):
            self.output_dir = './checkpoints'
            self.finetune = './checkpoints/labram-base.pth'
            self.resume = './checkpoints/checkpoint-best.pth'
            self.model_key = 'model|module'
            self.model_prefix = ''
            self.device = device
            self.epochs = 30
            self.batch_size = 128
            self.auto_resume = False

    args = Args()
    
    # Get model
    eeg_encoder = get_models(args)
    eeg_encoder = eeg_encoder.to(device)

    # After base model is loaded, try loading fine-tuning weights
    if os.path.exists('./checkpoints/checkpoint-best.pth'):
        args.resume = './checkpoints/checkpoint-best.pth'
        try:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'model' in checkpoint:
                msg = eeg_encoder.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded fine-tuning checkpoint: {msg}")
        except Exception as e:
            print(f"Error loading fine-tuning checkpoint: {e}")

    

    eeg_encoder.eval()  # Set to eval mode
    for param in eeg_encoder.parameters():
        param.requires_grad = False
    # Create discriminator and other components
    discriminator = Discriminator(embedding_dim=200).to(device)
    
    # Get SEED dataset - only need train dataset
    train_dataset, val_dataset, test_dataset = utils.prepare_SEEDV2_dataset(
        "./EEG_preprocessed/train_val_test"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    # Train discriminator
    discriminator = train_discriminator(eeg_encoder, discriminator, train_loader, args)
    
    # Save discriminator
    torch.save({
        'model': discriminator.state_dict(),
    }, 'classifier.pth')


if __name__ == '__main__':
    main()