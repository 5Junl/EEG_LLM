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

from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTokenizer, CLIPTextModel
import os

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
        

def load_checkpoint(model, checkpoint_path, model_key='model|module', model_prefix=''):
    from collections import OrderedDict
    
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Load checkpoint from {checkpoint_path}")
    
    # Find correct model key
    checkpoint_model = None
    for key in model_key.split('|'):
        if key in checkpoint:
            checkpoint_model = checkpoint[key]
            print(f"Load state_dict by model_key = {key}")
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        
    # Clean up student prefix if exists
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # Remove incompatible keys
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Remove position index keys
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)
    return model

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
    
def train_discriminator(eeg_encoder, discriminator, args):
    device = torch.device(args.device)
    # Initialize CLIP models
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_projection = nn.Linear(512, 200).to(device)


    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # Freeze EEG encoder
    eeg_encoder.eval()
    for param in eeg_encoder.parameters():
        param.requires_grad = False
    num_iters = 100  # 每个epoch的迭代次数
    best_acc = 0

    for epoch in tqdm(range(args.epochs), desc='Training Progress', position=0):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(range(num_iters), desc='Processing Batches', position=1, leave=False)


        for i in pbar:
            batch_size = args.batch_size
            
            # Generate real text embeddings using CLIP
            text_prompts = get_random_text(tokenizer, num_samples=batch_size)

            text_tokens = tokenizer(text_prompts, padding=True, truncation=True,return_tensors="pt").to(device)
            with torch.no_grad():
                text_features = text_encoder(**text_tokens).last_hidden_state[:, 0, :]  # [B, 512]
                # Project to match EEG embedding dimension
                real_embeddings = text_projection(text_features)  # [B, 200]

            
            with torch.no_grad():
                # Generate random EEG-like data: [B, channels, time]
                random_eeg = torch.randn(batch_size, 62, 1600).to(device)  # 1600 = 8 * 200
                # Reshape to match EEG encoder input
                random_eeg = rearrange(random_eeg, 'B N (A T) -> B N A T', T=200)
                # Get EEG features
                fake_embeddings = eeg_encoder.patch_embed(random_eeg)
                fake_embeddings = fake_embeddings.mean(dim=1)  # [B, 200]


            # Print shapes for debugging
            # print(f"Text embeddings shape: {real_embeddings.shape}")
            # print(f"EEG embeddings shape: {fake_embeddings.shape}")     

            # Labels (1 for text, 0 for EEG)
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
            correct += ((predictions > 0.9) == labels).sum().item()
            total += len(predictions)

            pbar.set_description(
                f'Epoch {epoch} - Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%'
            )
        epoch_loss = total_loss / num_iters
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')
                # Save latest checkpoint
        try:
            # Save latest checkpoint
            checkpoint = {
                'model': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, 
                f'discriminator_checkpoints/checkpoint_epoch_{epoch}.pth')
            
            # Save best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(checkpoint, 
                    'discriminator_checkpoints/checkpoint_best.pth')
        except Exception as e:
            print(f"Warning: Could not save checkpoint - {str(e)}")
            
            
    return discriminator

def main():
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    num_workers = 4
    epochs = 30
    
    # Load pretrained EEG encoder with all required parameters
    eeg_encoder = NeuralTransformer(
        patch_size=200,
        embed_dim=200, 
        depth=12,
        num_heads=10,
        mlp_ratio=4,
        num_classes=7,  # SEED dataset has 7 classes
        init_values=0.1,  # Add missing init_values
        qkv_bias=True,
        use_mean_pooling=True,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_scale=0.001,
    )
    
    eeg_encoder = load_checkpoint(
        eeg_encoder,
        './checkpoints/checkpoint-best.pth',
        model_key='model|module'
    )
    eeg_encoder = eeg_encoder.to(device)
    
    # Create discriminator and other components
    discriminator = Discriminator(embedding_dim=200).to(device)
    

    
    # Create args namespace for training
    class Args:
        def __init__(self):
            self.device = device
            self.epochs = epochs
            self.batch_size = batch_size
            
    args = Args()
    
    # Train discriminator
    discriminator = train_discriminator(eeg_encoder, discriminator, args)
    
    # Save discriminator
    # torch.save({
    #     'model': discriminator.state_dict(),
    # }, 'classifier.pth')


if __name__ == '__main__':
    os.makedirs('discriminator_checkpoints', exist_ok=True)

    main()