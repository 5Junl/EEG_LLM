import torch
import torch.nn as nn
from modeling_finetune import NeuralTransformer
import utils
from transformers import CLIPTokenizer, CLIPTextModel
from einops import rearrange
from tqdm import tqdm
import os
from pathlib import Path
from collections import OrderedDict
from utils import NativeScalerWithGradNormCount as NativeScaler
import random
import torch.nn.functional as F

def get_args():
    import argparse
    parser = argparse.ArgumentParser('CLIP alignment fine-tuning script')
    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./checkpoints/clip_aligned_losschanged')
    parser.add_argument('--finetune', default='./checkpoints/labram-base.pth')
    parser.add_argument('--model_key', default='model|module')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--resume', default='./checkpoints/clip_aligned/checkpoint_best.pth',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    parser.set_defaults(auto_resume=True)
    return parser.parse_args()

def setup_eeg_encoder(args):
    model = NeuralTransformer(
        embed_dim=200, 
        num_heads=10,
        mlp_ratio=4,
        init_values=0.1,
        num_classes=200,
        qkv_bias=True,
        use_mean_pooling=True,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_scale=0.001
    )

    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        print("debug 2", model, checkpoint_model)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    return model
def get_text_embeddings(labels, tokenizer, text_model, device):
    # Base emotion words
    emotion_words = {
        0: ['happy', 'joyful', 'cheerful', 'delightful', 'uplifting'],
        1: ['sad', 'sorrowful', 'melancholic', 'gloomy', 'depressing'],
        2: ['neutral', 'calm', 'balanced', 'ordinary', 'regular'],
        3: ['disgusting', 'repulsive', 'revolting', 'nauseating', 'unpleasant'],
        4: ['frightening', 'scary', 'terrifying', 'fearful', 'horror'],
        5: ['surprise', 'surprising', 'astonishing', 'amazing', 'unexpected', 'shocking'],
        6: ['angry', 'furious', 'enraging', 'irritating', 'infuriating']
    }

    # Templates for more natural descriptions
    templates = [
        "This is EEG data recorded while watching a {} movie",
        "These brain signals were measured during a {} film",
        "This brain activity was captured while viewing {} content",
        "The EEG recording was taken during a {} video clip",
        "This represents neural responses to {} movie scenes"
    ]
    
    # For each label, randomly select both template and emotion word
    texts = [
        random.choice(templates).format(random.choice(emotion_words[label.item()]))
        for label in labels
    ]
    
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_features = text_model(**inputs).last_hidden_state[:, 0, :]
    return text_features

def train_one_epoch(eeg_encoder, text_encoder_tuple, train_loader, optimizer, criterion, device, epoch):
    eeg_encoder.train()
    tokenizer, text_model,eeg_projection= text_encoder_tuple
    
    total_loss = 0
    pbar = tqdm(train_loader)
    
    for eeg_data, labels in pbar:
        # print("eeg_data:", eeg_data.shape)
        # Process EEG data
        eeg_data = eeg_data.float().to(device)
        eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=200)        
        # Get EEG embeddings
        eeg_embeddings = eeg_encoder(eeg_data, [0, 1, 2, 3, 7, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 75, 77, 78, 79, 85, 81, 82, 83, 86]) # 62 is the number of channels

        if isinstance(eeg_embeddings, tuple):
            eeg_embeddings = eeg_embeddings[0]
        eeg_embeddings = eeg_projection(eeg_embeddings)  # Project to 512d

        # Get text embeddings
        text_embeddings = get_text_embeddings(labels, tokenizer, text_model, device)
        # text_embeddings = text_projection(text_embeddings)
        
        # Calculate loss
        loss = criterion(eeg_embeddings, text_embeddings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f}')
        
    return total_loss / len(train_loader)

def clip_save_model(args, epoch, model, model_without_ddp, optimizer=None, loss_scaler=None, save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    print("Saving model, model's state dict:")
    print(model_without_ddp.state_dict())

    checkpoint_paths = [output_dir / 'checkpoint.pth']
    if epoch == 'best':
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
    elif (epoch + 1) % save_ckpt_freq == 0:
        checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            # 'scaler': loss_scaler.state_dict(),
            # 'args': args,
        }
        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()

        torch.save(to_save, checkpoint_path)
# Add new loss class
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, eeg_features, text_features):
        # Normalize features
        eeg_features = F.normalize(eeg_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(eeg_features, text_features.T) / self.temperature
        
        # Labels for contrastive learning (diagonal is positive pairs)
        labels = torch.arange(len(logits), device=logits.device)
        
        # Compute loss in both directions (eeg->text and text->eeg)
        loss_eeg = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        
        return (loss_eeg + loss_text) / 2
def main():
    args = get_args()
    device = torch.device(args.device)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup models
    eeg_encoder = setup_eeg_encoder(args).to(device)
    
    # Setup CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_model.eval()
    

    # Add projection layer
    # text_projection = nn.Sequential(
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 200)
    # ).to(device)    
    # Freeze text encoder and projection
    for param in text_model.parameters():
        param.requires_grad = False

    # Add projection layer on EEG side (200d -> 512d)
    eeg_projection = nn.Sequential(
        nn.Linear(200, 356),
        nn.ReLU(),
        nn.Linear(356, 512)
    ).to(device)    
    # Freeze projection layer
    for param in eeg_projection.parameters():
        param.requires_grad = False

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(
        eeg_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = CLIPLoss(temperature=0.07)
    
    # Load dataset
    train_dataset, _, _ = utils.prepare_SEEDV2_dataset(
        "./EEG_preprocessed/train_val_test"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            eeg_encoder,
            (tokenizer, text_model,eeg_projection),
            train_loader,
            optimizer,
            criterion,
            device,
            epoch
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'model': eeg_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # torch.save(
            #     checkpoint,
            #     os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            # )
            model_without_ddp = eeg_encoder
            
            clip_save_model(
                args=args,epoch="best", model=eeg_encoder,model_without_ddp= model_without_ddp)
        # Save best model
        if loss < best_loss:
            best_loss = loss
            checkpoint = {
                'model': eeg_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
            }
            # torch.save(
            #     checkpoint,
            #     os.path.join(args.output_dir, 'checkpoint_best.pth')
            # )
            model_without_ddp = eeg_encoder

            print("saving", model_without_ddp.state_dict())
            clip_save_model(
                args=args,epoch="best", model=eeg_encoder,model_without_ddp= model_without_ddp)
if __name__ == '__main__':
    main()