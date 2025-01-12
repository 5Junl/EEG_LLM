# Add near the top of run_class_finetuning.py where other imports are
from discriminator import Discriminator

# Add this function to load and prepare the discriminator
def setup_discriminator(device):
    # Initialize discriminator with same architecture 
    discriminator = Discriminator(embedding_dim=200)
    
    # Load saved weights
    checkpoint = torch.load('discriminator_checkpoints/checkpoint_epoch_8.pth', map_location='cpu')
    discriminator.load_state_dict(checkpoint['model'])
    
    # Move to device and freeze weights
    discriminator = discriminator.to(device)
    discriminator.eval()
    for param in discriminator.parameters():
        param.requires_grad = False
        
    return discriminator

# Modify the main training loop to add discriminator loss
def train_class_batch(model, samples, targets, criterion, input_chans, discriminator=None, disc_weight=0.1):
    """Modified training function that includes discriminator loss"""
    outputs = model(samples, input_chans=input_chans)
    base_loss = criterion(outputs, targets)
    
    # Add discriminator loss if provided
    if discriminator is not None:
        # Get embeddings from the encoder (need to modify based on your model architecture)
        with torch.no_grad():
            embeddings = model.get_embeddings(samples)  # You'll need to add this method to your model
            
        # Get discriminator predictions (trying to fool it to predict these as text)
        disc_outputs = discriminator(embeddings)
        disc_targets = torch.ones_like(disc_outputs)  # Want discriminator to think these are text
        disc_loss = nn.BCELoss()(disc_outputs, disc_targets)
        
        # Combine losses
        total_loss = base_loss + disc_weight * disc_loss
        return total_loss, outputs
        
    return base_loss, outputs

# In the main() function, before the training loop:
discriminator = setup_discriminator(device)

# Modify the training loop call:
train_one_epoch(
    model, criterion, data_loader_train,
    optimizer, device, epoch, loss_scaler,
    max_norm=args.clip_grad, model_ema=model_ema,
    mixup_fn=mixup_fn, log_writer=log_writer,
    start_steps=args.start_steps, lr_schedule_values=lr_schedule_values,
    wd_schedule_values=wd_schedule_values,
    num_training_steps_per_epoch=num_training_steps_per_epoch,
    update_freq=args.update_freq,
    ch_names=ch_names,
    is_binary=args.nb_classes == 1,
    discriminator=discriminator  # Add this line
)