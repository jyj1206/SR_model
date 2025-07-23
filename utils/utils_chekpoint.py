import torch


def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_path)
    
    
def load_checkpoint(model, optimizer, scheduler, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    return model, optimizer, scheduler, start_epoch, start_step