"""Learning rate scheduler factory with safe, well-tested schedulers."""
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, scheduler_type, scheduler_params=None):
    """
    Factory for learning rate schedulers.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'none', 'cosine', or 'plateau'
        scheduler_params: Dict of scheduler-specific parameters
    
    Returns:
        Scheduler object or None
    
    Example:
        >>> opt = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> sched = get_scheduler(opt, 'cosine', {'T_0': 10})
        >>> # In training loop:
        >>> step_scheduler(sched)
    """
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    params = scheduler_params or {}
    
    if scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=params.get('T_0', 10),
            T_mult=params.get('T_mult', 2),
            eta_min=params.get('eta_min', 1e-7)
        )
    elif scheduler_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params.get('patience', 5),
            factor=params.get('factor', 0.5),
            min_lr=params.get('min_lr', 1e-7)
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}. Choose 'none', 'cosine', or 'plateau'.")


def step_scheduler(scheduler, val_loss=None):
    """Step scheduler with appropriate arguments."""
    if scheduler is None:
        return
    
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        if val_loss is None:
            raise ValueError("val_loss required for ReduceLROnPlateau")
        scheduler.step(val_loss)
    else:
        scheduler.step()

