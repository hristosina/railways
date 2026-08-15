"""UI-independent calculations for model-training progress."""


def calculate_training_progress(epoch_index, batch_number, total_batches, total_epochs):
    total_batches = max(1, int(total_batches))
    total_epochs = max(1, int(total_epochs))
    epoch_number = min(total_epochs, max(1, int(epoch_index) + 1))
    batch_number = min(total_batches, max(0, int(batch_number)))
    epoch_percent = round(batch_number / total_batches * 100)
    overall_fraction = ((epoch_number - 1) + batch_number / total_batches) / total_epochs
    overall_percent = round(min(1.0, max(0.0, overall_fraction)) * 100)
    return epoch_number, epoch_percent, overall_percent
