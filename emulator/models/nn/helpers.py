import numpy as np
import torch

def rebalance(train_data, train_labels):
    filtered_train_data = []
    filtered_train_labels = []
    filter_array = train_labels > 75
    np.concatenate(filtered_train_data, filtered_train_data[filter_array])
    for ex, label in zip(train_data, train_labels):
        if label[0] > 75:
            for i in range(1):
                filtered_train_data.append(ex)
                filtered_train_labels.append(label)

        # if label < 75:
        #     num = randint(0, 100)
        #     if num < 97:
        #         continue
        # elif label < 400:
        #     num = randint(0, 100)
        #     if num < 92:
        #         continue
        filtered_train_data.append(ex)
        filtered_train_labels.append(label)


    filtered_train_data = np.stack(filtered_train_data)
    filtered_train_labels = np.stack(filtered_train_labels)
    print(filtered_train_labels.shape)
    print(filtered_train_data.shape)

    print(f"reduced train labels from size {len(train_data)} to size {len(filtered_train_data)}")
    print(f"changed mean from {train_labels.mean()} to {filtered_train_labels.mean()}")
    return filtered_train_data, filtered_train_labels

def compute_grad_norm(model: torch.nn.Module):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm