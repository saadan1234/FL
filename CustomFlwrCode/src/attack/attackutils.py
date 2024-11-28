import torch
import tensorflow as tf
from torch import nn, optim
from sklearn.metrics import mean_squared_error

def deep_leakage_from_gradients(model, origin_grad, input_shape, num_classes):
    # Initialize dummy data and labels with required shapes
    dummy_data = torch.randn(input_shape, requires_grad=True)
    dummy_label = torch.randint(0, num_classes, (1,), dtype=torch.long, requires_grad=False)  # One label per input
    print(f"Dummy Data {type(dummy_data)}")
    print(f"Dummy Label {type(dummy_label)}")
    optimizer = torch.optim.LBFGS([dummy_data])
    criterion = torch.nn.CrossEntropyLoss()

    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            # Ensure `dummy_data` is a PyTorch tensor
            dummy_data_pt = (
                torch.from_numpy(dummy_data.numpy()) if isinstance(dummy_data, tf.Tensor) else dummy_data
            )

            # Ensure `dummy_label` is a PyTorch tensor
            dummy_label_pt = (
                torch.from_numpy(dummy_label.numpy()) if isinstance(dummy_label, tf.Tensor) else dummy_label
            )

            # Add batch dimension and detach
            dummy_data_pt = dummy_data_pt.unsqueeze(0).detach().requires_grad_()

            # Pass through the model (assuming the model is a TensorFlow model)
            dummy_pred_tf = model(dummy_data_pt.detach().numpy())

            # Convert TensorFlow tensor to PyTorch tensor
            dummy_pred_pt = torch.from_numpy(dummy_pred_tf).float()
            dummy_pred_pt.requires_grad_()
            dummy_label_pt = torch.from_numpy(dummy_label.numpy()).float()

            # Move to the correct device if necessary
            if torch.cuda.is_available():
                dummy_pred_pt = dummy_pred_pt.cuda()
                dummy_label_pt = dummy_label_pt.cuda()

            # Compute loss
            dummy_loss = criterion(dummy_pred_pt, dummy_label_pt)
            # Ensure dummy_grad and origin_grad are PyTorch tensors
            dummy_grad = [
                torch.from_numpy(g).float().to(dummy_data_pt.device) if isinstance(g, tf.Variable) else g
                for g in dummy_grad
            ]
            origin_grad = [
                torch.from_numpy(g).float().to(dummy_data_pt.device) if isinstance(g, tf.Variable) else g
                for g in origin_grad
            ]

            # Compute gradient difference
            grad_diff = sum(((dummy_g - origin_grad) ** 2).sum() for dummy_g, origin_grad in zip(dummy_grad, origin_grad))

            # Backpropagate
            grad_diff.backward()

            return grad_diff
        
        optimizer.step(closure)
        print(f"Optimizer Closure {type(optimizer.step(closure))}")

    print(f"Dummy Data Detached {type(dummy_data.detach())}")
    print(f"Dummy Data Detached Numpy {type(dummy_data.detach().numpy())}")
    print(f"Dummy Label Item{type(dummy_label.item())}")
    
    return dummy_data.numpy(), dummy_label.item()



def gradient_inversion_attack(gradients, model, input_shape):
    
    print("Performing Gradient Inversion Attack...")

    # Initialize dummy data
    dummy_data = torch.randn((1, *input_shape), requires_grad=True)

    optimizer = optim.Adam([dummy_data], lr=1e-3)

    for _ in range(1000):
        optimizer.zero_grad()
        outputs = model(dummy_data)
        loss = nn.CrossEntropyLoss()(outputs, model.true_labels)
        dummy_gradients = torch.autograd.grad(loss, model.trainable_variables, create_graph=True)

        # Match gradients
        gradient_loss = sum(
            ((dummy - real) ** 2).sum()
            for dummy, real in zip(dummy_gradients, gradients)
        )
        gradient_loss.backward()
        optimizer.step()

    print("Gradient Inversion Attack Completed")
    return dummy_data.detach()



def inverting_gradients_attack(gradients, model, input_shape):

    print("Performing Inverting Gradients Attack...")

    dummy_data = torch.randn((1, *input_shape), requires_grad=True)
    optimizer = optim.Adam([dummy_data], lr=1e-3)

    def tv_loss(x):
        return ((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2).mean() + \
               ((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2).mean()

    for _ in range(1000):
        optimizer.zero_grad()
        outputs = model(dummy_data)
        loss = nn.CrossEntropyLoss()(outputs, model.true_labels)
        dummy_gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gradient_loss = sum(
            ((dummy - real) ** 2).sum()
            for dummy, real in zip(dummy_gradients, gradients)
        )
        total_loss = gradient_loss + 1e-5 * tv_loss(dummy_data)
        total_loss.backward()
        optimizer.step()

    print("Inverting Gradients Attack Completed")
    return dummy_data.detach()



def multiple_updates_attack(model, gradients_series, input_shape, num_classes):
    import torch

    print("Performing Multiple Updates Attack...")

    dummy_data_series = [torch.randn((1, *input_shape), requires_grad=True) for _ in gradients_series]
    optimizer = torch.optim.Adam(dummy_data_series, lr=1e-3)

    for _ in range(1000):
        optimizer.zero_grad()
        total_loss = 0

        for t, (observed_gradients, dummy_data) in enumerate(zip(gradients_series, dummy_data_series)):
            outputs = model(dummy_data)
            loss = nn.CrossEntropyLoss()(outputs, model.true_labels)
            dummy_gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Match gradients at each time step
            gradient_loss = sum(
                ((dummy - real) ** 2).sum()
                for dummy, real in zip(dummy_gradients, observed_gradients)
            )
            total_loss += gradient_loss

        total_loss.backward()
        optimizer.step()

    print("Multiple Updates Attack Completed")
    return dummy_data_series

def evaluate_similarity(original_data, reconstructed_data, metrics=["mse", "cosine"], plot=True):
    """
    Evaluates the similarity between original and reconstructed data.
    """
    original_flat = original_data.reshape(original_data.shape[0], -1)
    reconstructed_flat = reconstructed_data.reshape(reconstructed_data.shape[0], -1)

    results = {}
    if "mse" in metrics:
        mse = mean_squared_error(original_flat, reconstructed_flat)
        results["mse"] = mse

    if "cosine" in metrics:
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(original_flat, reconstructed_flat).mean()
        results["cosine_similarity"] = cos_sim

    if plot:
        import matplotlib.pyplot as plt
        num_to_plot = min(len(original_data), 5)
        fig, axes = plt.subplots(2, num_to_plot, figsize=(15, 6))
        for i in range(num_to_plot):
            axes[0, i].imshow(original_data[i])
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed_data[i])
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()

    return results