from datasets import load_dataset, DatasetDict

def load_and_save_cifar10_subset():
    # Load CIFAR-10 dataset
    dataset = load_dataset('uoft-cs/cifar10', split='train')
    
    # Select the first 100 rows
    subset = dataset.select(range(100))
    
    # Save the subset locally
    subset.save_to_disk('cifar10_subset')

    # Push the subset to Hugging Face
    subset.push_to_hub('wasifis/cifar-10-100',token='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')

if __name__ == "__main__":
    load_and_save_cifar10_subset()