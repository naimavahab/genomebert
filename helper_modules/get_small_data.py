from datasets import load_dataset, Dataset,load_from_disk

# Load the dataset (example: 'imdb' dataset)
dataset = load_from_disk('input_data/DNA_256')
print(dataset)
# Select the first 5000 rows
subset = dataset.select(range(2000))

# Save the subset to a new dataset file

subset.save_to_disk('input_data_2k')

print(f"First 2000 lines have been written")

