import importlib.util
import os

# Path to the Python file containing the results dictionary
submission_file = "submission/submission_resnet.py"

# Dynamically load the module from the Python file
spec = importlib.util.spec_from_file_location("submission_results", submission_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Access the "data" dictionary
data = module.data

total_correct = 0
total_checked = 0

for query_filename, gallery_filenames in data.items():
    query_class = query_filename.split('_')[0]

    correct = sum(query_class in img_name for img_name in gallery_filenames)

    print(f"Query '{query_class}': {correct}/{len(gallery_filenames)} correct images")

    total_correct += correct
    total_checked += len(gallery_filenames)

# Global statistics
print("\nTotal correct images:", total_correct)
print("Total images checked:", total_checked)
print("Global accuracy:", round(total_correct / total_checked * 100, 2), "%")