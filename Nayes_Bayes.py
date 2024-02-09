import csv
import pandas as pd


def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# the training_data.csv is now read
training_data = readCSVFile('training_data.csv')

# observe that the header (column names) are present, and so are the record numbers on each line
# print(training_data)


def list_to_dataframe(data_list):
    # Extract the column headers (the first inner list)
    columns = data_list[0]
    # Extract the data rows (all inner lists after the first)
    data_rows = data_list[1:]
    # Create and return a DataFrame using the headers and rows
    return pd.DataFrame(data_rows, columns=columns)


def calculate_conditional_probabilities(df):
    # Ensure 'Class' column exists in the DataFrame
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must contain a 'Class' column")

    # Calculate the counts for each class in the 'Class' column
    class_counts = df['Class'].value_counts()

    # Initialize a dictionary to hold the conditional probabilities
    conditional_probabilities = {col: {}
                                 for col in df.columns if col != 'Class'}

    # Calculate conditional probabilities
    for class_value in class_counts.index:
        class_subset = df[df['Class'] == class_value]
        for col in df.columns:
            if col != 'Class':
                value_counts = class_subset[col].value_counts()
                conditional_probabilities[col][class_value] = (
                    value_counts / class_counts[class_value]).to_dict()

    return conditional_probabilities


# Usage example:
df = pd.read_csv("training_data.csv")
df = df.drop("ID", axis=1)
cond_probs = calculate_conditional_probabilities(df)


def naive_bayes_classifier(cond_probs, input_list):
    # Convert input list to a dictionary with column names
    column_names = [col for col in cond_probs.keys()] + ['Class']
    input_dict = {col: val for col, val in zip(
        column_names, input_list) if val is not None}

    # Identify the missing field
    missing_field = [
        col for col in column_names if col not in input_dict.keys()][0]

    # Initialize probabilities for each class
    class_probabilities = {
        class_value: 1 for class_value in cond_probs[next(iter(cond_probs))].keys()}

    # Calculate the probability for each class
    for class_value in class_probabilities.keys():
        for field, value in input_dict.items():
            field_prob = cond_probs[field][class_value].get(value, 0)
            class_probabilities[class_value] *= field_prob

    # Normalize the probabilities
    total_prob = sum(class_probabilities.values())
    class_probabilities = {
        class_value: prob / total_prob for class_value, prob in class_probabilities.items()}

    # Find the class with the highest probability for the missing field
    predicted_value = max(class_probabilities, key=class_probabilities.get)

    return missing_field, predicted_value


# Example usage
input_list = ['Yes', 'No', 'Italian', None]  # Assuming 'Class' is missing
missing_field, prediction = naive_bayes_classifier(cond_probs, input_list)
print(f"The missing field '{missing_field}' is predicted to be '{prediction}'")
