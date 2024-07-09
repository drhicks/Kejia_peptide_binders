import pandas as pd
import sys

# Check if the filename is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <filename>")
    sys.exit(1)

# Read the data from the file specified in the first command-line argument
filename = sys.argv[1]
df = pd.read_csv(filename, delim_whitespace=True)

# Function to modify the 'description' to group by
def modify_description(desc):
    # Strip the last two characters if they are "_#" where # is a digit
    if desc[-2] == '_' and desc[-1].isdigit():
        return desc[:-2]
    return desc

# Apply the function to the 'description' column
df['description_base'] = df['description'].apply(modify_description)

# Group by the modified 'description' and calculate the mean for numeric columns
result = df.groupby('description_base').mean()

# Find the description with the highest iptm value within each group
best_description = df.loc[df.groupby('description_base')['iptm'].idxmax(), ['description_base', 'description']]
# best_description = best_description.rename(columns={'description': 'description_best'})

# Merge the best description back to the result
result = result.merge(best_description, on='description_base')

# Round the results to 3 decimal places
result = result.round(3)

# Reset index to make 'description_base' a regular column
result = result.reset_index()

# Print the headers first, separated by tabs
print('\t'.join(result.columns))

# Print the results in the desired format
for index, row in result.iterrows():
    print('\t'.join(map(str, row.values)))
