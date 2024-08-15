import sys

def parse_range_segment(segment):
    """Parses a single segment and returns the number of combinations possible for that segment."""
    if segment.startswith('A'):
        return 1  # Any segment preceded by 'A' counts as 1 combination
    elif '-' in segment:
        start, end = map(int, segment.split('-'))
        return end - start + 1  # Number of possible values in the range
    else:
        return 1  # Single numbers count as 1 combination

def calculate_combinations(sequence):
    """Calculates the total number of combinations given the sequence string."""
    segments = sequence.split(',')
    total_combinations = 1
    
    for segment in segments:
        total_combinations *= parse_range_segment(segment)
    
    return total_combinations

# Example input sequence
sequence = sys.argv[1]

# Calculate and print the total number of combinations
total_combinations = calculate_combinations(sequence)
print(f"Total combinations: {total_combinations}")

