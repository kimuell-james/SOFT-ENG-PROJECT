import pandas as pd
import numpy as np
import random

# Define the number of rows (1000 in this case)
num_rows = 1000

# Define possible values for categorical data
genders = ['Male', 'Female']
tracks = ['Academic', 'TVL']
academic_strands = ['ABM', 'HUMSS', 'STEM']
tvl_strands = ['IA-AS', 'IA-CES/EPAS', 'ICT', 'HE']

# Function to generate a random grade (between 75 and 100)
def random_grade():
    return round(random.uniform(75, 100), 2)

# Function to assign the strand based on the track
def assign_strand(track):
    if track == 'Academic':
        return random.choice(academic_strands)  # Only Academic strands for Academic track
    elif track == 'TVL':
        return random.choice(tvl_strands)  # Only TVL strands for TVL track

# Generate the track data first
tracks_data = np.random.choice(tracks, size=num_rows)

# Generate the data
data = {
    'age': np.random.randint(15, 19, size=num_rows),
    'gender': np.random.choice(genders, size=num_rows),
    'track': tracks_data,
    # Assign the correct strand based on the track using pre-generated tracks_data
    'strand': [assign_strand(track) for track in tracks_data],
    # Generate grades for G7, G8, G9, G10 Filipino, English, Math, Science, AP, TLE, MAPEH, ESP, and average
    'g7_filipino': [random_grade() for _ in range(num_rows)],
    'g7_english': [random_grade() for _ in range(num_rows)],
    'g7_math': [random_grade() for _ in range(num_rows)],
    'g7_science': [random_grade() for _ in range(num_rows)],
    'g7_ap': [random_grade() for _ in range(num_rows)],
    'g7_tle': [random_grade() for _ in range(num_rows)],
    'g7_mapeh': [random_grade() for _ in range(num_rows)],
    'g7_esp': [random_grade() for _ in range(num_rows)],
    'g7_average': [round(np.mean([random_grade() for _ in range(8)]), 2) for _ in range(num_rows)],
    'g8_filipino': [random_grade() for _ in range(num_rows)],
    'g8_english': [random_grade() for _ in range(num_rows)],
    'g8_math': [random_grade() for _ in range(num_rows)],
    'g8_science': [random_grade() for _ in range(num_rows)],
    'g8_ap': [random_grade() for _ in range(num_rows)],
    'g8_tle': [random_grade() for _ in range(num_rows)],
    'g8_mapeh': [random_grade() for _ in range(num_rows)],
    'g8_esp': [random_grade() for _ in range(num_rows)],
    'g8_average': [round(np.mean([random_grade() for _ in range(8)]), 2) for _ in range(num_rows)],
    'g9_filipino': [random_grade() for _ in range(num_rows)],
    'g9_english': [random_grade() for _ in range(num_rows)],
    'g9_math': [random_grade() for _ in range(num_rows)],
    'g9_science': [random_grade() for _ in range(num_rows)],
    'g9_ap': [random_grade() for _ in range(num_rows)],
    'g9_tle': [random_grade() for _ in range(num_rows)],
    'g9_mapeh': [random_grade() for _ in range(num_rows)],
    'g9_esp': [random_grade() for _ in range(num_rows)],
    'g9_average': [round(np.mean([random_grade() for _ in range(8)]), 2) for _ in range(num_rows)],
    'g10_filipino': [random_grade() for _ in range(num_rows)],
    'g10_english': [random_grade() for _ in range(num_rows)],
    'g10_math': [random_grade() for _ in range(num_rows)],
    'g10_science': [random_grade() for _ in range(num_rows)],
    'g10_ap': [random_grade() for _ in range(num_rows)],
    'g10_tle': [random_grade() for _ in range(num_rows)],
    'g10_mapeh': [random_grade() for _ in range(num_rows)],
    'g10_esp': [random_grade() for _ in range(num_rows)],
    'g10_average': [round(np.mean([random_grade() for _ in range(8)]), 2) for _ in range(num_rows)]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_student_data.csv', index=False)

print("CSV file has been generated and saved as 'synthetic_student_data.csv'")
