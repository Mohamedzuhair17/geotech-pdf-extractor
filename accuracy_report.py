import json
import pandas as pd

# Load JSON
with open('output/extracted_results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Calculate accuracy metrics
total_rows = len(df)
high_conf = len(df[df['confidence'] >= 0.8])
above_threshold = len(df[df['confidence'] >= 0.6])
below_threshold = len(df[df['confidence'] < 0.6])

# Data completeness by field
fields_to_check = ['pct_fines', 'pct_sand', 'pct_gravel', 'll', 'pl', 'pi', 'ls', 'soil_description']
field_completeness = {}
for field in fields_to_check:
    complete = len(df[df[field] != '-'])
    pct = (complete / total_rows) * 100
    field_completeness[field] = (complete, pct)

# Extraction method breakdown
method_counts = df['extraction_method'].value_counts()

print(f'EXTRACTION ACCURACY SUMMARY (n={total_rows})')
print('=' * 60)
print(f'\nCONFIDENCE SCORES:')
print(f'  High confidence (≥0.8):     {high_conf:2d} rows ({high_conf/total_rows*100:.1f}%)')
print(f'  Above threshold (≥0.6):     {above_threshold:2d} rows ({above_threshold/total_rows*100:.1f}%)')
print(f'  Below threshold (<0.6):     {below_threshold:2d} rows ({below_threshold/total_rows*100:.1f}%)')
print(f'  Average confidence:         {df["confidence"].mean():.3f}')
print(f'  Range:                      {df["confidence"].min():.3f} - {df["confidence"].max():.3f}')

print(f'\nFIELD COMPLETENESS:')
for field in fields_to_check:
    complete, pct = field_completeness[field]
    print(f'  {field:20s}: {complete:2d}/26 ({pct:5.1f}%)')

print(f'\nEXTRACTION METHODS:')
for method, count in method_counts.items():
    print(f'  {method:25s}: {count:2d} rows ({count/total_rows*100:.1f}%)')

# Calculate overall accuracy score
# Weight by field importance and completeness
weights = {
    'pct_fines': 0.10,
    'pct_sand': 0.10,
    'pct_gravel': 0.10,
    'll': 0.12,
    'pl': 0.12,
    'pi': 0.12,
    'ls': 0.12,
    'soil_description': 0.22,
}

field_scores = {}
for field, weight in weights.items():
    complete, pct = field_completeness[field]
    field_scores[field] = (pct / 100) * weight

overall_accuracy = sum(field_scores.values()) * 100
print(f'\nOVERALL ACCURACY: {overall_accuracy:.1f}%')
print('(Based on field completeness and confidence weighting)')
