import pandas as pd

df = pd.read_csv('aligned_corpus.tsv', sep='|', encoding='utf-8')

# 1. Find all rows with 'בראשית' in Hebrew
breshit_rows = df[df['Samaritan'].str.contains('בראשית', na=False)]
print(f'Found {len(breshit_rows)} rows with בראשית in Hebrew:')
for i, row in breshit_rows.head(5).iterrows():
    print(f'Hebrew: {row["Samaritan"]}')
    print(f'Aramaic: {row["Targum"]}\n')

# 2. Find all rows with 'ראשית' in Hebrew
reshit_rows = df[df['Samaritan'].str.contains('ראשית', na=False)]
print(f'Found {len(reshit_rows)} rows with ראשית in Hebrew:')
for i, row in reshit_rows.head(5).iterrows():
    print(f'Hebrew: {row["Samaritan"]}')
    print(f'Aramaic: {row["Targum"]}\n')

# 3. Check for common Aramaic equivalents
aramaic_equivs = ["בריש", "ברישא", "בראש", "בראשיתא"]
for equiv in aramaic_equivs:
    count = df[df['Targum'].str.contains(equiv, na=False)].shape[0]
    print(f'{equiv}: {count} times in Aramaic') 