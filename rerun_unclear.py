import anthropic
import pandas as pd

client = anthropic.Anthropic()
df = pd.read_csv('experiment_results.csv')

targets = [
    ('S050', 'Prompt_FewShot', 'Result_FewShot', 'Correct_FewShot', 'Raw_FewShot'),
    ('S068', 'Prompt_CoT', 'Result_CoT', 'Correct_CoT', 'Raw_CoT'),
]

def extract_winner(text, team_a, team_b):
    last = text[-300:]
    for team in [team_a, team_b]:
        if team.lower() in last.lower():
            return team
    for team in [team_a, team_b]:
        if team.lower() in text.lower():
            return team
    return "UNCLEAR"

for sid, prompt_col, result_col, correct_col, raw_col in targets:
    row = df[df['Scenario ID'] == sid].iloc[0]
    idx = df[df['Scenario ID'] == sid].index[0]
    print(f"처리 중: {sid} ({prompt_col})")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=3000,
        messages=[{"role": "user", "content": row[prompt_col]}]
    )
    raw = response.content[0].text
    result = extract_winner(raw, row['Team A'], row['Team B'])
    correct = 1 if result == row['Winner'] else 0
    df.at[idx, raw_col] = raw
    df.at[idx, result_col] = result
    df.at[idx, correct_col] = correct
    print(f"결과: {result} | 정답: {row['Winner']}")

df.to_csv('experiment_results.csv', index=False)
print("저장 완료!")