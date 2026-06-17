import anthropic
import pandas as pd
import time

client = anthropic.Anthropic()
df = pd.read_csv('experiment_results.csv')

def extract_winner(text, team_a, team_b):
    last = text[-300:]
    for team in [team_a, team_b]:
        if team.lower() in last.lower():
            return team
    for team in [team_a, team_b]:
        if team.lower() in text.lower():
            return team
    return "UNCLEAR"

total = len(df)
for i, idx in enumerate(df.index):
    row = df.loc[idx]
    sid = row['Scenario ID']
    print(f"[{i+1}/{total}] {sid} 처리 중...")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            messages=[{"role": "user", "content": row['Prompt_CoT']}]
        )
        raw = response.content[0].text
        result = extract_winner(raw, row['Team A'], row['Team B'])
        correct = 1 if result == row['Winner'] else 0

        df.at[idx, 'Raw_CoT'] = raw
        df.at[idx, 'Result_CoT'] = result
        df.at[idx, 'Correct_CoT'] = correct
        print(f"  결과: {result} | 정답: {row['Winner']} | {'정답' if correct else '오답'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        df.at[idx, 'Result_CoT'] = 'ERROR'
        df.at[idx, 'Correct_CoT'] = -1

    if (i + 1) % 10 == 0:
        df.to_csv('experiment_results.csv', index=False)
        print(f"  >>> {i+1}개 중간 저장 완료")

    time.sleep(0.5)

df.to_csv('experiment_results.csv', index=False)
cot_acc = df['Correct_CoT'].sum()
print(f"\n=== CoT 재실험 완료 ===")
print(f"정확도: {cot_acc}/100 ({cot_acc}%)")