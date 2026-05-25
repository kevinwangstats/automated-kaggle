from baseline_engine import evaluate_baselines
base_score, script_path, task = evaluate_baselines("data/titanic/train.csv", "Survived", "data/titanic/test.csv")
print("SCORE:", base_score)
