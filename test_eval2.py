from baseline_engine import evaluate_baselines
import baseline_engine
baseline_engine.suppress_stdout_stderr = lambda: __import__("contextlib").nullcontext()
base_score, script_path, task = evaluate_baselines("data/titanic/train.csv", "Survived", "data/titanic/test.csv")
print("SCORE:", base_score)
