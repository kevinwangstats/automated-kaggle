from baseline_engine import create_template_script
print(create_template_script("data/titanic/train.csv", "Survived", "xgb", "data/titanic/test.csv"))
