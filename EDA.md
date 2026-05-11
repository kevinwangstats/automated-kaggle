# Exploratory Data Analysis

## Dataset Shape
- Rows: 891
- Columns: 12

## Columns
```json
[
  "PassengerId",
  "Survived",
  "Pclass",
  "Name",
  "Sex",
  "Age",
  "SibSp",
  "Parch",
  "Ticket",
  "Fare",
  "Cabin",
  "Embarked"
]
```

## Missing Values (>0%)
```json
{
  "Age": 19.87,
  "Cabin": 77.1,
  "Embarked": 0.22
}
```

## Categorical Cardinality
```json
{
  "Name": 891,
  "Sex": 2,
  "Ticket": 681,
  "Cabin": 147,
  "Embarked": 3
}
```

## Highly Skewed Features (|skew| > 1)
```json
{
  "SibSp": 3.7,
  "Parch": 2.75,
  "Fare": 4.79
}
```