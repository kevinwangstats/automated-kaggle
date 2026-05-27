# Exploratory Data Analysis

## Dataset Overview
- Rows: 891
- Columns: 12
- Target Column: `Survived`

## Target Profiling
```json
{
  "Task": "classification",
  "Class Distribution": {
    "0": "549 (61.62%)",
    "1": "342 (38.38%)"
  }
}
```

## Feature-to-Target Correlation (Numerical)
```json
{
  "Top 5 Positive": {
    "Fare": 0.2573,
    "Parch": 0.0816,
    "PassengerId": -0.005,
    "SibSp": -0.0353,
    "Age": -0.0772
  },
  "Top 5 Negative": {
    "Parch": 0.0816,
    "PassengerId": -0.005,
    "SibSp": -0.0353,
    "Age": -0.0772,
    "Pclass": -0.3385
  }
}
```

## Missing Values (>0%)
```json
{
  "Age": 19.87,
  "Cabin": 77.1,
  "Embarked": 0.22
}
```

## Categorical Value Samples (Top 5 Unique)
```json
{
  "Name": [
    "Braund, Mr. Owen Harris",
    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
    "Heikkinen, Miss. Laina",
    "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
    "Allen, Mr. William Henry"
  ],
  "Sex": [
    "male",
    "female"
  ],
  "Ticket": [
    "A/5 21171",
    "PC 17599",
    "STON/O2. 3101282",
    "113803",
    "373450"
  ],
  "Cabin": [
    "C85",
    "C123",
    "E46",
    "G6",
    "C103"
  ],
  "Embarked": [
    "S",
    "C",
    "Q"
  ]
}
```

## Data Signature (Random 5-Row Sample)
```csv
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
710,1,3,"Moubarek, Master. Halim Gonios (""William George"")",male,,1,1,2661,15.2458,,C
440,0,2,"Kvillner, Mr. Johan Henrik Johannesson",male,31.0,0,0,C.A. 18723,10.5,,S
841,0,3,"Alhomaki, Mr. Ilmari Rudolf",male,20.0,0,0,SOTON/O2 3101287,7.925,,S
721,1,2,"Harper, Miss. Annie Jessie ""Nina""",female,6.0,0,1,248727,33.0,,S
40,1,3,"Nicola-Yarred, Miss. Jamila",female,14.0,1,0,2651,11.2417,,C

```
