# Expected Data Format

The adversarial debiasing pipeline expects CSV files with the following structure:

## Required Columns

```csv
id,full_text_english,gender,ict_label
1,"Software engineer with 5 years of experience in Python and machine learning...",1,1
2,"Administrative assistant with strong organizational skills...",0,0
3,"Data scientist specializing in NLP and deep learning...",1,1
4,"Marketing manager with 10 years in digital campaigns...",0,0
```

### Column Specifications

- **id**: Unique identifier (int or str)
- **full_text_english**: Full resume text after preprocessing (str)
  - Should include complete work history, education, skills
  - Preprocessed and translated to English if needed
  - No PII (names, contacts removed during anonymization)

- **gender**: Binary gender label (int)
  - 0 = Female
  - 1 = Male
  - Non-binary or missing values filtered out during loading

- **ict_label**: Binary job classification (int)
  - 0 = Non-ICT position
  - 1 = ICT position

## Example Entry

```
id: 42
full_text_english: "Professional experience includes software development at multiple tech companies. Bachelor's degree in Computer Science from Technical University. Proficient in Java, Python, SQL, and cloud technologies. Led team of 5 developers on enterprise applications."
gender: 1
ict_label: 1
```

## Data Split

- `data/processed/train.csv`: Training set
- `data/processed/test.csv`: Test set

Typical split: 80/20 train/test with stratification on gender and ict_label.

## Notes

- Dataset is under NDA and cannot be shared
- Original FINDHR dataset: 949 samples after filtering
- Gender distribution: ~48% female, ~46% male
- ICT distribution: ~47% ICT positions
