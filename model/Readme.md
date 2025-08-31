Project structure
root/
│── frontend/ has src/components/HomePage/HomePage.jsx
│── model/
│   ├── telecom/
│   └── super_market/
│       ├── cleaning/
│       ├── dataset/
│       ├── models/
│       │    └── modelwith_input.ipynb   # contains training + last cell has prediction logic
│       ├── output_datasets/
│       └── saved_models/
│            ├── model1.pkl
│            ├── model2.pkl
│            └── model3.pkl
│── backend/







Project structure
root/
│── model/
│   ├── telecom/
│   └── super_market/
│       ├── cleaning/
                 ├── clean.ipynb
│                ├── cleaning2.ipynb
│                └── train.ipynb
│       ├── dataset/
│       ├── models/
│       │    └── modelwith_input.ipynb   # contains training + last cell has prediction logic
             ├── model.ipynb
│            └── Report Generation using_LLM.py
│       ├── output_datasets/
│       └── saved_models/
│            ├── model1.pkl
│            ├── model2.pkl
│            └── model3.pkl
