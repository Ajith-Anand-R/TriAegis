import pandas as pd
import uuid
import random

def generate_sample_data(num_patients=10):
    symptoms_list = [
        "chest pain", "shortness of breath", "severe headache", 
        "high fever", "nausea", "dizziness", "cough", "fatigue",
        "abdominal pain", "joint pain", "blurred vision", "confusion",
        "persistent vomiting", "uncontrolled bleeding", "seizure",
        "loss of consciousness", "severe allergic reaction", "minor injury"
    ]
    
    conditions_list = [
        "diabetes", "hypertension", "asthma", "heart disease",
        "obesity", "chronic kidney disease", "none", "anxiety",
        "COPD", "thyroid disorder", "arthritis", "depression"
    ]
    
    genders = ["Male", "Female", "Other"]
    
    data = []
    for _ in range(num_patients):
        patient_id = str(uuid.uuid4())
        age = random.randint(18, 90)
        gender = random.choice(genders)
        
        # Pick 1-3 random symptoms
        num_symptoms = random.randint(1, 3)
        symptoms = ", ".join(random.sample(symptoms_list, num_symptoms))
        
        # Blood pressure (Normal is 120/80, High can be 180/120)
        systolic = random.randint(100, 200)
        diastolic = random.randint(60, 120)
        bp = f"{systolic}/{diastolic}"
        
        # Heart rate (Normal 60-100)
        hr = random.randint(50, 160)
        
        # Temperature (Normal 98.6)
        temp = round(random.uniform(97.0, 105.0), 1)
        
        # Pre-Existing Conditions
        num_cond = random.randint(0, 2)
        if num_cond == 0:
            conditions = "none"
        else:
            conditions = ", ".join(random.sample(conditions_list, num_cond))
            
        data.append({
            "Patient_ID": patient_id,
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Blood Pressure": bp,
            "Heart Rate": hr,
            "Temperature": temp,
            "Pre-Existing Conditions": conditions
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_sample_data(20)
    
    # Save as CSV
    csv_file = "ehr_sample_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Generated {csv_file}")
    
    # Save as XLSX
    xlsx_file = "ehr_sample_data.xlsx"
    df.to_excel(xlsx_file, index=False)
    print(f"Generated {xlsx_file}")
