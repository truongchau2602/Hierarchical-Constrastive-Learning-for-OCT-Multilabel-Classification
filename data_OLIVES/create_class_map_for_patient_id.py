import pandas as pd
import json

# Đường dẫn đến file CSV
csv_file = '/content/drive/MyDrive/IEEE_2023_Ophthalmic_Biomarker_Det/TRAIN/Training_Unlabeled_Clinical_Data.csv'

# Đọc file CSV vào DataFrame
df = pd.read_csv(csv_file)

# Lọc các giá trị duy nhất của cột "Patient_ID" và sắp xếp theo thứ tự tăng dần
unique_patient_ids = sorted(df['Patient_ID'].unique())

# Chuyển đổi giá trị int64 thành int
unique_patient_ids = [int(patient_id) for patient_id in unique_patient_ids]

# Tạo dictionary với key từ 1 đến n và giá trị là các giá trị duy nhất của "Patient_ID"
patient_id_dict = {patient_id: i for i, patient_id in enumerate(unique_patient_ids)}

# Lưu dictionary thành file JSON
output_file = '/content/drive/MyDrive/hierarchicalContrastiveLearning_new/data_OLIVES/class_map.json'
with open(output_file, 'w') as file:
    json.dump(patient_id_dict, file, indent=2)