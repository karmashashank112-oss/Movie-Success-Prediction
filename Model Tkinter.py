import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Load Saved Features
reg_model = joblib.load('xgb_reg_model.pkl')
x_features = joblib.load('X_features.pkl')

# GUI Window setup
root = tk.Tk()
root.title('🎥 Movie Success Prediction')
root.geometry('600x500')
root.resizable(False,False)
root.config(bg='#0F445C')
 

#Dropdown List
Genre =['Action', 'Adventure', 'Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','Foregin',
        'History','Horror','Music','Mystery','Romance','Science Fiction','Thriller','Tv Movie','War','Western']

Language = ['En','De','Es','Zh','Ja','Fr','Da','It','Sv','Hi','Ru','Pt','Ko','Af','Ro','Nl','Ar','He','Th','Cn',
'Tr','Cs','Fa','No','Ps','Vi','El','Hu','Nb','Xx','Id','Pl','Is','Te','Ta','Ky','Sl']

# Title Label
title_label = tk.Label(
    root,
    text ='🎥 Movie Success Prediction Dashboard',
    font = ('Arial', 18,'bold'),
    bg="#f8f9fa",
    fg="#212529"
)
title_label.pack(pady=15)

#Input Frame

frame = tk.Frame(root, bg="#f8f9fa")
frame.pack(pady=10)

#Label and Input

labels = ['Budget (in millions)','Runtime (in minutes)','Genre','Language', 'Release Year']
entries = {}

for i, label_text in enumerate(labels):
    label = tk.Label(frame, text=label_text, font=('Arial', 10), bg="White")
    label.grid(row=i, column=0, sticky='w', padx=10, pady=8)

    if label_text == 'Genre':        
        entries[label_text] = ttk.Combobox(frame, values=Genre, width=25, state='readonly')
        entries[label_text].current(0)

    elif label_text == 'Language':
        entries[label_text] = ttk.Combobox(frame, values=Language, width=25, state='readonly')
        entries[label_text].current(0)

    else:
        entries[label_text] = tk.Entry(frame, width=28)

    entries[label_text].grid(row=i, column=1, pady=8, padx=10)


# Prediction Function

def predict():
    try:
        # Get inputs
        budget = float(entries['Budget (in millions)'].get())
        runtime = float(entries['Runtime (in minutes)'].get())
        genre = entries['Genre'].get()
        language = entries['Language'].get()
        year = int(entries['Release Year'].get())

        # Label encoding maps (must match what was used in training)
        genre_map = {
            'Action': 1, 'Adventure': 2, 'Animation': 3, 'Comedy': 4, 'Crime': 5,
            'Documentary': 6, 'Drama': 7, 'Family': 8, 'Fantasy': 9, 'Foregin': 10,
            'History': 11, 'Horror': 12, 'Music': 13, 'Mystery': 14, 'Romance': 15,
            'Science Fiction': 16, 'Thriller': 17, 'Tv Movie': 18, 'War': 19, 'Western': 20
        }

        language_map = {
            'En': 1, 'De': 2, 'Es': 3, 'Zh': 4, 'Ja': 5, 'Fr': 6, 'Da': 7, 'It': 8,
            'Sv': 9, 'Hi': 10, 'Ru': 11, 'Pt': 12, 'Ko': 13, 'Af': 14, 'Ro': 15,
            'Nl': 16, 'Ar': 17, 'He': 18, 'Th': 19, 'Cn': 20, 'Tr': 21, 'Cs': 22,
            'Fa': 23, 'No': 24, 'Ps': 25, 'Vi': 26, 'El': 27, 'Hu': 28, 'Nb': 29,
            'Xx': 30, 'Id': 31, 'Pl': 32, 'Is': 33, 'Te': 34, 'Ta': 35, 'Ky': 36, 'Sl': 37
        }

        # Encode categorical inputs
        genre_code = genre_map.get(genre, 0)
        language_code = language_map.get(language, 0)

        # Prepare input vector for regression model
        input_vector = np.array([[budget, runtime, year, genre_code, language_code]])

        # Predict Profit
        predicted_profit = reg_model.predict(input_vector)[0]

        # Decide Hit/Flop based on profit threshold
        profit_threshold = 20000000  # 2 crore example
        result = 'Hit' if predicted_profit >= profit_threshold else 'Flop'

        # Display results
        profit_label.config(
            text=f"💰 Predicted Profit: ₹{predicted_profit:,.2f}",
            fg="#28a745" if predicted_profit > 0 else "#dc3545"
        )
        result_label.config(
            text=f"🎯 Movie Result: {result}",
            fg="#28a745" if result == 'Hit' else "#dc3545"
        )

        import pandas as pd
        import os

        save_data = pd.DataFrame([{
            "Budget (M)": budget,
            "Runtime (min)": runtime,
            "Genre": genre,
            "Language": language,
            "Year": year,
            "Predicted Profit": round(predicted_profit, 2),
            "Result": result
        }])

        file_path = "predictions_log.csv"

        # If file exists, append without header; else create new
        if os.path.exists(file_path):
            save_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            save_data.to_csv(file_path, index=False)

        print(f"✅ Prediction saved to {file_path}")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for Budget, Runtime, and Year.")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# Predict Button

predict_btn = tk.Button(
    root,
    text = 'Predict',
    font = ('Arial', 14, 'bold'),
    bg="#007bff",
    fg = 'white',
    width=15,
    command=predict
)
predict_btn.pack(pady=15)

profit_label = tk.Label(root, text='', font=('Arial', 14,'bold'), bg='white')
profit_label.pack(pady=5)

result_label = tk.Label(root, text='', font=('Arial', 16,'bold'), bg='white')
result_label.pack(pady=5)


root.mainloop()
