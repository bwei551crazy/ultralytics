#analysing and graphing data from csv file

#load the csv file. 

import pandas as pd
import matplotlib.pyplot as plt

def analyze_csv(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

#print a specified column on console

    print(df['metrics/mAP50-95(B)'])

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='metrics/mAP50-95(B)', color='blue')
    plt.title('mAP50-95 (B) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50-95 (B)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    csv_path = 'my_training_runs/yolo11l_ua_detrac_100/results.csv' #change this to your csv path
    analyze_csv(csv_path)
