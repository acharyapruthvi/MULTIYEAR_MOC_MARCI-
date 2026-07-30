import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

# ==========================Variables==========================
polar_north_base_dir = "polar_north"
polar_south_base_dir = "polar_south"

moc_north_dir = f"{polar_north_base_dir}/MOC/*.csv"
marci_north_dir = f"{polar_north_base_dir}/MARCI/*.csv"

moc_south_dir = f"{polar_south_base_dir}/MOC/*.csv"
marci_south_dir = f"{polar_south_base_dir}/MARCI/*.csv"

# ==========================Import Data==========================
moc_north_files = glob.glob(moc_north_dir)
marci_north_files = glob.glob(marci_north_dir)

moc_north_dataframe = pd.concat([pd.read_csv(f) for f in moc_north_files], ignore_index=True)

# ==========================Manipulation==========================
moc_north_dataframe['avg_major_latitude'] = moc_north_dataframe[['major_end1_lat', 'major_end2_lat']].mean(axis=1)

for my_val, group in moc_north_dataframe.groupby('MY'):
    plt.scatter(group['Ls'], group['avg_major_latitude'], label=f'MY {my_val}')

plt.xlabel('Ls')
plt.ylabel('Avg Major Latitude')
plt.legend(title='Mars Year')
plt.title('MOC North')
plt.show()