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

