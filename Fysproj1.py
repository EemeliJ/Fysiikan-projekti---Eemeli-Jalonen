# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scipy.signal as signal
import folium
import streamlit.components.v1 as components

st.markdown("<h1 style='text-align: center; color: black;'>Iltalenkki</h1>", unsafe_allow_html=True) 

# df = pd.read_csv('Accelerometer.csv')
df = pd.read_csv('https://raw.githubusercontent.com/EemeliJ/Fysiikan-projekti---Eemeli-Jalonen/main/Accelerometer.csv')

# Z KOMPONENTTI JA SUODATUS
st.markdown("<h2 style='text-align: center; color: black;'>Suodatettu Z-komponentti</h2>", unsafe_allow_html=True)

step_comp = df['Acceleration z (m/s^2)']

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

T = df['Time (s)'].iloc[-1] - df['Time (s)'].iloc[0]
n = len(df['Time (s)'])
fs = n / T  


order = 3
cutoff = 1 / (0.2)

filtered_signal = butter_lowpass_filter(step_comp, cutoff, fs, order)

start_time = 60
end_time = 120
mask = (df['Time (s)'] >= start_time) & (df['Time (s)'] <= end_time)

filtered_signal_short = filtered_signal[mask]
time_short = df['Time (s)'][mask]

plt.figure(figsize=(20, 4))
plt.plot(time_short, filtered_signal_short, color='purple')
plt.grid()
plt.axis([start_time, end_time, -2, 22])
# plt.title('Suodatettu Kiihtyvyys Z (60-120 s)')
plt.xlabel('Aika (s)')
plt.ylabel('Kiihtyvyys (m/s²)')
plt.legend()

st.pyplot(plt)
plt.clf()

# SUODATETTU ASKELMÄÄRÄ
filtered_signal_mean_removed = filtered_signal - np.mean(filtered_signal)

jaksot = 0
for i in range(1, len(filtered_signal_mean_removed)):
    if filtered_signal_mean_removed[i] * filtered_signal_mean_removed[i - 1] < 0:
        jaksot += 1

st.write('Askelmäärä suodatuksen perusteella', np.floor(jaksot / 2))

# TEHOSPEKTRI
st.markdown("<h2 style='text-align: center; color: black;'>Tehospektri</h2>", unsafe_allow_html=True)

filtered_signal_short -= np.mean(filtered_signal_short)
n = len(filtered_signal_short)
frequencies = np.fft.fftfreq(n, 1/fs)
fft_values = np.fft.fft(filtered_signal_short)

power = np.abs(fft_values)**2 / n 

positive_frequencies = frequencies[:n//2]
positive_power = power[:n//2]

freq_mask = (positive_frequencies > 0) & (positive_frequencies <= 14)
filtered_frequencies = positive_frequencies[freq_mask]
filtered_power = positive_power[freq_mask]

filtered_power[filtered_power > 450] = 450

plt.figure(figsize=(10, 6))
plt.plot(filtered_frequencies, filtered_power, color='purple')
# plt.title(f'Tehospektri ({start_time}-{end_time} s)')
plt.xlabel('Taajuus (Hz)')
plt.ylabel('Teho')
plt.grid()
plt.axis([0, 14, 0, 450])

st.pyplot(plt)
plt.clf()

# ASKELMÄÄRÄ FOURIER-ANALYYSI
step_count_from_power = 0
for f, p in zip(filtered_frequencies, filtered_power):
    if 1 <= f <= 3:
        step_count_from_power += p

step_count_from_power = step_count_from_power / 7

st.write('Askelmäärä Fourier-analyysin perusteella:', np.floor(step_count_from_power))

# KARTTA
st.markdown("<h2 style='text-align: center; color: black;'>Reitti kartalla</h2>", unsafe_allow_html=True)

# df_location = pd.read_csv('Location.csv')
df_location = pd.read_csv('https://raw.githubusercontent.com/EemeliJ/Fysiikan-projekti---Eemeli-Jalonen/main/Location.csv')

df_location.columns = df_location.columns.str.strip()

lat_mean = df_location['Latitude (°)'].mean()
long_mean = df_location['Longitude (°)'].mean()

my_map = folium.Map(location=[lat_mean, long_mean], zoom_start=14)

start_time = 10
mask = df_location['Time (s)'] >= start_time

folium.PolyLine(df_location.loc[mask, ['Latitude (°)', 'Longitude (°)']], color='purple', opacity=1).add_to(my_map)

map_file = 'map.html'
my_map.save(map_file)

with open(map_file, 'r', encoding='utf-8') as f:
    map_html = f.read()

components.html(map_html, height=600)

# KESKIMÄÄRÄINEN NOPEUS (m/s) ASKELPITUUS (cm) JA MATKA

df_location['Distance (m)'] = np.sqrt((df_location['Latitude (°)'].diff() * 111320)**2 + df_location['Longitude (°)'].diff() * 111320 * np.cos(np.radians(df_location['Latitude (°)']))**2)**2)

total_distance = df_location['Distance (m)'].sum()
total_time = df_location['Time (s)'].iloc[-1] - df_location['Time (s)'].iloc[0]

average_speed = total_distance / total_time if total_time > 0 else 0

step_length = total_distance / (np.floor(jaksot / 2)) * 100

st.write(f'Keskimääräinen nopeus (m/s): {average_speed:.1f}')
st.write(f'Askelpituus (cm): {step_length:.1f}')
st.write(f'Kokonaismatka (km): {total_distance / 1000:.1f}')

