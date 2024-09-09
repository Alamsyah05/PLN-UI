import streamlit as st
import pandas as pd
import pulp
import numpy as np
from pulp import LpMinimize, LpVariable, lpSum
from io import BytesIO

st.title("HRS NUMBER OPTIMIZATION & LOCATION SELECTION")
hrs = ['HRS 1', 'HRS 2', 'HRS 3']
selected_hrs = st.selectbox('HRS', hrs)

# Upload file Excel
uploaded_file = st.file_uploader("Unggah file Excel hasil Clustering Multidemand PSO", type="xlsx")
hrs1 = 188
hrs2 = 21
hrs3 = 20
D1 = 140356 #Kg
D2 = 16108 #kg
D3 = 15796 #kg
HRS1 = 249 #alt location
HRS2 = 103 #alt location
HRS3 = 87 #alt location


if uploaded_file is not None:
    # Load the data from the CSV file
    if selected_hrs == "HRS 1":
        data = pd.read_excel(uploaded_file, sheet_name='Alt Loc1')
        st.success("File berhasil diunggah! Silakan masukkan nilai HRS.")
        st.dataframe(data)
        d1 = st.number_input("Masukkan nilai demand HRS 1 (Kg)", min_value=1, value = D1)
        alt_location1 = st.number_input("Masukkan jumlah lokasi alternatif HRS 1", min_value=1, value=HRS1)
        hrs1 = st.number_input("Masukkan nilai HRS (jumlah demand yang dipilih)", min_value=1, max_value=len(data), value=hrs1)
    elif selected_hrs == "HRS 2":
        data = pd.read_excel(uploaded_file, sheet_name='Alt Loc2')
        st.success("File berhasil diunggah! Silakan masukkan nilai HRS.")
        st.dataframe(data)
        d2 = st.number_input("Masukkan nilai demand HRS 2 (Kg)", min_value=1, value = D2)
        alt_location2 = st.number_input("Masukkan jumlah lokasi alternatif HRS 2", min_value=1, value=HRS2)
        hrs2 = st.number_input("Masukkan nilai HRS (jumlah demand yang dipilih)", min_value=1, max_value=len(data), value=hrs2)
    elif selected_hrs == "HRS 3":
        data = pd.read_excel(uploaded_file, sheet_name='Alt Loc3')
        st.success("File berhasil diunggah! Silakan masukkan nilai HRS.")
        st.dataframe(data)
        d3 = st.number_input("Masukkan nilai demand HRS 3", min_value=1, value = D3)
        alt_location3 = st.number_input("Masukkan jumlah lokasi alternatif HRS 3", min_value=1, value=HRS3)
        hrs3 = st.number_input("Masukkan nilai HRS (jumlah demand yang dipilih)", min_value=1, max_value=len(data), value=hrs3)

    st.header("Fasilitas A")
    C_a = st.number_input("CAPEX (miliar)", min_value=0.0, value=157.0)
    K_a = 500  # Fixed value
    st.write(f"Kapasitas A (kg): {K_a}")

    st.header("Fasilitas B")
    C_b = st.number_input("CAPEX (miliar)", min_value=0.0, value=1.19)
    K_b = 1000  # Fixed value
    st.write(f"Kapasitas B (kg): {K_b}")

    # Button
    if st.button("Lanjutkan perhitungan"):
        #Assign weights based on Demand Type
        data['Weight'] = data['Demand Type'].apply(lambda x: 1 if x == 1 else 2)

        # Calculate the distances
        data['Distance'] = np.sqrt((data['Supply X'] - data['Demand X']) ** 2 + (data['Supply Y'] - data['Demand Y']) ** 2)

        # Set up the LP problem for HRS number optimization


        prob = pulp.LpProblem("Minimize_CAPEX", pulp.LpMinimize)

        # Decision variables
        x = pulp.LpVariable('x', lowBound=0, cat='Integer')  # Facility a
        y = pulp.LpVariable('y', lowBound=0, cat='Integer')  # Facility b

        # Objective function
        prob += C_a * x + C_b * y, "Total_CAPEX"

        # Constraints
        prob += K_a * x + K_b * y >= D3, "Demand"
        prob += x >= y + 1, "More_facility_a_than_b"
        prob += x + y <= HRS3, "Total_facility_limit"

        # Solve the problem
        prob.solve()
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Jumlah Fasilitas A", value=int(x.varValue), delta=f"CAPEX (miliar): Rp{C_a * x.varValue:,.2f}")
            st.metric(label="Jumlah Fasilitas B", value=int(y.varValue), delta=f"CAPEX (miliar): Rp{C_b * y.varValue:,.2f}")
        
        with col2:
            st.metric(label="Total CAPEX (miliar)", value=f"Rp{pulp.value(prob.objective):,.2f}")

        # Set up the Linear Programming problem for Location Selection
        prob = pulp.LpProblem("Minimize_Total_Distance", LpMinimize)

        # Decision variables
        demand_ids = data['Demand ID'].unique()
        x = LpVariable.dicts("select", demand_ids, cat='Binary')

        # Objective function
        prob += lpSum([x[i] * data.loc[data['Demand ID'] == i, 'Distance'].values[0] * data.loc[data['Demand ID'] == i, 'Weight'].values[0] for i in demand_ids])

        if selected_hrs == "HRS 1":
            prob += lpSum([x[i] for i in demand_ids]) == hrs1
        elif selected_hrs == "HRS 2":
            prob += lpSum([x[i] for i in demand_ids]) == hrs2
        elif selected_hrs == "HRS 3":
            prob += lpSum([x[i] for i in demand_ids]) == hrs3

        prob.solve()


        selected_demands = [i for i in demand_ids if x[i].varValue == 1]

        # Output the results
        st.success("Optimasi selesai! Berikut adalah Demand ID yang terpilih:")
        # st.write(selected_demands)

        # Save the selected demand points to a new CSV file
        selected_data = data[data['Demand ID'].isin(selected_demands)].reset_index(drop=True)
        st.dataframe(selected_data)
