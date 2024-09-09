import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
import io

source_proj = "epsg:4326"  # WGS84
dest_proj = "epsg:32748"   # UTM zone 48S

st.title('Particle Swarm Optimization Clustering')

# Constants for PSO
number = st.number_input("Input Cognitive Weight (scale : 0 - 1)", min_value=0.00, max_value=1.00, format="%.2f")
numbers = round(number, 2)
cw = numbers
sw1 = 1 - cw
sw = round(sw1, 2)
write = st.write("Cognitive Weight: ", cw)
write1 = st.write("Social Weight: ", sw)
st.write("")
num_particles = st.number_input("Input Numbers of Particle", min_value=0)
st.write("Jumlah Partikel: ", num_particles)
st.write("")
iw_initial = st.number_input("Input Current Inertia Weight", min_value=0.00, value=0.9)
st.write("Current Inertia Weight: ", iw_initial)
st.write("")
iw_final = st.number_input("Input Final Inertia Weight", min_value=0.00, value=0.1)
st.write("Inertia Final Weight: ", iw_final)
st.write("")

uploaded_files = st.file_uploader("Upload your Excel File Contains : Koordinat SPBU, Koordinat SPKLU, Koordinat PLTH", accept_multiple_files=True, type=['xlsx'])
if uploaded_files is not None:
    for i in uploaded_files:
        data = pd.read_excel(i)
        st.write("filename:", i.name)
        st.dataframe(data)
else:
    st.warning("Please upload an Excel file to proceed.")

# Initialize the transformer
transformer = Transformer.from_crs(source_proj, dest_proj, always_xy=True)

x0, y0, x1, y1, x2, y2 = None, None, None, None, None, None

# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Determine which file is being processed
        filename = uploaded_file.name
        
        if "spbu" in filename.lower():
            # Process SPBU file
            data0 = pd.read_excel(uploaded_file, sheet_name='kordinat_spbu')
            latitude0 = data0['Latitude']
            longitude0 = data0['Longitude']
            coordinates_spbu = np.vstack([latitude0, longitude0]).T
            x0, y0 = transformer.transform(coordinates_spbu[:, 1], coordinates_spbu[:, 0])
            st.write(f"Processed {filename} (SPBU)")
            # st.write("X0:", x0)
            # st.write("Y0:", y0)
            
        elif "spklu" in filename.lower():
            # Process SPKLU file
            data1 = pd.read_excel(uploaded_file, sheet_name='Sheet1')
            latitude1 = data1['latitude']
            longitude1 = data1['longitude']
            coordinates_spklu = np.vstack([latitude1, longitude1]).T
            x1, y1 = transformer.transform(coordinates_spklu[:, 1], coordinates_spklu[:, 0])
            st.write(f"Processed {filename} (SPKLU)")
            # st.write("X1:", x1)
            # st.write("Y1:", y1)
        
        elif "plth" in filename.lower():
            # Process PLTH file
            data2 = pd.read_excel(uploaded_file, sheet_name='Sheet1')
            latitude2 = data2['Latitude']
            longitude2 = data2['Longitude']
            coordinates_plth = np.vstack([latitude2, longitude2]).T
            x2, y2 = transformer.transform(coordinates_plth[:, 1], coordinates_plth[:, 0])
            st.write(f"Processed {filename} (PLTH)")
            # st.write("X2:", x2)
            # st.write("Y2:", y2)
        
        else:
            st.write(f"Filename {filename} does not match any known patterns.")


# Define the objective function
def objective_function(demand_points_type1, demand_points_type2, supply_points, assignment):
    demand_points = np.vstack([demand_points_type1, demand_points_type2])
    assignment = np.round(assignment).astype(int)  # Ensure assignment is integer indices
    total_distance = 0
    total_travel_emission = 0
    total_production_emission = 0
    travel_emission_factors = 2
    production_emission_factors = np.array([0.02, 0.04, 0.04, 0.04, 0.04, 0.01, 0.04])
    units_per_demand_point = 500


    # Calculate travel emissions
    for i, demand_point in enumerate(demand_points):
        supply_point = supply_points[assignment[i]]
        distance = np.linalg.norm(demand_point - supply_point)
        total_distance += distance
        total_travel_emission += distance * travel_emission_factors


    # Calculate hydrogen emissions
    for i in range(len(supply_points)):
        total_units_produced = np.sum(assignment == i) * units_per_demand_point
        total_production_emission += production_emission_factors[i] * total_units_produced


    # Total emissions
    total_emission = total_travel_emission + total_production_emission
    return total_emission


# Define the Particle class
class Particle:
    def __init__(self, bounds, num_demand_points):
        self.position = np.random.randint(0, bounds[1] + 1, num_demand_points)
        self.velocity = np.zeros(num_demand_points)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.value = float('inf')


    def update_personal_best(self, objective_function, demand_points_type1, demand_points_type2, supply_points):
        self.value = objective_function(demand_points_type1, demand_points_type2, supply_points, self.position)
        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position.copy()


    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight=cw, social_weight=sw):
        cognitive_component = cognitive_weight * np.random.rand(len(self.position)) * (self.best_position - self.position)
        social_component = social_weight * np.random.rand(len(self.position)) * (global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component


    def update_position(self, bounds):
        self.position = np.round(self.position + self.velocity).astype(int)
        self.position = np.clip(self.position, bounds[0], bounds[1])


# Define the PSO algorithm
def pso_clustering(demand_points_type1, demand_points_type2, supply_points, num_particles, num_iterations):
    demand_points = np.vstack([demand_points_type1, demand_points_type2])
    num_demand_points = len(demand_points)
    num_supply_points = len(supply_points)
    bounds = [0, num_supply_points - 1]


    # Initialize the swarm
    swarm = [Particle(bounds, num_demand_points) for _ in range(num_particles)]
    global_best_position = np.random.randint(0, num_supply_points, num_demand_points)
    global_best_value = float('inf')


    # Define initial and final inertia weights
    # iw_initial = 0.9
    # iw_final = 0.1
   

    # To track best value per iteration
    best_values_per_iteration = []


    for iteration in range(num_iterations):
        # Calculate the current inertia weight
        iw = iw_initial - (iw_initial - iw_final) * (iteration / num_iterations)


        for particle in swarm:
            particle.update_personal_best(objective_function, demand_points_type1, demand_points_type2, supply_points)
            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = particle.best_position.copy()


        for particle in swarm:
            particle.update_velocity(global_best_position, inertia_weight=iw)
            particle.update_position(bounds)


        # Store the best value for the current iteration
        best_values_per_iteration.append(global_best_value)


    return global_best_position, global_best_value, best_values_per_iteration


# Plotting function
def plot_clustering(demand_points, demand_types, supply_points, assignment):
    fig, ax = plt.subplots()
    # Define markers for different demand types
    markers = {1: 'o', 2: 's'}  # 'o' for Type 1, 's' for Type 2


    # Create a scatter plot for each type of demand point
    for demand_type in np.unique(demand_types):
        indices = np.where(demand_types == demand_type)
        scatter = ax.scatter(demand_points[indices][:, 0], demand_points[indices][:, 1],
                    c=assignment[indices], cmap='viridis', marker=markers[demand_type], label=f'Demand Type {demand_type}')


    # Plot the supply points
    ax.scatter(supply_points[:, 0], supply_points[:, 1], c='red', marker='x', label='Supply Points')


    # Draw lines between demand points and their assigned supply points
    for i, demand_point in enumerate(demand_points):
        supply_point = supply_points[assignment[i]]
        ax.plot([demand_point[0], supply_point[0]], [demand_point[1], supply_point[1]], 'k-', lw=0.25)


    # Add titles and labels
    ax.set_title('Demand Points Clustering to Supply Points')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    st.pyplot(fig)
    # return fig


demand_spklu = np.array([x1, y1]).T  # demand point from SPKLU
demand_spbu = np.array([x0, y0]).T   # demand point from SPBU
supply_points = np.array([x2, y2]).T  # supply points from PLTH

# Create an array to store the types of demand points
type_spklu = np.ones(len(demand_spklu))  # 1 for demand type A
type_spbu = np.full(len(demand_spbu), 2)  # 2 for demand type B


# Merge the demand points
demand_points_type1 = demand_spklu
demand_points_type2 = demand_spbu
demand_points = np.vstack((demand_spklu, demand_spbu))
demand_types = np.concatenate((type_spklu, type_spbu))



# num_particles = 100
num_iterations = 100

def save_results_to_excel(demand_points, demand_types, supply_points, assignment, best_value, best_values_per_iteration):
    results = []
    for i, demand_point in enumerate(demand_points):
        supply_point = supply_points[assignment[i]]
        results.append({
            'Demand ID': i,
            'Demand Type': demand_types[i],
            'Demand X': demand_point[0],
            'Demand Y': demand_point[1],
            'Supply ID': assignment[i],
            'Supply X': supply_point[0],
            'Supply Y': supply_point[1]
        })

    results_df = pd.DataFrame(results)
    results_df['Best Value'] = best_value

    best_values_df = pd.DataFrame(best_values_per_iteration, columns=['Best Value per Iteration'])

    # Simpan ke buffer memori menggunakan BytesIO
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name='Clustered Results', index=False)
        best_values_df.to_excel(writer, sheet_name='Best Values per Iteration', index=True)
    output.seek(0)  # Reset posisi buffer ke awal

    return output


if st.button("Calculate"):
    # Jalankan clustering
    best_position, best_value, best_values_per_iteration = pso_clustering(demand_points_type1, demand_points_type2, supply_points, num_particles, num_iterations)

    # Simpan hasil ke memori
    excel_data = save_results_to_excel(demand_points, demand_types, supply_points, best_position, best_value, best_values_per_iteration)

    # Plot hasil clustering
    plot_clustering(demand_points, demand_types, supply_points, best_position)

    # Tambahkan tombol unduhan di Streamlit
    st.download_button(
        label="Unduh Hasil dalam Excel",
        data=excel_data,
        file_name='Hasil.xlsx',
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # best_position, best_value, best_values_per_iteration = pso_clustering(demand_points_type1, demand_points_type2, supply_points, num_particles, num_iterations)
    # save_results_to_excel(demand_points, demand_types, supply_points, best_position, best_value, best_values_per_iteration, 'results_2-3(12).xlsx')


    # # Plot the results
    # plot_clustering(demand_points, demand_types, supply_points, best_position)


    # # Save results to Excel and provide download link
    # buffer = save_results_to_excel(demand_points, demand_types, supply_points, best_position, best_values_per_iteration)
    # st.download_button(
    #     label="Download Results",
    #     data=buffer,
    #     file_name='results_2-3.xlsx',
    #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    # )


# if uploaded_file is not None:
#     # Membaca file Excel menggunakan pandas
#     df = pd.read_excel(uploaded_file)
   
#     # Menampilkan dataframe
#     st.write("Data dari file Excel:")
#     st.write(df)


#     # Menampilkan info tambahan jika diperlukan
#     st.write(f"Jumlah baris: {len(df)}")
#     st.write(f"Jumlah kolom: {len(df.columns)}")


# st.write(demand_points_type1, demand_points_type2, supply_points)
# st.success(plot_clustering(demand_points, demand_types, supply_points, best_position))







