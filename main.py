import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import io

# Function 1: Create a graph from the data
def create_graph(adjacency_df, cij_df):
    G = nx.DiGraph()
    for i, row in adjacency_df.iterrows():
        source = row['node_code']
        for j, connected in row.items():
            if j != 'node_code' and pd.notna(connected):
                target = int(j)
                weight = cij_df.iloc[i][j]
                if pd.notna(weight):
                    G.add_edge(source, target, weight=weight)
    return G

# Function 2: Compute the shortest path
def compute_shortest_path(G, start_node, end_node):
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return edges
    except nx.NetworkXNoPath:
        return []

# Function 3: Allocate flow increment
def allocate_flow_increment(path, demand, flow_matrix):
    for edge in path:
        i, j = edge
        flow_matrix[int(i) - 1, int(j) - 1] += demand
    return flow_matrix

# Main processing function
def process_file(uploaded_file):
    try:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)

        adjacency_df = pd.read_excel(xls, sheet_name='adjacency')
        cij_df = pd.read_excel(xls, sheet_name='cij')
        od_demand_df = pd.read_excel(xls, sheet_name='OD')

        # Parse OD demand
        od_demand = {}
        for i, row in od_demand_df.iterrows():
            origin = row['node_code']
            for destination, demand in row.items():
                if destination != 'node_code' and pd.notna(demand):
                    od_demand[(origin, int(destination))] = demand

        # Create graph
        G = create_graph(adjacency_df, cij_df)

        # Initialize flow matrix
        num_nodes = len(adjacency_df)
        flow_matrix = np.zeros((num_nodes, num_nodes))

        # Process each OD pair
        for (origin, destination), demand in od_demand.items():
            path = compute_shortest_path(G, origin, destination)
            if path:
                flow_matrix = allocate_flow_increment(path, demand, flow_matrix)

        # Convert flow matrix to DataFrame
        flow_df = pd.DataFrame(flow_matrix, index=adjacency_df['node_code'], columns=adjacency_df['node_code'])

        return flow_df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Streamlit UI
st.title("Shortest Path Flow Allocation")

st.write("Upload an Excel file containing adjacency, cost matrices, and OD demand to compute the shortest path flow matrix.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    st.success("File uploaded successfully! Processing...")

    # Process file
    flow_df = process_file(uploaded_file)

    if flow_df is not None:
        st.write("### Generated Flow Matrix")
        st.dataframe(flow_df)

        # Convert DataFrame to Excel for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            flow_df.to_excel(writer, sheet_name='FlowMatrix')

        st.download_button(
            label="Download Flow Matrix",
            data=output.getvalue(),
            file_name="flow_matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
