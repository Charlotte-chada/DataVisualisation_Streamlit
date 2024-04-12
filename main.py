import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

# Constants
DATA_URL = ("dataset.csv")
DATE_TIME = "date_time"  # Adjusted to use underscore

@st.cache_resource
def load_data():
    data = pd.read_csv(DATA_URL, parse_dates=[['crash_date', 'crash_time']])
    data.dropna(subset=['latitude', 'longitude'], inplace=True)  # Corrected the typo from 'longtitude' to 'longitude'
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={"crash_date_crash_time": DATE_TIME}, inplace=True)
    return data

# Load the data
data = load_data()

# Streamlit application headings
st.title("Motor Vehicle Collisions in New York City")
st.markdown("This application is a Streamlit dashboard used to analyze motor vehicle collisions in NYC 🗽💥🚗")

# Where are the most people injured in NYC?
st.header("Where are the most people injured in NYC?")
injured_people = st.slider("Number of persons injured in vehicle collisions", 0, 19)
st.map(data.query("injured_persons >= @injured_people")[['latitude', 'longitude']].dropna(how="any"))

# How many collisions occur during a given time of day?
st.header("How many collisions occur during a given time of day?")
hour = st.slider("Hour to look at", 0, 23)
data_by_hour = data[data[DATE_TIME].dt.hour == hour]
st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))

# Vehicle collisions map
midpoint = (np.average(data_by_hour["latitude"]), np.average(data_by_hour["longitude"]))
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data_by_hour[[DATE_TIME, 'latitude', 'longitude']],
            get_position=['longitude', 'latitude'],
            radius=100,
            extruded=True,
            pickable=True,
            elevation_scale=4,
            elevation_range=[0, 1000],
        ),
    ],
))

# Breakdown by minute between selected hours
st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = data_by_hour
hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({"minute": range(60), "crashes": hist})
fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
st.write(fig)

# Top 5 dangerous streets by affected class
st.header("Top 5 dangerous streets by affected class")
select = st.selectbox('Affected class', ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.write(data.query("number_of_pedestrians_injured >= 1")[["on_street_name", "number_of_pedestrians_injured"]]
             .sort_values(by="number_of_pedestrians_injured", ascending=False).dropna(how="any")[:5])
elif select == 'Cyclists':
    st.write(data.query("number_of_cyclist_injured >= 1")[["on_street_name", "number_of_cyclist_injured"]]
             .sort_values(by="number_of_cyclist_injured", ascending=False).dropna(how="any")[:5])
else:  # Motorists
    st.write(data.query("number_of_motorist_injured >= 1")[["on_street_name", "number_of_motorist_injured"]]
             .sort_values(by="number_of_motorist_injured", ascending=False).dropna(how="any")[:5])
    


# Group data by hour and calculate the number of collisions
hourly_collisions = data.groupby(data[DATE_TIME].dt.hour).size()

# Create a heatmap
fig_hourly_heatmap = px.imshow([hourly_collisions.values],
                               labels=dict(x="Hour of the Day", y="Number of Collisions", color="Collisions"),
                               x=hourly_collisions.index,
                               y=['Collisions'],
                               aspect="auto",
                               color_continuous_scale='Viridis')

fig_hourly_heatmap.update_xaxes(side="top")

st.write("Heatmap of Collisions Over Time (Hourly)", fig_hourly_heatmap)


# Group data by borough and calculate the number of collisions
borough_collisions = data['borough'].value_counts()

# Create a bar chart
fig_borough_collisions = px.bar(borough_collisions,
                                x=borough_collisions.index,
                                y=borough_collisions.values,
                                labels={'x': 'Borough', 'y': 'Number of Collisions'},
                                title="Number of Collisions by Borough")

st.write("Bar Chart for Collisions by Borough", fig_borough_collisions)


# Accident Severity Map
st.header("Map of Accident Severity")
severity = data.groupby(['latitude', 'longitude']).agg({'injured_persons':'sum'}).reset_index()
fig_severity = px.scatter_mapbox(severity, lat="latitude", lon="longitude", size="injured_persons", color="injured_persons",
                                   color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                                   mapbox_style="carto-positron")
st.write(fig_severity)



# Factor Analysis for Collisions
st.header("Top Contributing Factors for Collisions")
factor_columns = ['contributing_factor_vehicle_1', 'contributing_factor_vehicle_2']
factors = data[factor_columns].stack().value_counts().head(20)
fig_factors = px.bar(factors, x=factors.values, y=factors.index, orientation='h', labels={'y':'Contributing Factor', 'x':'Count'}, color=factors.values, color_continuous_scale=px.colors.sequential.Viridis)
st.write(fig_factors)


# Correlation Heatmap of Numerical Features
st.header("Correlation Heatmap")
numerical_features = ['number_of_persons_killed', 'number_of_pedestrians_injured', 'number_of_cyclist_injured', 'number_of_motorist_injured']
correlation_matrix = data[numerical_features].corr()
fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation Coefficient"), x=numerical_features, y=numerical_features)
st.write(fig_corr)


# Breakdown by Vehicle Type
st.header("Breakdown of Collisions by Vehicle Type")
vehicle_type_columns = ['vehicle_type_code_1', 'vehicle_type_code_2']
vehicle_types = data[vehicle_type_columns].stack().value_counts().head(10)
fig_vehicle_types = px.pie(values=vehicle_types.values, names=vehicle_types.index, title='Top 10 Vehicle Types Involved in Collisions')
st.write(fig_vehicle_types)


# Raw Data
if st.checkbox("Show Raw Data", False):
    st.subheader("Raw Data")
    st.write(data)
