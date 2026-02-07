import folium

def create_map(data):
    m = folium.Map(location=[20, 78], zoom_start=4)

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            popup=f"Magnitude: {row['Magnitude']}",
            color="red",
            fill=True
        ).add_to(m)

    m.save("earthquake_map.html")
