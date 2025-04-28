import os
import requests

# Define parameters for Nepal
lat_min = 26.0  # Southern boundary
lat_max = 31.0  # Northern boundary
lon_min = 80.0  # Western boundary
lon_max = 88.0  # Eastern boundary

# Models and scenarios
# models = ['ACCESS-CM2', 'CNRM-CM6-1']
models = ['CNRM-CM6-1']
# scenarios = ['ssp245', 'ssp585']
scenarios = ['historical']
product = ['tasmax', 'tasmin']
# Base URL
# version = 'r1i1p1f1' #check if climate data is available for this version
version = 'r1i1p1f2' #check if climate data is available for this version
base_url = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
#this provides precipitation in kg/m2/s and temperature in K

# Loop through models, scenarios, and products
for model in models:
    for scenario in scenarios:
        for prod in product:
            # Create output directory
            output_dir = f"ncss_data/{model}/{scenario}/{prod}"
            os.makedirs(output_dir, exist_ok=True)

            # Loop through years from 2020 to 2100
            for year in range(1950, 2015):
                filename = f"{prod}_day_{model}_{scenario}_{version}_gr_{year}.nc" #may need to change gr to gr1 or gn or .. depending on climate model
                request_url = (
                    f"{base_url}/{model}/{scenario}/{version}/{prod}/{filename}?var={prod}"
                    f"&north={lat_max}&south={lat_min}&west={lon_min}&east={lon_max}"
                    f"&horizStride=1&time_start={year}-01-01T12:00:00Z&time_end={year}-12-31T12:00:00Z"
                    f"&accept=netcdf3&addLatLon=true"
                )
                
                output_path = os.path.join(output_dir, filename)
                
                # Download file using requests
                print(f"Downloading {filename} for model {model}, scenario {scenario}, and product {prod}...")
                response = requests.get(request_url)
                response.raise_for_status()  # Raise an error for bad responses
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved to {output_path}\n")

print("Download complete!")

