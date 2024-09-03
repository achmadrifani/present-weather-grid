#!/home/metpublic/PYTHON_VENV/present_weather/bin/python

# Script to generate present weather in grid format
# (30/3) First script
# (31/3) Add send ftp function
# (16/7) Modify config file to yaml, export in netcdf format
# (4/9) Add send sftp function

# Author: Achmad Rifani
# Created : 30/3/2024
# Updated : 4/9/2024

import pandas as pd
import pickle
import numpy as np
import requests
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import rioxarray
import geopandas as gpd
from datetime import datetime, timedelta
from ftplib import FTP, error_perm
import os
import yaml
import xarray as xr
import paramiko


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        print(key, value)
        globals()[key] = value


def get_latest_radar():
    """
    Fetches the latest radar data from a server, parses the filename to get the timestamp,
    and prepares the data in the form of an xarray dataset.

    Returns:
        radar_data (xarray Dataset): The latest radar data.
        radar_time (datetime): The timestamp of the radar data.
    """
    try:
        print("Fetching latest radar MERGE data")
        response = requests.get(API_RADAR, verify=False)
        response.raise_for_status()  # Check if the request was successful
        with rasterio.open(response.json()['file']) as src:
            radar_data = rioxarray.open_rasterio(src)

        filename = response.json()["file"].split("/")[4]
        time_str = filename[13:25]
        radar_time = datetime.strptime(time_str,"%Y%m%d%H%M")
        return radar_data, radar_time
    except Exception as e:
        print("Fetching latest radar MOSAIC data")
        response = requests.get(API_RADAR_MOSAIC, verify=False)
        response.raise_for_status()  # Check if the request was successful
        with rasterio.open(response.json()['file']) as src:
            radar_data = rioxarray.open_rasterio(src)

        filename = response.json()["file"].split("/")[4]
        time_str = filename[7:19]
        radar_time = datetime.strptime(time_str, "%Y%m%d%H%M")
        return radar_data, radar_time
        # print(f"An error occurred while fetching or processing the data: {e}")
        # return None, None


def get_latest_ldn():
    """
    Fetches the latest Lightning Detection Network (LDN) data from a server,
    and converts the data into a GeoDataFrame for spatial analysis.

    Returns:
        ldn_data (GeoDataFrame): A GeoDataFrame containing the latest LDN data.
    """
    try:
        print("Fetching latest LDN data")
        response = requests.get(API_LDN, verify=False)
        response.raise_for_status()  # Check if the request was successful
        ldn_data_json = response.json()
        ldn_data = gpd.GeoDataFrame.from_features(ldn_data_json["geojson"][0])
        return ldn_data
    except Exception as e:
        print(f"An error occurred while fetching or processing the data: {e}")
        return None


def get_latest_ot():
    """
    Fetches the latest Overshooting Top (OT) data from a CSV file hosted satellite server,
    renames a column for clarity, and converts the data into a GeoDataFrame for spatial analysis.

    Returns:
        ot_data (GeoDataFrame): A GeoDataFrame containing the latest OT data.
    """
    try:
        print("Fetching latest OT data")
        df = pd.read_csv(URL_OT, sep=",")
        df.rename(columns={"OT_corrected": "aggregate"}, inplace=True)
        ot_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
        return ot_data
    except Exception as e:
        print(f"An error occurred while fetching or processing the data: {e}")
        return None


def get_latest_sat_filename(ftp, remote_dir, local_dir):
    ftp.cwd(remote_dir)  # Pindah ke direktori yang diinginkan di server FTP

    # Mendapatkan daftar nama file di direktori
    file_list = ftp.nlst(remote_dir)

    # Mengurutkan daftar nama file secara ascending
    file_list_sorted = sorted(file_list, reverse=True)

    # Mengambil nama file pertama setelah diurutkan
    if file_list_sorted:
        latest_file = file_list_sorted[0]
        filename = latest_file.split("/")[-1]

        # Unduh file pertama
        local_path = os.path.join(local_dir, filename)
        with open(local_path, 'wb') as local_file:
            ftp.retrbinary('RETR ' + latest_file, local_file.write)
    else:
        print("Tidak ada file di direktori.")
    ftp.quit()
    return filename


def get_latest_sat_bt():
    # inisiasi waktu
    now = datetime.utcnow()
    # mengambil data terakhir dari ftp satelit
    ftp_server = FTP_SATELIT.get("HOST")
    ftp_user = FTP_SATELIT.get("USER")
    ftp_password = FTP_SATELIT.get("PASS")
    remote_dir = f"{FTP_SATELIT.get('DATA_DIR')}/{now:%Y}/{now:%m}/{now:%d}"
    ftp = FTP(ftp_server)
    ftp.login(user=ftp_user, passwd=ftp_password)
    print(f"Fetching satellite data at : {now: %Y-%m-%d %H:%M}")
    # latest_file, latest_mtime = get_latest_modified_file(ftp, remote_dir, local_dir)
    sat_latest_file = get_latest_sat_filename(ftp, remote_dir, DATA_DIR)
    print(f"Latest file available sat_bt : {sat_latest_file}")

    # parsing nama file menjadi objek datetime
    date_str = sat_latest_file.split("_")[4]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[10:12])
    minute = int(date_str[12:14])

    # Buat objek datetime
    sat_datetime = datetime(year, month, day, hour, minute)
    print(f"Parsing filename to datetime object: {sat_datetime}")

    raster_path = os.path.join(DATA_DIR, sat_latest_file)
    rds = rasterio.open(raster_path)
    sat_data_bt = rioxarray.open_rasterio(rds)
    sat_data_bt = sat_data_bt.where(~np.isnan(sat_data_bt), np.nan)
    print(f"Read satellite data success {raster_path}")
    return sat_data_bt, sat_datetime


def create_radius(point):
    return point.buffer(5 / 111.32) # 1 degree latitude = 111.32 km


def calculate_wx_grid(radar_data, lg_grid, sat_data_int):
    # Calculate the weather category
    wx_grid_shape = radar_data[0].shape
    weather_category = np.full(wx_grid_shape, 9999)  # Initialize with default value
    rdr_data = radar_data[0]
    sat_data = sat_data_int[0]

    # Assign weather categories based on the conditions
    weather_category[(np.isnan(rdr_data)) & (sat_data >= 21)] = 1
    weather_category[(np.isnan(rdr_data)) & (sat_data < 21) & (sat_data >= 0)] = 2
    weather_category[(np.isnan(rdr_data)) & (sat_data < 0)] = 3

    weather_category[(rdr_data > 0) & (rdr_data <= 0.5)] = 4  # Combined condition for simplicity

    weather_category[(rdr_data > 0.5) & (rdr_data < 5) & np.isnan(lg_grid)] = 60
    weather_category[(rdr_data > 0.5) & (rdr_data < 5) & ~np.isnan(lg_grid)] = 95

    weather_category[(rdr_data >= 5) & (rdr_data < 10) & np.isnan(lg_grid)] = 61
    weather_category[(rdr_data >= 5) & (rdr_data < 10) & ~np.isnan(lg_grid)] = 95

    weather_category[(rdr_data >= 10) & (rdr_data < 20) & np.isnan(lg_grid)] = 63
    weather_category[(rdr_data >= 10) & (rdr_data < 20) & ~np.isnan(lg_grid)] = 95

    weather_category[(rdr_data >= 20) & np.isnan(lg_grid)] = 63
    weather_category[(rdr_data >= 20) & ~np.isnan(lg_grid)] = 97

    return weather_category

def send_ftp(host, port, username, password, local_file, remote_file):
    """
    Sends a file to an FTP server.

    Args:
        host (str): The FTP server address.
        port (int): The FTP server port.
        username (str): The FTP server username.
        password (str): The FTP server password.
        local_file (str): The path to the local file to be uploaded.
        remote_file (str): The path on the remote server where the file will be uploaded.
    """
    print(f"Sending to {host}:{port}")
    with FTP() as ftp:
        try:
            ftp.connect(host, int(port))  # Ensure port is an integer
            print(f"Logging in as {username}")
            ftp.login(user=username, passwd=password)
            with open(local_file, 'rb') as file:
                ftp.storbinary("STOR " + remote_file, file)
        except error_perm as e:
            print(f"FTP permission error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        else:
            print(f"File '{local_file}' uploaded successfully to '{remote_file}'")


def send_sftp(host, port, username, password, local_file, remote_file):
    """
    Sends a file to a remote server over SFTP.

    Parameters:
        host (str): The SFTP server's hostname or IP address.
        port (int): The port on which to connect.
        username (str): The username for authentication.
        password (str): The password for authentication.
        local_file (str): The path to the local file to be uploaded.
        remote_file (str): The path on the remote server where the file will be stored.
    """
    try:
        with paramiko.Transport((host, port)) as transport:
            print(f"Connecting to {host}:{port}")
            transport.connect(username=username, password=password)
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                sftp.put(local_file, remote_file)
                print(f"File '{local_file}' uploaded successfully to '{remote_file}'")
    except paramiko.AuthenticationException as e:
        print(f"Authentication failed: {e}")
    except paramiko.SSHException as e:
        print(f"SSH connection error: {e}")
    except IOError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



def main():
    # Radar
    radar_data, radar_time = get_latest_radar()
    radar_lons = radar_data.x.values
    radar_lats = radar_data.y.values

    # lightning
    ldn_data = get_latest_ldn()
    ot_data = get_latest_ot()
    all_lightning = pd.concat([ldn_data, ot_data], ignore_index=True)

    # satelite brightness temperature
    sat_bt, _ = get_latest_sat_bt()
    new_coords = {'x': radar_lons, 'y': radar_lats}

    # Interpolate data to new coordinates
    sat_bt_c = sat_bt - 273.15
    sat_data_int = sat_bt_c.interp(new_coords, method='linear')
    # Create polygon of ldn
    lg_radius = all_lightning.geometry.apply(create_radius)
    radius_df = gpd.GeoDataFrame(lg_radius)
    merged_poly = radius_df.unary_union

    # Preparing raster grid
    resolution = radar_lons[1] - radar_lons[0]
    width = int((radar_lons.max() - radar_lons.min()) / resolution) + 1
    height = int((radar_lats.max() - radar_lats.min()) / resolution) + 1
    transform = from_origin(radar_lons.min(), radar_lats.max(), resolution, resolution)
    lg_grid = np.zeros((height, width), dtype=np.uint8)

    # masking, convert 0 (non-lightning) to nan
    mask = geometry_mask([merged_poly], transform=transform, out_shape=(height, width), invert=True)
    lg_grid[mask] = 1
    lg_grid = lg_grid.astype(float)
    lg_grid[lg_grid == 0] = np.nan

    # Calculate weather grid
    weather_grid = calculate_wx_grid(radar_data, lg_grid, sat_data_int)

    # Save weather grid to raster
    height, width = weather_grid.shape
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',  # Misalnya, dengan CRS WGS84
        'transform': transform,
        'compress': 'lzw',
        'fillValue': 9999,
        'productTime': f'{radar_time:%Y%m%d%H%M%S}',
        "aggregate": "present_weather",
        "process": "bmkg_pws",
        "dim_time": f"{radar_time:%Y-%m-%dT%H:%M:%SZ}"
    }

    print("Creating tif file")
    tif_filename = f"present_weather_grid_{radar_time:%Y%m%d%H%M%S}.tif"
    with rasterio.open(f'{OUT_DIR}/{tif_filename}', 'w', **metadata) as dst:
        dst.write(weather_grid, 1)
        
    with rasterio.open(f'{OUT_DIR}/present_weather_grid_latest.tif', 'w', **metadata) as dst:
        dst.write(weather_grid, 1)
    print(f"TIF File created: {OUT_DIR}/present_weather_grid_{radar_time:%Y%m%d%H%M%S}.tif")

    # Create a netcdf file
    print("Creating netcdf file")
    nc_filename = f"present_weather_grid_{radar_time:%Y%m%d%H%M%S}.nc"
    ds = xr.Dataset(
        {
            "wx": (["lat", "lon"], weather_grid),
        },
        coords={
            "lat": radar_lats,
            "lon": radar_lons,
            "time": radar_time,
        },
        attrs={
            "description": "Present weather grid",
            "longName": "Present weather category",
            "institution": "BMKG",
            "units": "category",
            "crs": "EPSG:4326",
            "productTime": f"{radar_time:%Y-%m-%dT%H:%M:%SZ}",
            "Nx": width,
            "Ny": height,
        }
    )

    # Define compression settings
    compression = {
        "zlib": True,
        "complevel": 5  # Level of compression from 1 (lowest) to 9 (highest)
    }

    # Specify the encoding for each variable
    encoding = {
        "wx": compression
    }

    ds.to_netcdf(f"{OUT_DIR}/{nc_filename}", encoding=encoding, engine='netcdf4')
    ds.to_netcdf(f"{OUT_DIR}/present_weather_grid_latest.nc", encoding=encoding, engine='netcdf4')
    print(f"NC File created: {OUT_DIR}/present_weather_grid_{radar_time:%Y%m%d%H%M%S}.tif")

    return tif_filename, nc_filename


if __name__ == "__main__":
    now = datetime.now()

    config_file_path = 'pwx_grid_config.yml'
    read_config(config_file_path)

    tif_filename, nc_filename = main()

    if FTP_SEND_TIFF:
        for addr in FTP_SEND_TIFF:
            TYPE = addr.get("TYPE")
            HOST = addr.get("HOST")
            PORT = addr.get("PORT")
            USER = addr.get("USER")
            PASS = addr.get("PASS")
            REMOTE_DIR = addr.get("REMOTE_DIR")
            REMOTE_FILE = addr.get("REMOTE_FILE")
            if TYPE == "ftp":
                if REMOTE_FILE:
                    local_file = f"{OUT_DIR}/{tif_filename}"
                    remote_file = f"{REMOTE_DIR}/{REMOTE_FILE}"
                    send_ftp(HOST, PORT, USER, PASS, local_file, remote_file)

    if FTP_SEND_NC:
        for addr in FTP_SEND_NC:
            TYPE = addr.get("TYPE")
            HOST = addr.get("HOST")
            PORT = addr.get("PORT")
            USER = addr.get("USER")
            PASS = addr.get("PASS")
            REMOTE_DIR = addr.get("REMOTE_DIR")
            REMOTE_FILE = addr.get("REMOTE_FILE")
            if TYPE == "ftp":
                if REMOTE_FILE:
                    local_file = f"{OUT_DIR}/{nc_filename}"
                    remote_file = f"{REMOTE_DIR}/{REMOTE_FILE}"
                    send_ftp(HOST, PORT, USER, PASS, local_file, remote_file)
            elif TYPE == "sftp":
                if REMOTE_FILE:
                    local_file = f"{OUT_DIR}/{nc_filename}"
                    remote_file = f"{REMOTE_DIR}/{REMOTE_FILE}"
                    send_sftp(HOST, int(PORT), USER, PASS, local_file, remote_file)
            else:
                print("Invalid FTP type")



