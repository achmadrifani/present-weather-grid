ROOT_DIR = "/home/metpublic"
TASK_DIR = "TASK_RELEASE/EFAN/present_weather_grid"
OUT_DIR = f"{ROOT_DIR}/{TASK_DIR}/tif_output"
DATA_DIR = f"{ROOT_DIR}/{TASK_DIR}/data"
SRC_DIR = ""

API_RADAR = "https://radar.bmkg.go.id:8060/getRadarGeotif?token=19387e71e78522ae4172ec0fda640983b8438c9cfa0ca571623cb69d8327&radar=amandemenForecast&type=latest"
API_LDN = "https://radar.bmkg.go.id:8060/getLatestLD?typeData=1"
URL_OT = "http://202.90.198.22/IMAGE/HIMA/H09_OT_Indonesia.csv"
FTP_SATELIT = {"HOST":"202.90.199.64",
               "USER":"kspcr",
               "PASS":"kspcr!@#",
               "DATA_DIR":"/himawari_vol/himawari6/others/hima_cor"}

FTP_SEND = [{
    "NAME":"publik",
    "TYPE":"ftp",
    "HOST":"publik.bmkg.go.id",
    "PORT":"21",
    "USER":"amandemen",
    "PASS":"bmkg2303",
    "REMOTE_DIR":"data",
    "REMOTE_FILE":"present_weather_grid_latest.tif"
}]