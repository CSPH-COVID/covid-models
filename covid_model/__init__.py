from .model import CovidModel
from .regional_model import RegionalCovidModel

all_regions = {
  "cent": "Central",
  "cm": "Central Mountains",
  "met": "Metro",
  "ms": "Metro South",
  "ne": "Northeast",
  "nw": "Northwest",
  "slv": "San Luis Valley",
  "sc": "South Central",
  "sec": "Southeast Central",
  "sw": "Southwest",
  "wcp": "West Central Partnership"
}

all_counties = {
  "ad": "Adams County",
  "ar": "Arapahoe County",
  "bo": "Boulder County",
  "brm": "Broomfield County",
  "den": "Denver County",
  "doug": "Douglas County",
  "ep": "El Paso County",
  "jeff": "Jefferson County",
  "lar": "Larimer County",
  "mesa": "Mesa County",
  "pueb": "Pueblo County",
  "weld": "Weld County",
}

all_regions_and_counties = {**all_regions, **all_counties}
