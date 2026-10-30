# AEGIS: Automated Event & HydroGraph Identification Strategy
Digital Filter-based Runoff Event Identification Python Tool

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Event-detector is a Python module to Identify Runoff Events in Streamflow Data

## Streamlit app

`app/aegis_app.py` runs the AEGIS workflow on any USGS gauge with instantaneous discharge:
pick a state and gauge (shown on a map with its NLDI upstream basin and drainage area), a
period, the baseflow filter and its parameters, and an event threshold relative to the mean
annual maximum flow. The app downloads instantaneous discharge and stage from USGS NWIS,
applies `code/preproc.py`, `code/metric_calc.py` (S4 signal clarity) and
`code/EventDetector.py`, and shows the daily record with the detected events, a clickable
event timeline, and a detailed plot of the selected event.

Options:
- **Event separation**: the AEGIS start-to-start windows, or the peak-based split in
  `code/event_split.py`, which keeps every independent peak above the event threshold.
- **AORC rainfall**: hourly basin-average precipitation from NOAA AORC v1.1 (1 km, read
  anonymously from AWS) is added above the hydrographs, with event rainfall, peak intensity
  and runoff coefficient. AORC currently ends on 2025-12-31.

Downloaded data are stored per gauge in `data/USGS-<site_no>/` (USGS instantaneous values,
AORC rainfall, basin polygon, and the detected events with the settings that produced them)
and can be downloaded from the app.

```bash
pip install -r app/requirements.txt
streamlit run app/aegis_app.py
```

## References
[1]	K. Eckhardt, “How to construct recursive digital filters for baseflow separation,” Hydrol. Process, vol. 19, no. 2, pp. 507–515, Feb. 2005, doi: 10.1002/hyp . 5675.<br>
[2]	T. Chapman and A. Maxwell, “Baseflow Separation - Comparison of Numerical Methods with Tracer Experiments,” presented at the Hydrology and Water Resources Symposium 1996: Water and the Environment; Preprints of Papers, Barton, ACT: Institution of Engineers, Australia, Jan. 1996. Accessed: Aug. 17, 2025. [Online]. Available: https://search.informit.org/doi/10.3316/informit.360361071346753 <br>
[3]	R. J. Nathan and T. A. McMahon, “Evaluation of automated techniques for base flow and recession analyses,” Water Resour. Res., vol. 26, no. 7, pp. 1465–1473, Jul. 1990, doi: 10.1029/wr026i007p01465.


