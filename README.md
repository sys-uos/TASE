

<div align="center">
  <h1>Territorial Acoustic Species Estimation Algorithm</h1>
  <p>Territorial Acoustic Species Estimation using Acoustic Sensor Networks using species classifiers such as BirdNET</p>
  <img src="pics/TASE_grabs.png" alt="TASE Workflow" width="750">
</div>

## Abstract 
Accurate biodiversity assessment is fundamental for effective conservation management and environmental policy-making.
But monitoring local species populations is time-consuming as experts can cover only one limited area at a time and is also prone to errors due to varying knowledge and experience by the experts. 
Advances in low-cost autonomous recording units and AI-based classifiers offer new tools for species monitoring.
However, current tools for acoustic species monitoring, while useful in identifying species, fall short in providing data on local populations. 
This limitation emphasizes the demand for more sophisticated methods, as uncertainties in estimating species populations can lead to misleading conclusions and misclassification of conservation statuses. 
In this work, we take a significant step towards more sophisticated monitoring by presenting a Territorial Acoustic Species Estimation approach, called TASE, to extract spatial, territorial patterns of species using acoustic sensor networks, allowing the estimation of territorial individuals of a species. 
It requires a distributed sensor network and exploits the characteristic spatial distribution of territorial species. 
We formalize TASE, apply it on bird acoustics, and share a proof-of-concept evaluation in a real-world deployment in a nature reserve, deploying 29 devices over 12 hectares. 
We show that it works on-par compared to the time-consuming practice applied by bird experts and can provide novel insights into spatial use of sound-producing territorial species. 

For details, please read our ([Pre-Print]( http://ssrn.com/abstract=5113322)) 

## Introduction

This repo contains the TASE algorithm and scripts for processing data used for publication.
This is the most advanced version of TASE for acoustic analyses, and we will keep this repository up-to-date and provide improved interfaces to enable scientists with no CS background to run the analysis.

Feel free to use TASE for your acoustic analyses and research. Please cite as: 

```
@article{Bruggemann2025,
  author    = {Brüggemann, Leonhard and Otten, Daniel and Sachser, Frederik and Aschenbruck, Nils},
  title     = {Territorial Acoustic Species Estimation using Acoustic Sensor Networks},
  year      = {2025},
  month     = {January 8},
  journal   = {SSRN Electronic Journal},
  doi       = {10.2139/ssrn.5113322},
  url       = {https://ssrn.com/abstract=5113322}
}
```


## How to Use
1. Clone the repository:
    
```
git clone https://github.com/sys-uos/TASE.git
```

2. Install the dependencies:

    ```pip install -r requirements.txt```

3. See in ```main.py``` the method ```main_minimal_usage_example.py```.

## Minimal Working Example

This minimal working example demonstrates how to apply Territorial Acoustic Species Estimation (TASE) using a subset of data from a real-world deployment. 
The script processes acoustic sensor network data to estimate bird territories based on Sylvia atricapilla recordings.

1. **Define Species of Interest**: The script initializes the Sylvia atricapilla species from the core species definitions.

```
   spec = Sylvia_atricapilla()
```

2. **Parse Node Locations**: The function reads node locations from a CSV file and converts them into UTM coordinates.
```
    node_locations = parse_audiomoth_locations(csv_node_locations)
   location_data_list = convert_wgs84_to_utm(node_locations, zone_number=32, zone_letter='N')
```

3. **Set Deployment Duration**: Defines a 10-minute subset for processing (to limit computation time).
```
   dt_start = datetime.datetime(2023, 6, 3, 7, 0, 0)  # 07:00 AM CEST
   dt_end = datetime.datetime(2023, 6, 3, 7, 10, 0)  # 07:10 AM CEST
   deployment_start, deployment_end = dt1_aware.timestamp(), dt2_aware.timestamp()
```

4. **Parse Classifier Results**: The script reads acoustic classification results and aligns them with timestamps.
```
if not os.path.exists(out_pkl_file):
    dict_devid_df = parse_classifications_as_dir(dir_path=dir_classification)
    dict_devid_df = add_date_to_classification_dataframe(dict_devid_df, deployment_start)
    dict_devid_df = check_and_fill_missing_entries(dict_devid_df)
    save_classification_data(dict_devid_df, out_pkl_file)
```

5. **Define TASE Parameters**: Thresholds and configuration for the Territorial Acoustic Species Estimation (TASE) algorithm.: The script reads acoustic classification results and aligns them with timestamps.

```
   params = Parameters(
       threshold_R=0.8,
       threshold_B=0.1,
       TS_delta=0.2,
       threshold_T=300,
   )
```

6. **Build Graph and Extract Territories**: Creates a directed graph and applies Delaunay triangulation to form a connectivity network. For each time step, the estimated bird locations are then stored.
```
   graph = BirdEstimatorDirected()
   graph.add_nodes_with_coordinates(device_list=location_data_list)
   graph.add_classifications_for_each_node(pkl_file=out_pkl_file)
   for ts in range(int(deployment_start) + 0, int(deployment_end) - 3, 1):
        print(f"Apply TASE on Epoch-Time {ts} to {ts+3}")
        graph.init_graph(directedGraph=True)
        graph.set_weight_to_timestamp(ts)
        graph.delauny(e_delta=0.2)
        graph.remove_long_edges(threshold_meter=700.0)
        territorial_subgraphs = graph.tase(threshold_R=params.threshold_R,
                                           threshold_B=params.threshold_B,
                                           threshold_T=params.threshold_T,
                                           TS_delta=params.TS_delta)

        # --- Estimate the birds location and append them to location --- #
        for root in territorial_subgraphs:
            territorial_subgraphs[root]['location'] = calculate_weighted_centroid(territorial_subgraphs[root]['TS'])
        territorial_subgraphs_all[ts] = territorial_subgraphs
```
7. **Extract and Visualize Results**: Estimated species locations are extracted and plotted as a heatmap overlay on a geospatial map.
```
dict_ts_centroids = extract_locations(territorial_subgraphs_all)
viewer = WMSMapViewer()
viewer.add_and_convert_pointcloudUTM_2_pointcloudPIXEL(dict_ts_centroids, zone_number=32, zone_letter='N')
viewer.add_node_locations(location_data_list)
viewer.display_with_heatmap(font_size=16, kde_bw_param=0.2, vmax=0.0001, figpath="./example/example_figure.pdf")
```

<div align="center">
   <img src="./pics/example_pointcloud.png" alt="TASE Pointcloud" width="300">
   <img src="./pics/example_heatmap.png" alt="TASE Heatmap" width="300">
</div>


## Adapting TASE for custom Deployment

The minimal working example above serves as a good reference for adapting TASE to a custom deployment. The main steps need to be adjusted based on specific requirements. 
It is recommended to experiment with different parameters when applying and visualizing the results.

1. **Create a Custom Species Class**: Define a new class that inherits from core.species.Species and set the required attributes:
```
class SomeCustomClass(core.species.Species):
    def __init__(self):
        self.lat_name = "Nomen Consuetudinarium"  # Latin Name
        self.eng_name = "Custom Name"            # English Name
```

2. **Parse Location Data**: Ensure that the location data is formatted as a list of Recording_Node objects (List[Recording_Node]).


3. **Calculate Deployment Start and End**: Convert the deployment's start and end times into epoch timestamps.


4. **Parse Classification Results**: Convert the classification results into a pandas DataFrame for processing.


5. **Set TASE Parameters**: Configure the parameters specific to your deployment to optimize performance.


6. **Apply TASE Across the Deployment Duration**: Run the TASE algorithm over the entire deployment period to estimate species territories.



## Integrate Data used in Publication for Replication
The data used in this project have been processed using the university service. Unfortunately, the original directory structure could not be preserved. Instead, the data are provided as a split .zip archive that must be recombined (refer to the provided link for instructions).

Once the unzipped directory is available, e.g., on an external drive, it can be easily accessed by creating a symbolic link. In Linux, this can be done by executing the following command within the cloned Github repository: ln -s path/to/unzipped_dir ./data

Data available at: [Link](TODO)


## Contact
For questions or issues, please contact [brueggemann@uni-osnabrueck.de].
