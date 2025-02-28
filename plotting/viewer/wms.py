from datetime import datetime

import networkx as nx
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from PIL import Image
import io
from pyproj import Proj, transform, Transformer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import matplotlib.patches as mpatches
from matplotlib.path import Path
from scipy.stats import gaussian_kde

from TASE.src.utils import calc_geometric_center_in_Graph

#######################################################################################################################
############### This viewer version is similar to the viewer outside, but modified for the publication ################
#######################################################################################################################


def s_to_time(seconds_total):
    h = 4 + int(seconds_total / 3600)
    min = int((seconds_total - (h - 4) * 3600) / 60)
    seconds_rest = seconds_total - ((h - 4) * 60 * 60 + min * 60)
    return f"{h:02}:{min:02}:{seconds_rest:02}"

class WMSMapViewer:

    def __init__(self, layer_name=None, bbox=None, img_width=600, img_height=800):
        # Connect to the WMS service
        self.wms = WebMapService("http://www.wms.nrw.de/geobasis/wms_nw_dtk", version="1.3.0")

        # Display available layers
        #print("Available layers:")
        # for layer in list(self.wms.contents):
        #     print(f"Layer name: {layer}, Title: {self.wms[layer].title}")

        self.layer_name = layer_name if layer_name != None else "nw_dtk_col"
        # self.layer_name = layer_name if layer_name != None else "nw_dtk_sw"

        # self.bbox = bbox if bbox is not None else (16.1223830963, 47.0569438684, 16.1774645014, 47.0864800496)self.bbox = bbox if bbox is not None else (8.05214167, 52.00720153, 8.05751681, 52.01278839)
        left, bottom, right, top = bbox if bbox is not None else (8.05214167, 52.00720153, 8.05751681, 52.01278839)
        # Margin factor (e.g., 0.05 for a 5% margin)
        margin_factor = 0.05
        # Calculating the margin for each side
        width_margin = (right - left) * margin_factor
        height_margin = (top - bottom) * margin_factor
        # Applying the margin
        self.bbox = (
            left - width_margin,
            bottom - height_margin,
            right + width_margin,
            top + height_margin
        )

        # Calculate width and height
        width = self.bbox[2] - self.bbox[0]  # max_lon - min_lon
        height = self.bbox[3] - self.bbox[1]  # max_lat - min_lat

        # Derive the aspect ratio as width/height
        aspect_ratio = width / height
        #print(f"Aspect Ratio (width:height) = {aspect_ratio:.2f}:1")

        self.img_width = img_width
        self.img_height = int(img_width / aspect_ratio)  # Calculate height based on the aspect ratio
        # self.img_width = img_width
        # self.img_height = img_height
        self.img = None
        self.center_gt = []

        # Define projections: WGS84 (lat/lon) and UTM (zone 32N for Germany)
        proj_wgs84 = Proj(proj="latlong", datum="WGS84")
        proj_utm = Proj(proj="utm", zone=32, south=False)  # UTM zone 32, northern hemisphere
        # Create a transformer for WGS84 to UTM conversion
        transformer = Transformer.from_proj(proj_wgs84, proj_utm, always_xy=True)
        # Convert each corner of the bounding box to UTM
        min_easting, min_northing = transformer.transform(self.bbox[0], self.bbox[1])
        max_easting, max_northing = transformer.transform(self.bbox[2], self.bbox[3])
        # Define the UTM bounding box
        margin = 500  # in meters
        self.bbox_utm = (min_easting - margin, min_northing - margin, max_easting + margin, max_northing + margin)

        self.node_locations = []

        # Fetch and display the map
        self.fetch_map()
        # self.show_map()
        #plt.show()

    def fetch_map(self):
        # Fetch the map image from the WMS service
        try:
            print(f"Requesting WMS map for layer '{self.layer_name}' with bbox {self.bbox}")
            response = self.wms.getmap(
                layers=[self.layer_name],
                srs="EPSG:4326",
                bbox=self.bbox,
                size=(self.img_width, self.img_height),
                format="image/png",
                transparent=True,
            )
            self.img = Image.open(io.BytesIO(response.read()))
            print("Map image fetched successfully.")
        except Exception as e:
            print(f"Error fetching map image: {e}")

    def show_map(self, alpha=0.25):
        # Display the map image using matplotlib
        if self.img is None:
            raise ValueError("Map image not fetched. Call fetch_map() first.")

        plt.imshow(self.img, alpha=alpha)
        #plt.axis("off")
        #plt.title("WMS Map Layer from NRW")
        #plt.show()

    def display(self, img_alpha=0.25, utm_zone=32, utm_zone_letter='N', font_size=20):

        fig, ax = plt.subplots(figsize=(10, 9))

        # set background map
        if not self.img is None:
            plt.imshow(self.img, alpha=img_alpha)

        # set node locations
        if not self.node_locations is None:
            node_x_coords = [location[0] for location in self.node_locations]
            node_y_coords = [location[1] for location in self.node_locations]
            nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='dimgray', marker='x', s=75,
                                       label='Recorder')


        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [(-1) * min([int(self.bbox_utm[0]), int(self.bbox_utm[2])]) + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0]))
                    for tick in x_ticks]
        y_labels = [(-1) * min([int(self.bbox_utm[1]), int(self.bbox_utm[3])]) +
                    int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (
                                self.bbox_utm[3] - self.bbox_utm[1]))
                    for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        ax.set_xlabel(f"UTM {utm_zone}{utm_zone_letter} Easting {min([int(self.bbox_utm[0]), int(self.bbox_utm[2])]):,}m", fontsize=font_size)
        ax.set_ylabel(f"UTM {utm_zone}{utm_zone_letter} Northing {min([int(self.bbox_utm[1]), int(self.bbox_utm[3])]):,}m", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        plt.show()

    def display_graph(self, img_alpha=0.25, utm_zone=32, utm_zone_letter='N', font_size=20):

        fig, ax = plt.subplots(figsize=(10, 9))

        # set background map
        if not self.img is None:
            plt.imshow(self.img, alpha=img_alpha)

        # set node locations
        if not self.node_locations is None:
            node_x_coords = [location[0] for location in self.node_locations]
            node_y_coords = [location[1] for location in self.node_locations]
            nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='dimgray', marker='x', s=75,
                                       label='Recorder')


        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [(-1) * min([int(self.bbox_utm[0]), int(self.bbox_utm[2])]) + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0]))
                    for tick in x_ticks]
        y_labels = [(-1) * min([int(self.bbox_utm[1]), int(self.bbox_utm[3])]) +
                    int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (
                                self.bbox_utm[3] - self.bbox_utm[1]))
                    for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        ax.set_xlabel(f"UTM {utm_zone}{utm_zone_letter} Easting {min([int(self.bbox_utm[0]), int(self.bbox_utm[2])]):,}m", fontsize=font_size)
        ax.set_ylabel(f"UTM {utm_zone}{utm_zone_letter} Northing {min([int(self.bbox_utm[1]), int(self.bbox_utm[3])]):,}m", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        plt.show()



    def add_node_locations(self, node_locations, zone_number=33, zone_letter='N'):
        for node in node_locations:
            self.node_locations.append(self.convert_utm_to_pixel(node.lat, node.lon, zone_number=zone_number, zone_letter=zone_letter))

    def add_circle(self, center, radius, opacity=0.5, color='red'):
        """
        Adds a circle overlay to the map.

        Parameters:
        - center: tuple (x, y) - Center of the circle in the image's pixel coordinates.
        - radius: int - Radius of the circle in pixels.
        - opacity: float - Opacity of the circle (0 to 1).
        - color: str - Color of the circle.
        """
        if self.img is None:
            raise ValueError("Map image not fetched. Call fetch_map() first.")

        # Show the map (in case not already displayed)
        self.show_map()

        # Add the circle overlay with specified opacity
        circle = plt.Circle(center, radius, color=color, alpha=opacity)
        plt.gca().add_patch(circle)

    def convert_radius_to_pixels(self, radius_meters):
        """
        Converts a radius in meters to a pixel radius for drawing a circle on an image.

        Parameters:
        - radius_meters: float - The radius in meters (UTM units) to convert.
        - utm_bbox: tuple (min_easting, min_northing, max_easting, max_northing) - UTM bounding box for the map.
        - img_width, img_height: int - Width and height of the image in pixels.

        Returns:
        - radius_pixels: int - Radius in pixels for drawing on the image.
        """
        # Calculate the spatial extent in UTM units
        utm_width = self.bbox_utm[2] - self.bbox_utm[0]  # Easting range
        utm_height = self.bbox_utm[3] - self.bbox_utm[1]  # Northing range

        # Calculate scale factor (meters to pixels)
        scale_x = self.img_width / utm_width
        scale_y = self.img_height / utm_height

        # Average the scale factors to maintain aspect ratio for the radius
        scale_avg = (scale_x + scale_y) / 2

        # Convert the radius to pixels
        radius_pixels = int(radius_meters * scale_avg)

        return radius_pixels

    def add_circleset_from_utm(self, centerset, color='red'):
        if self.img is None:
            raise ValueError("Map image not fetched. Call fetch_map() first.")

        for elem in centerset:
            lat, lon, radius, color = elem
            pixel_x, pixel_y = self.convert_latlon_to_pixel(lat, lon, zone_number=32, zone_letter='N')
            try:
                radius_pixel = self.convert_radius_to_pixels(radius_meters=radius)
            except ValueError:
                radius_pixel = self.convert_radius_to_pixels(radius_meters=50)
            self.center_gt.append((pixel_x, pixel_y, radius_pixel, color))

    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    import numpy as np

    def display_with_pointcloud(self, deployment_start, deployment_end,
                                size=20, font_size=20, alpha=0.5, figpath=''):

        fig, ax = plt.subplots(figsize=(10, 9))

        # Plot the map background image
        ax.imshow(self.img, alpha=alpha)  # Display the background map with specified transparency

        # Define colormap based on timestamps
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(vmin=deployment_start, vmax=deployment_end)
        colors = [cmap(normalize(timestamp)) for timestamp in self.pointcloud_pixel.keys()]

        # xloc = [loc[0] for loc in self.pointcloud_pixel.values() if len(loc) >= 1]
        xloc, yloc, ts = [], [], []
        for key in self.pointcloud_pixel.keys():
            values = self.pointcloud_pixel[key]
            for p in values:
                xloc.append(p[0])
                yloc.append(p[1])
                ts.append(key)

        c = plt.scatter(x=xloc, y=yloc, c=ts, marker='.', s=80, vmin=deployment_start, vmax=deployment_end, cmap=cmap,
                        alpha=0.75)

        # Create a new axis on the right side of the plot with the same height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and pad as needed

        # Add a colorbar to the right
        cbar = plt.colorbar(c, cax=cax, label='Time of day', ticks=np.arange(deployment_start, deployment_end + 1, 3600))
        cbar.ax.set_ylabel('Time of day', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

        labels = [s_to_time(ts) for ts in np.arange(0, 3600 * 6 + 1, 3600)]
        cbar.ax.set_yticklabels(labels, fontsize=font_size)

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-434421 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0]))
                    for tick in
                    x_ticks]
        y_labels = [-5761732 +
                    int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (
                                self.bbox_utm[3] - self.bbox_utm[1]))
                    for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        ax.set_xlabel("UTM 32N Easting [+434.421m]", fontsize=font_size)
        ax.set_ylabel("UTM 32N Northing [+5.761.732m]", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        plt.subplots_adjust(right=0.15)
        plt.tight_layout()
        if figpath:
            plt.savefig(figpath)

        plt.show()

    def display_with_heatmap(self, size=20, font_size=20, alpha=0.5, figpath='', bw_method= 0.1, heatmap_vmax = 0.00001):

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 9))

        # Extract x and y coordinates from the point cloud
        x_coords = [centroid[0] for centroids in self.pointcloud_pixel.values() for centroid in centroids]
        y_coords = [centroid[1] for centroids in self.pointcloud_pixel.values() for centroid in centroids]

        # print(x_coords)
        # print(y_coords)

        # Plot node locations with a distinctive marker and add to legend
        node_x_coords = [location[0] for location in self.node_locations]
        node_y_coords = [location[1] for location in self.node_locations]
        nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='dimgray', marker='x', s=75,
                                   label='Recorder')

        # Plot node locations with a distinctive marker and add to legend
        node_x_coords = [location[0] for location in self.node_locations]
        node_y_coords = [location[1] for location in self.node_locations]
        nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='dimgray', marker='x', s=75,
                                   label='Recorder')

        # Check if there are any coordinates to plot
        calculate_kde = True
        if not x_coords or not y_coords or len(x_coords) < 20:
            img = None # set in order to have a empty colorbar
            calculate_kde = False
            print("No points to display.")
            ax.text(
                0.5, 0.5,  # Center of the plot
                "Point density could not be computed by KDE\n (Too few points)",
                transform=ax.transAxes,
                fontsize=font_size,
                color='black',
                ha='center',
                va='center',
                fontstyle='italic',
                rotation=25,  # Rotate the text by 45 degrees
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
            )
        if calculate_kde:
            try:
                # Calculate the kernel density estimate
                xy = np.vstack([x_coords, y_coords])
                kde = gaussian_kde(xy, bw_method=bw_method)  # Adjust the bandwidth for smoothing; lower value = sharper peaks

                # Create a grid over the image for evaluating the KDE
                x_grid = np.linspace(0, self.img_width, 500)
                y_grid = np.linspace(0, self.img_height, 500)
                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                z = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)

                img = None
                if not z.max() == 0: # in case too few points are available, all values in z are 0. then the img gets corrupted
                    # Define a custom colormap that starts with white and transitions to red (or any color desired)
                    white_to_red = LinearSegmentedColormap.from_list("white_to_green", ["white", "green"])

                    print(z.max() / 2)
                    # Display the KDE heatmap with the custom colormap
                    img = ax.imshow(z, extent=(0, self.img_width, 0, self.img_height), origin='lower',
                                    cmap=white_to_red, alpha=1.0, vmax=heatmap_vmax)  # vmax=heatmap_vmax

                else:
                    ax.text(
                        0.5, 0.5,  # Center of the plot
                        "Point density could not be computed by KDE\n(Too few points or other issues)",
                        transform=ax.transAxes,
                        fontsize=font_size,
                        color='black',
                        ha='center',
                        va='center',
                        fontstyle='italic',
                        rotation=25,  # Rotate the text by 45 degrees
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
                    )
            except ValueError:
                img = None # set in order to have a empty colorbar
                print("Singular covariance matrix - > KDE cannot handle that")
                #exit(1)
                pass

        # Plot the map background image
        ax.imshow(self.img, alpha=alpha)  # Display the background map with specified transparency

        # Add the circles of the territory
        territory_circle_different_colors = {}  # used for legend
        for circle in self.center_gt:
            if circle[3] == "red":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label="Field-monitored territory (safe)")
            elif circle[3] == "orange":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label="Field-monitored territory (possible)")
            else:
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label='Field-monitored territory (outside)')

            territory_circle_different_colors[circle[3]] = territory_circle
            plt.gca().add_patch(territory_circle)

        # Create an axis on the right for the colorbar with matching height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if img is not None:
            cbar = fig.colorbar(img, cax=cax, extend='max')
            cbar.set_label('TASE Point Density', fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            formatter.set_scientific(True)
            cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.update_ticks()
            cbar.ax.yaxis.offsetText.set_visible(True)
        else:
            # Define a custom colormap from white to red
            white_to_red = LinearSegmentedColormap.from_list('white_to_green', ['white', 'green'])

            # Create the ScalarMappable with the colormap and normalization
            norm = plt.Normalize(vmin=0, vmax=heatmap_vmax)  # vmin=0 ensures the colorbar starts at 0
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=white_to_red)

            # Create the colorbar
            cbar = fig.colorbar(scalar_mappable, cax=cax, extend="max")
            cbar.set_label('TASE Point Density', fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size)

            # Optional: Customize tick labels further if needed
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-1, 1))
            formatter.set_scientific(True)
            cbar.ax.yaxis.set_major_formatter(formatter)
            cbar.update_ticks()

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-434421 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [ -5761732 +
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        ax.set_xlabel("UTM 32N Easting [+434,421m]", fontsize=font_size)
        ax.set_ylabel("UTM 32N Northing [+5,761,732m]", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        # Add legend for territory_circle with the title "Ground Truth"
        try:
            # Create a custom legend entry for the number of points
            num_points_legend = mpatches.Patch(color='none', label=f'Number of points: {len(x_coords)}')

            # empty_handle = Line2D([], [], color='none', marker='', label=r"$\underline{\text{Territory}}$")
            # Assuming territory_circle_different_colors is a dictionary with labels as keys and plot objects as values
            territory_handles = list(territory_circle_different_colors.values())
            territory_labels = list([ x.get_label() for x in territory_circle_different_colors.values()])

            # Combine territory handles with nodes_scatter into one list for a single legend
            # all_handles = [nodes_scatter] + [empty_handle] + territory_handles
            # all_labels = ["Node Locations"] + ["Territory"] + territory_labels
            all_handles = [num_points_legend]+ [nodes_scatter] + territory_handles
            all_labels = [f"\# Points: {len(x_coords)}"] + ["Recorder"] + territory_labels

            # Create a single legend in the upper right with a title "Territory" only for the first section
            legend = ax.legend(handles=all_handles, labels=all_labels, loc="lower left", fontsize=font_size)
            # legend.set_title("Territory")
            legend.get_title().set_fontsize(font_size)  # Set font size for the title

            ax.add_artist(legend)  # Manually add the legend to the plot
            legend.get_frame().set_alpha(0.5)
        except UnboundLocalError:
            # If territory_circle_different_colors is not defined, show only the nodes legend
            ax.legend(handles=[nodes_scatter], labels=["Node Locations"], loc="lower left", fontsize=font_size)

        plt.tight_layout()

        # Save figure if a path is provided
        if figpath:
            plt.savefig(figpath)
        plt.show()

    def display_grid_only(self, font_size=20, figpath=''):
        # Create a figure for the grid map only
        fig, ax = plt.subplots(figsize=(12, 5))

        # Prepare data for grid map, ensuring all timestamps are datetime objects
        datetime_objs = [
            datetime.fromtimestamp(ts) if isinstance(ts, int) else ts
            for ts in self.pointcloud_pixel.keys()
        ]

        # Extract unique days and 30-minute intervals for grid dimensions
        unique_days = sorted(set(dt.date() for dt in datetime_objs))
        night_intervals = [(hour, minute) for hour in range(18, 24) for minute in (0, 30)] + \
                          [(hour, minute) for hour in range(0, 7) for minute in (0, 30)]  # 18:00 to 6:30

        # Initialize a 2D grid for counts with dimensions [night_intervals x days]
        day_night_grid = np.zeros((len(night_intervals), len(unique_days)))

        # Populate the grid with counts, only considering night intervals
        for dt in datetime_objs:
            interval = (dt.hour, 0 if dt.minute < 30 else 30)
            if interval in night_intervals:  # Check if the datetime falls within night intervals
                day_idx = unique_days.index(dt.date())
                interval_idx = night_intervals.index(interval)  # Map the interval to the row in the grid
                # Convert datetime to timestamp integer for key access if necessary
                timestamp_key = dt if dt in self.pointcloud_pixel else int(dt.timestamp())
                day_night_grid[interval_idx, day_idx] += len(self.pointcloud_pixel.get(timestamp_key, []))

        # Plot the heatmap
        cax = ax.imshow(day_night_grid, aspect='auto', cmap='plasma', origin='lower', vmax=200)

        # Set the labels for x-axis (days) and y-axis (night intervals)
        ax.set_xticks(np.arange(len(unique_days)))
        ax.set_xticklabels([day.strftime('%Y-%m-%d') for day in unique_days], rotation=45, ha='right',
                           fontsize=font_size)

        # Set y-axis ticks to every second row
        y_tick_indices = np.arange(0, len(night_intervals), 2)  # Every second row
        y_tick_labels = [f'{hour:02d}:{minute:02d}' for i, (hour, minute) in enumerate(night_intervals) if i % 2 == 0]
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels(y_tick_labels, fontsize=font_size)

        # Move the color bar to the right of the heatmap
        divider = make_axes_locatable(ax)
        cax_cb = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(cax, cax=cax_cb, orientation='vertical', label="Number of Centroids", extend='max')

        plt.tight_layout()
        if figpath != '':
            plt.savefig(figpath)

        plt.show()

    def display_with_cumulative_line(self, size=20, font_size=20, alpha=0.5, figpath=''):
        # Create a figure with 2 subplots: one for the map and one for the cumulative plot
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [12, 5]})
        fig.subplots_adjust(hspace=0.5)

        # Plot the map background
        ax.imshow(self.img, alpha=alpha)  # Display the background map

        # Prepare coordinates and timestamps for coloring
        x_coords = [centroid[0] for centroids in self.pointcloud_pixel.values() for centroid in centroids]
        y_coords = [centroid[1] for centroids in self.pointcloud_pixel.values() for centroid in centroids]
        timestamps = [dt for dt in self.pointcloud_pixel.keys() for _ in self.pointcloud_pixel[dt]]

        # Scatter plot of points with color-coded timestamps
        scatter = ax.scatter(x=x_coords, y=y_coords, c=timestamps, cmap='plasma', marker='.', s=size, alpha=0.75)
        colorbar = fig.colorbar(scatter, ax=ax, orientation="vertical", label="Timestamp")

        # Format the colorbar to show datetime labels
        def timestamp_to_date(x, pos):
            return datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M')

        colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(timestamp_to_date))

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting for the main map
        ax.set_xlabel("Easting [m]", fontsize=font_size)
        ax.set_ylabel("Northing [m]", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        # Plot node locations with a distinctive marker and add to legend
        node_x_coords = [location[0] for location in self.node_locations]
        node_y_coords = [location[1] for location in self.node_locations]
        nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='red', marker='x', s=100, label='Node Locations')

        # Add legend for node locations
        ax.legend(loc="upper right", fontsize=font_size)

        # Prepare cumulative data with modified cumulative_counts calculation
        datetime_objs = [
            datetime.fromtimestamp(ts) if isinstance(ts, int) else ts
            for ts in self.pointcloud_pixel.keys()
        ]
        datetime_objs_sorted = sorted(datetime_objs)
        cumulative_counts = np.cumsum([len(self.pointcloud_pixel[dt.timestamp()]) for dt in datetime_objs_sorted])

        # Plot cumulative line
        ax2.plot(datetime_objs_sorted, cumulative_counts, color='blue', linewidth=2)
        ax2.set_xlabel("Time", fontsize=font_size)
        ax2.set_ylabel("Cumulative Number of Points", fontsize=font_size)

        # Format x-axis to display hour, minute, and second
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax2.tick_params(axis='x', rotation=45)
        ax2.xaxis.set_tick_params(labelsize=font_size)
        ax2.yaxis.set_tick_params(labelsize=font_size)

        plt.tight_layout()
        if figpath != '':
            plt.savefig(figpath)

        plt.show()

    import networkx as nx

    def display_with_voronoi(self, points, node_size=250, font_size=20, alpha=1.0, figpath=''):
        import matplotlib.colors as mcolors
        import matplotlib as mpl
        from scipy.spatial import Voronoi, voronoi_plot_2d
        # Plot the map background and overlay the points
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(self.img, alpha=0.5)  # Display the background map

        points_pixel = []
        for p in points:
            points_pixel.append(self.convert_utm_to_pixel(p[0], p[1]))

        vor = Voronoi(points_pixel)

        # Plot the Voronoi diagram on the same axes
        # Customize line colors, width, etc. as needed
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=3, line_alpha=0.7,
                        point_size=0, linestyles='solid')

        # Retrieve all line collections from the axes and set them to solid
        for artist in ax.get_children():
            if isinstance(artist, mpl.collections.LineCollection):
                artist.set_linestyles('solid')

        # Optionally, plot the points themselves
        x_coords = [p[0] for p in points_pixel]
        y_coords = [p[1] for p in points_pixel]
        ax.scatter(x_coords, y_coords, marker="o", color='lightblue', s=node_size)
        ax.invert_yaxis()  # Flip the y-axis so it increases upwards

        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-434421 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [ -5761732 +
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        # ax.set_ylabel("Northing [m]", fontsize=font_size)
        ax.set_xlabel("UTM 32N Easting [+434421m]", fontsize=font_size)
        ax.set_ylabel("UTM 32N Northing  [+5761732m]", fontsize=font_size)

        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        plt.tight_layout()
        if figpath != '':
            plt.savefig(figpath)

        plt.show()

    def display_with_nodes_colored_by_weight(self, graph, node_size=500, font_size=20, alpha=1.0, figpath=''):
        import matplotlib.colors as mcolors
        # Plot the map background and overlay the points
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(self.img, alpha=0.25)  # Display the background map

        # Draw the graph on the map
        pos = {node: (graph.nodes[node]['pos'][0], graph.nodes[node]['pos'][1]) for node in graph.nodes}
        for key in pos:
            x_pixel, y_pixel = self.convert_utm_to_pixel(pos[key][0], pos[key][1])
            pos[key] = [x_pixel, y_pixel]
        labels = nx.get_node_attributes(graph, "weight")
        for key in labels.keys():
            labels[key] = "." + str(round(labels[key], 3)).split('.')[1]

        # Extract node weights for coloring
        node_weights = [graph.nodes[node]['weight'] for node in graph.nodes]

        cmap = plt.cm.RdYlGn

        # Draw the nodes using their weights as colors
        node_collection = nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_weights,
            cmap=cmap,
            alpha=alpha,
            vmin=0.0,
            vmax=1.0
        )
        # nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=font_size, font_color='black')
        # nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='black', alpha=0.7, width=3)
        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-434421 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [ -5761732 +
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        # ax.set_ylabel("Northing [m]", fontsize=font_size)
        ax.set_xlabel("UTM 32N Easting [+434421m]", fontsize=font_size)
        ax.set_ylabel("UTM 32N Northing  [+5761732m]", fontsize=font_size)

        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        # Create a new axis on the right side of the plot with the same height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and pad as needed

        # Add a colorbar to the right
        cb = plt.colorbar(node_collection, cax=cax)
        cb.set_label("Classifier's probability", fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size)  # Set colorbar tick font size

        plt.tight_layout()

        # Save figure if a path is provided
        if figpath:
            plt.savefig(figpath)
            print(f"Saved to {figpath}")
        plt.show()


    def display_with_graph(self, graph, territorial_subgraphs, node_size=1000, font_size=20, alpha=0.5, figpath='', no_legend=False):
        # Plot the map background and overlay the points
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(self.img, alpha=alpha)  # Display the background map

        # Add the circles of the territory
        territory_circle_different_colors = {}  # used for legend
        for circle in self.center_gt:
            if circle[3] == "red":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label="GT's Territory (safe)")
            elif circle[3] == "orange":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label="GT's Territory (possible)")
            else:
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label="GT's Territory (outside)")

            territory_circle_different_colors[circle[3]] = territory_circle
            plt.gca().add_patch(territory_circle)


        # Calculate and set UTM ticks and labels
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)

        # Add the circles of the territory
        for circle in self.center_gt:
            territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.10,
                                          label='Territory')
            plt.gca().add_patch(territory_circle)

        # Convert pixel positions to UTM coordinates for display
        x_labels = [int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for
            tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        plt.xlabel("Easting [m]", fontsize=font_size)
        plt.ylabel("Northing [m]", fontsize=font_size)

        plt.xlim((min(x_ticks), max(x_ticks)))
        plt.ylim((max(y_ticks), min(y_ticks)))

        # Draw the graph on the map
        pos = {node: (graph.nodes[node]['pos'][0], graph.nodes[node]['pos'][1]) for node in graph.nodes}
        for key in pos:
            x_pixel, y_pixel = self.convert_utm_to_pixel(pos[key][0], pos[key][1])
            pos[key] = [x_pixel, y_pixel]
        labels = nx.get_node_attributes(graph, "weight")
        for key in labels.keys():
            labels[key] = "." + str(round(labels[key], 3)).split('.')[1]
        # labels = nx.get_node_attributes(graph, "node")
        # for key in labels.keys():
        #     labels[key] = "." + str(labels[key].deployment_id)

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-434421 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [ -5761732 +
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        # nx.draw(graph, pos, ax=ax, with_labels=True, node_size=1000, arrowsize=10, font_size=font_size, node_color='lightblue',
        #         labels=labels, width=3, alpha=0.8)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=font_size, font_color='black')
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='black', alpha=0.7, width=3, node_size=node_size)
        # Define a list of colors
        colors = ['yellow', 'g', 'm', 'c', 'b', 'r', 'k']
        centroids = []

        if len(territorial_subgraphs) != 0:
            # Step 1: Initialize a dictionary to count node occurrences
            node_occurrences = {}

            # Step 2: Iterate over each cluster and count nodes
            for cluster_info in territorial_subgraphs.values():
                cluster_graph = cluster_info['TS']
                for node in cluster_graph.nodes():
                    if node in node_occurrences:
                        node_occurrences[node] += 1
                    else:
                        node_occurrences[node] = 1

            # Step 3: Get nodes that are part of more than one cluster
            nodes_in_multiple_clusters = [node for node, count in node_occurrences.items() if count >= 2]

            # Step 4: Draw nodes that are part of multiple clusters with dark gray color
            nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=nodes_in_multiple_clusters, node_size=1000, node_color='darkgray')

            # Draw other nodes and edges
            for ctr, i in enumerate(territorial_subgraphs):
                cluster_graph = territorial_subgraphs[i]['TS']
                cluster_nodes = [node for node in cluster_graph.nodes() if node not in nodes_in_multiple_clusters]
                nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=cluster_nodes, node_size=1000, node_color=colors[ctr])
                # nx.draw_networkx_edges(G, pos, edgelist=cluster_graph.edges())

            # Assuming you have a method to compute centroids
            for ctr, i in enumerate(territorial_subgraphs):
                cluster_graph = territorial_subgraphs[i]['TS']
                centroid = calc_geometric_center_in_Graph(G=cluster_graph, cluster_nodes=cluster_graph.nodes(),
                                                               weighted=True)
                centroids.append(self.convert_utm_to_pixel(centroid[0], centroid[1]))


            x = [c[0] for c in centroids]
            y = [c[1] for c in centroids]
            # Plot each point with its respective color
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.plot(xi, yi,  # Plot individual points (xi, yi)
                         marker='P',  # Plus marker (filled)
                         markersize=20,  # Marker size
                         markerfacecolor=colors[i],  # Fill color
                         markeredgecolor='black',  # Edge color
                         markeredgewidth=1,  # Edge line thickness
                         linestyle='none')  # No connecting line


        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        # ax.set_ylabel("Northing [m]", fontsize=font_size)
        ax.set_xlabel("UTM 32N Easting [+434,421m]", fontsize=font_size)
        ax.set_ylabel("UTM 32N Northing  [+576,173,2m]", fontsize=font_size)

        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        if not no_legend:
            # Create legend elements for clusters
            legend_elements = []
            legend_elements += [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)],
                       markersize=20, label=f'TS {i + 1}') for i in range(len(territorial_subgraphs))
            ]
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray',
                                          markersize=20, label='Part of multiple TS'))
            legend_elements.append(Line2D([0], [0], marker='P', color='w', markeredgecolor='black',
                                          markersize=15, label="Centroid of TS"))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                                          markersize=20, label='Not assigned to TS'))

            # Add legend for territory circles
            try:
                # Assuming territory_circle_different_colors is a dictionary with labels as keys and plot objects as values
                territory_handles = list(territory_circle_different_colors.values())
                territory_labels = [x.get_label() for x in territory_circle_different_colors.values()]

                # Add territory elements to the overall legend
                for handle, label in zip(territory_handles, territory_labels):
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=handle.get_facecolor(),
                                                  markersize=20, label=label))

                # Create a combined legend
                plt.legend(handles=legend_elements, loc='lower left', fontsize=font_size, ncol=2,
                           handleheight=0.7, labelspacing=0.0125, columnspacing=0.4)
            except UnboundLocalError:
                # Handle the case where territory_circle_different_colors is not defined
                plt.legend(handles=legend_elements, loc='lower left', fontsize=font_size, ncol=2,
                           handleheight=0.7, labelspacing=0.0125, columnspacing=0.4)
        plt.box(True)
        plt.tight_layout()

        # Save figure if a path is provided
        if figpath:
            plt.savefig(figpath)
            print(f"Saved to {figpath}")
        plt.show()


    def display_with_pointOverTime(self, size=20, font_size=20, alpha=0.5, figpath='', bin_size=900):
        # Define colormap based on timestamps
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(vmin=0, vmax=3600)

        # Create a figure with two subplots: one for the map and one for the bar plot
        fig, (ax_map, ax_activity) = plt.subplots(2, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Plot the map background and overlay the points on the first subplot (ax_map)
        ax_map.imshow(self.img, alpha=alpha)  # Display the background map

        # Calculate and set UTM ticks and labels
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)

        # Add the circles of the territory
        territory_circle_different_colors = {}  # used for legend
        for circle in self.center_gt:
            if circle[3] == "red":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label='assured')
            elif circle[3] == "orange":
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label='unassured')
            else:
                territory_circle = plt.Circle((circle[0], circle[1]), circle[2], color=circle[3], alpha=0.2,
                                              label='outside')

        # Scatter plot of points with color-coded timestamps
        scatter = ax_map.scatter(x=self.x_loc, y=self.y_loc, c=self.timestamps_seconds, marker='.', s=size, vmin=0,
                                 vmax=3600 * 6,
                                 cmap=cmap, alpha=0.75)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_map, label='Time of day', ticks=np.arange(0, 3600 * 6 + 1, 3600))
        cbar.ax.set_ylabel('Time of day', fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

        # Custom y-tick labels for colorbar
        labels = [s_to_time(ts) for ts in np.arange(0, 3600 * 6 + 1, 3600)]
        cbar.ax.set_yticklabels(labels, fontsize=font_size)

        # Convert pixel positions to UTM coordinates for display
        x_labels = [int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax_map.set_xticks(x_ticks)
        ax_map.set_xticklabels(x_labels, fontsize=font_size)
        ax_map.set_yticks(y_ticks)
        ax_map.set_yticklabels(y_labels, fontsize=font_size)

        # Additional formatting
        ax_map.set_xlabel("Easting [m]", fontsize=font_size)
        ax_map.set_ylabel("Northing [m]", fontsize=font_size)
        ax_map.set_xlim((min(x_ticks), max(x_ticks)))
        ax_map.set_ylim((max(y_ticks), min(y_ticks)))

        # Add the legend
        try:
            ax_map.legend(handles=territory_circle_different_colors.values(), loc="upper right", fontsize=font_size,
                          title_fontsize=font_size, title="Ground Truth")
        except UnboundLocalError:
            pass

        # Bin the timestamps and count occurrences in each bin for activity plot
        bins = np.arange(0, 21600 + bin_size, bin_size)  # Bins from 0 to 21600 in steps of bin_size
        counts, _ = np.histogram(self.timestamps_seconds, bins=bins)

        # Plot the number of points per time interval on the second subplot (ax_activity)
        ax_activity.bar(bins[:-1], counts, width=bin_size, align='edge')
        ax_activity.set_xlim(0, 21600)
        ax_activity.set_xlabel('Time (hours)', fontsize=font_size)
        ax_activity.set_ylim(0, 800)
        ax_activity.set_ylabel('Number of Points', fontsize=font_size)
        ax_activity.tick_params(axis='both', which='major', labelsize=font_size)

        # Custom x-axis labels from 4 to 10 (for the range 0 to 21600 seconds)
        hour_labels = [4 + (i * 1) for i in range(7)]  # Generate labels 4, 5, 6, ..., 10
        ax_activity.set_xticks(np.linspace(0, 21600, 7))  # Set ticks to match labels
        ax_activity.set_xticklabels(hour_labels)

        # Tight layout for both subplots
        plt.tight_layout()

        # Save figure if figpath is specified
        if figpath != '':
            plt.savefig(figpath)

        # Display the figure
        plt.show()


    def add_point_from_utm(self, utm_x, utm_y, zone_number, zone_letter, color='red', size=10):
        """
        Adds a point to the map based on UTM coordinates.

        Parameters:
        - utm_x, utm_y: float - UTM coordinates of the point.
        - zone_number: int - UTM zone number.
        - zone_letter: str - UTM zone letter.
        - color: str - Color of the point.
        - size: int - Size of the point marker.
        """
        proj_utm = Proj(proj="utm", zone=zone_number, south=(zone_letter >= 'U'))
        proj_wgs84 = Proj(proj="longlat", datum="WGS84")
        lon, lat = transform(proj_utm, proj_wgs84, utm_x, utm_y)

        # Convert lat/lon to image pixel coordinates within the bounding box
        x_pixel = int((lon - self.bbox[0]) / (self.bbox[2] - self.bbox[0]) * self.img_width)
        y_pixel = int((self.bbox[3] - lat) / (self.bbox[3] - self.bbox[1]) * self.img_height)

        # Plot the map if not already displayed
        self.show_map()

        # Plot the point on the map image
        plt.plot(x_pixel, y_pixel, 'o', color=color, markersize=size)

    def convert_latlon_to_pixel(self, lat, lon, zone_number=32, zone_letter='N'):
        # Convert lon/lat to image pixel coordinates
        x_pixel = int((lon - self.bbox[0]) / (self.bbox[2] - self.bbox[0]) * self.img_width)
        y_pixel = int((self.bbox[3] - lat) / (self.bbox[3] - self.bbox[1]) * self.img_height)
        return x_pixel, y_pixel

    def convert_utm_to_pixel(self, utm_x, utm_y, zone_number=32, zone_letter='N'):
        # Define projection systems
        proj_utm = Proj(proj="utm", zone=zone_number, south=(zone_letter < 'N'))
        proj_wgs84 = Proj(proj="longlat", datum="WGS84")

        # Create a transformer
        transformer = Transformer.from_proj(proj_utm, proj_wgs84, always_xy=True)

        try:
            # Perform the coordinate transformation
            lon, lat = transformer.transform(utm_x, utm_y)
        except Exception as e:
            print(f"Error during transformation: {e}")
            return None  # Handle the error appropriately

        # Convert lon/lat to image pixel coordinates
        x_pixel = int((lon - self.bbox[0]) / (self.bbox[2] - self.bbox[0]) * self.img_width)
        y_pixel = int((self.bbox[3] - lat) / (self.bbox[3] - self.bbox[1]) * self.img_height)

        return [x_pixel, y_pixel]

    def convert_pointcloudUTM_2_pointcloudPIXEL(self, points_xy, timestamps, zone_number=32, zone_letter='N'):
        if self.img is None:
            raise ValueError("Map image not loaded. Call fetch_map() first.")

        # Convert UTM points to pixel coordinates within the map
        self.pointcloud_pixel = {}
        for timestamp, centroid in zip(timestamps, points_xy):
            # print(timestamp, centroid)
            try:
                self.pointcloud_pixel[timestamp].append(self.convert_utm_to_pixel(centroid[0], centroid[1], zone_number=zone_number, zone_letter=zone_letter))
            except KeyError:
                self.pointcloud_pixel[timestamp] = [self.convert_utm_to_pixel(centroid[0], centroid[1], zone_number=zone_number, zone_letter=zone_letter)]
