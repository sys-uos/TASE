import datetime
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from owslib.wms import WebMapService
import matplotlib.pyplot as plt
from PIL import Image
import io
from pyproj import Proj, transform, Transformer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


class WMSMapViewer:

    def __init__(self, wms_service=None, layer_name=None, bbox=None, img_width=700):
        """
        Initializes a Web Map Service (WMS) viewer with a specified bounding box,
        layer, and image resolution.

        Parameters:
        -----------
        wms_service : dict, optional
            Dictionary containing WMS service configuration. Default is:
            - `url`: "https://ows.terrestris.de/osm/service?"
            - `version`: "1.3.0"
            - `layername`: "OSM-WMS"

        layer_name : str, optional
            The WMS layer to be used. Defaults to the `layername` in `wms_service`.

        bbox : tuple, optional
            The bounding box for the map in (min_lon, min_lat, max_lon, max_lat) format.
            Defaults to (8.05214167, 52.00720153, 8.05751681, 52.01278839).

        img_width : int, optional
            The width of the requested image in pixels. Default is 700.
            The height is automatically calculated based on the aspect ratio.

        Attributes:
        -----------
        wms : WebMapService
            An instance of `WebMapService` for retrieving map tiles.

        layer_name : str
            The selected layer name.

        bbox : tuple
            The bounding box including an additional margin for better visualization.

        aspect_ratio : float
            The aspect ratio of the bounding box (width/height).

        img_width : int
            The width of the map image in pixels.

        img_height : int
            The height of the map image in pixels, calculated from the aspect ratio.

        img : NoneType
            Placeholder for storing the fetched map image.

        bbox_utm : tuple
            The bounding box converted to UTM coordinates (with an additional margin).

        node_locations : list
            A list of node locations in the map.

        pointcloud_pixel : dict
            A dictionary for storing the converted point cloud in pixel coordinates.

        Notes:
        ------
        - The bounding box is extended by 5% margin for better visualization.
        - The UTM bounding box is further extended by 500 meters to ensure coverage.
        - The `Transformer` is used for coordinate conversion between WGS84 and UTM.
        """
        if wms_service is None:
            wms_service = {"url": "https://ows.terrestris.de/osm/service?", "version": "1.3.0", "layername": "OSM-WMS"}
        self.wms = WebMapService(wms_service["url"], version=wms_service["version"])
        self.layer_name = layer_name if layer_name != None else wms_service["layername"]

        left, bottom, right, top = bbox if bbox is not None else (8.05214167, 52.00720153, 8.05751681, 52.01278839)

        margin_factor = 0.05
        width_margin = (right - left) * margin_factor
        height_margin = (top - bottom) * margin_factor
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
        self.aspect_ratio = width / height

        self.img_width = img_width
        self.img_height = int(img_width / self.aspect_ratio)  # Calculate height based on the aspect ratio
        self.img = None

        # Define projections: WGS84 (lat/lon) and UTM (e.g. zone 32N for Germany)
        proj_wgs84 = Proj(proj="latlong", datum="WGS84")
        proj_utm = Proj(proj="utm", zone=32, south=False)
        transformer = Transformer.from_proj(proj_wgs84, proj_utm, always_xy=True)
        # Convert each corner of the bounding box to UTM
        min_easting, min_northing = transformer.transform(self.bbox[0], self.bbox[1])
        max_easting, max_northing = transformer.transform(self.bbox[2], self.bbox[3])
        # Define the UTM bounding box
        margin_within_margin = 500  # in meters
        self.bbox_utm = (min_easting - margin_within_margin, min_northing - margin_within_margin, max_easting + margin_within_margin, max_northing + margin_within_margin)

        # Variables where the visualized data is stored
        self.node_locations = []
        self.pointcloud_pixel = {}

        # Fetch and display the map
        self.__fetch_map()


    def __fetch_map(self, save_path="./wms_map"):
        """
        Fetches a map image from the WMS service and saves it to the specified location.
        If the map image already exists at the location, it will not be fetched again.

        :param save_path: The path where the map image should be saved.
        """
        try:
            # Check if the map image already exists
            if os.path.exists(save_path):
                print(f"Map image already exists at '{save_path}'.")
                self.img = Image.open(save_path)
                print("Loaded existing map image.")
                return

            # Fetch the map image from the WMS service
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

            # Save the image to the specified location
            self.img.save(save_path, format="PNG")
            print(f"Map image saved to '{save_path}'.")

        except Exception as e:
            print(f"Error fetching or saving map image: {e}")


    def show_map(self, alpha=0.25):
        # Display the map image using matplotlib
        if self.img is None:
            raise ValueError("Map image not fetched. Call fetch_map() first.")

        plt.imshow(self.img, alpha=alpha)


    def add_node_locations(self, node_locations, zone_number=33, zone_letter='N'):
        for elem in node_locations:
            self.node_locations.append(self.__convert_utm_to_pixel(elem.lat, elem.lon, zone_number=zone_number, zone_letter=zone_letter))


    def add_and_convert_pointcloudUTM_2_pointcloudPIXEL(self, points_xy, zone_number=32, zone_letter='N'):
        if self.img is None:
            raise ValueError("Map image not loaded. Call fetch_map() first.")

        # Convert UTM points to pixel coordinates within the map
        self.pointcloud_pixel = {}
        for timestamp in points_xy:
            for centroid in points_xy[timestamp]:
                try:
                    self.pointcloud_pixel[timestamp].append(self.__convert_utm_to_pixel(centroid[0], centroid[1], zone_number=zone_number, zone_letter=zone_letter))
                except KeyError:
                    self.pointcloud_pixel[timestamp] = []
                    self.pointcloud_pixel[timestamp].append(self.__convert_utm_to_pixel(centroid[0], centroid[1], zone_number=zone_number, zone_letter=zone_letter))


    def __convert_utm_to_pixel(self, utm_x, utm_y, zone_number=32, zone_letter='N'):
        proj_utm = Proj(proj="utm", zone=zone_number, south=(zone_letter < 'N'))
        proj_wgs84 = Proj(proj="longlat", datum="WGS84")
        transformer = Transformer.from_proj(proj_utm, proj_wgs84, always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)

        # Convert lon/lat to image pixel coordinates
        x_pixel = int((lon - self.bbox[0]) / (self.bbox[2] - self.bbox[0]) * self.img_width)
        y_pixel = int((self.bbox[3] - lat) / (self.bbox[3] - self.bbox[1]) * self.img_height)

        return [x_pixel, y_pixel]


    def display_with_heatmap(self,kde_bw_param=0.2,  size=20, font_size=20, alpha=0.5, figpath='',  vmax=0.0001):
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, int(10/self.aspect_ratio)))

        # Extract x and y coordinates from the point cloud
        x_coords = [centroid[0] for centroids in self.pointcloud_pixel.values() for centroid in centroids]
        y_coords = [centroid[1] for centroids in self.pointcloud_pixel.values() for centroid in centroids]

        # Check if there are any coordinates to plot
        if not x_coords or not y_coords:
            print("No points to display.")
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
        try:
            # Calculate the kernel density estimate
            xy = np.vstack([x_coords, y_coords])
            kde = gaussian_kde(xy, bw_method=kde_bw_param)  # Adjust the bandwidth for smoothing; lower value = sharper peaks

            # Create a grid over the image for evaluating the KDE
            x_grid = np.linspace(0, self.img_width, 500)
            y_grid = np.linspace(0, self.img_height, 500)
            x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
            z = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)
            z = np.clip(z, a_min=0, a_max=None) # Clip the output to ensure all densities are non-negative:

            img = None
            if not z.max() == 0: # in case too few points are available, all values in z are 0. then the img gets corrupted
                # Define a custom colormap that starts with white and transitions to red (or any color desired)
                white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "green"])

                # Display the KDE heatmap with the custom colormap
                img = ax.imshow(z, extent=(0, self.img_width, 0, self.img_height), origin='lower',
                                cmap=white_to_red, alpha=1.0, vmin=0, vmax=vmax)
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

        # Create an axis on the right for the colorbar with matching height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if img is not None:
            # Create the colorbar using the image
            cbar = fig.colorbar(img, cax=cax, extend="max")
            cbar.set_label("TASE' Point Density", fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size)
        else:
            # Define a custom colormap from white to red
            white_to_red = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])

            # Create an empty colorbar (placeholder)
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=white_to_red), cax=cax,  extend="max")
            cbar.set_label("TASE' Point Density", fontsize=font_size)  # No label
            cbar.ax.tick_params(labelsize=font_size)

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-584212 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0])) for tick in
                    x_ticks]
        y_labels = [ -5211754 +
            int(self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1]))
            for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)
        # Rotate y-axis labels
        for label in ax.get_yticklabels():
            label.set_rotation(0)

        # Additional formatting
        ax.set_xlabel("UTM 33N Easting [+584212m]", fontsize=font_size)
        ax.set_ylabel("UTM 33N Northing [+5211754m]", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        # Plot node locations with a distinctive marker and add to legend
        node_x_coords = [location[0] for location in self.node_locations]
        node_y_coords = [location[1] for location in self.node_locations]
        nodes_scatter = ax.scatter(node_x_coords, node_y_coords, color='black', marker='x', s=25, label='Recorders')

        # Create a custom legend entry for the number of points
        num_points_legend = mpatches.Patch(color='none', label=f'Number of points: {len(x_coords)}')

        # Add the custom legend entry along with existing ones
        ax.legend(
            handles=[num_points_legend, *ax.get_legend_handles_labels()[0]],  # Add the new entry and existing handles
            loc="upper right",
            fontsize=font_size
        )

        plt.tight_layout()

        # Save figure if a path is provided
        if figpath:
            plt.savefig(figpath)
        plt.show()

    def display_with_pointcloud(self, point_size=20, font_size=20, alpha=0.5, figpath=''):
        """
        Display the point cloud on the map, color-coding points based on timestamps
        and adding a corresponding colorbar.

        Parameters:
        -----------
        size : int, optional (default=20)
            Size of the scatter points.
        font_size : int, optional (default=20)
            Font size for axis labels and legend.
        alpha : float, optional (default=0.5)
            Transparency of the background map image.
        figpath : str, optional (default='')
            Path to save the generated figure. If empty, it will be displayed instead.
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, int(10 / self.aspect_ratio)))

        # Extract x, y coordinates and timestamps from the point cloud
        x_coords = []
        y_coords = []
        timestamps = []

        for ts, centroids in self.pointcloud_pixel.items():
            for centroid in centroids:
                x_coords.append(centroid[0])
                y_coords.append(centroid[1])
                timestamps.append(ts)  # Store the timestamp for coloring

        # Check if there are any coordinates to plot
        if not x_coords or not y_coords:
            print("No points to display.")
            ax.text(
                0.5, 0.5,
                "No point cloud data available.",
                transform=ax.transAxes,
                fontsize=font_size,
                color='black',
                ha='center',
                va='center',
                fontstyle='italic',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
            )
            return

        # Normalize timestamps for coloring
        norm = Normalize(vmin=min(timestamps), vmax=max(timestamps))
        cmap = plt.get_cmap("viridis")  # Use Viridis colormap for time-based coloring
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Scatter plot of point cloud, colored by timestamps
        scatter = ax.scatter(x_coords, y_coords, c=timestamps, cmap=cmap, norm=norm, s=point_size, edgecolors='k', alpha=0.7)

        # Plot the map background image
        ax.imshow(self.img, alpha=alpha)

        # Create an axis on the right for the colorbar with matching height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax, extend="max")

        # Convert min and max timestamps to HH:MM:SS format
        min_time = datetime.datetime.fromtimestamp(min(timestamps)).strftime('%H:%M:%S')
        max_time = datetime.datetime.fromtimestamp(max(timestamps)).strftime('%H:%M:%S')

        # Set the colorbar label
        cbar.set_label("Time", fontsize=font_size)

        # Set custom tick labels with HH:MM:SS format
        tick_locs = np.linspace(min(timestamps), max(timestamps), num=5)
        tick_labels = [datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S') for ts in tick_locs]
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=font_size)

        # Convert pixel positions to UTM coordinates for display
        x_ticks = np.linspace(0, self.img_width, num=5)
        y_ticks = np.linspace(0, self.img_height, num=5)
        x_labels = [-584212 + int(self.bbox_utm[0] + (tick / self.img_width) * (self.bbox_utm[2] - self.bbox_utm[0]))
                    for tick in x_ticks]
        y_labels = [-5211754 + int(
            self.bbox_utm[1] + ((self.img_height - tick) / self.img_height) * (self.bbox_utm[3] - self.bbox_utm[1])) for
                    tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=font_size)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=font_size)

        # Rotate y-axis labels
        for label in ax.get_yticklabels():
            label.set_rotation(0)

        # Additional formatting
        ax.set_xlabel("UTM 33N Easting [+584212m]", fontsize=font_size)
        ax.set_ylabel("UTM 33N Northing [+5211754m]", fontsize=font_size)
        ax.set_xlim((min(x_ticks), max(x_ticks)))
        ax.set_ylim((max(y_ticks), min(y_ticks)))

        # Plot node locations with a distinctive marker
        node_x_coords = [location[0] for location in self.node_locations]
        node_y_coords = [location[1] for location in self.node_locations]
        ax.scatter(node_x_coords, node_y_coords, color='black', marker='x', s=25, label='Recorders')

        # Create a legend entry for the number of points
        num_points_legend = mpatches.Patch(color='none', label=f'Number of points: {len(x_coords)}')

        # Add the legend
        ax.legend(
            handles=[num_points_legend, *ax.get_legend_handles_labels()[0]],
            loc="upper right",
            fontsize=font_size
        )

        plt.tight_layout()

        # Save figure if a path is provided, otherwise show
        if figpath:
            plt.savefig(figpath)
        plt.show()