import pandas as pd
import numpy as np
import pdb
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

PIXEL_PITCH = 5.86 * 1e-6
SENSOR_SIZE = 200
WORKING_RANGE = np.linspace(0.30, 2.70, 81)
LAPLACIAN_KERNEL = np.array([-1, 2, -1]) / (PIXEL_PITCH**2)
HEATMAP_RANGE = [
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
]


def plotSingleResult(Z_est, Ztrue, pathname, title=None, normalisation=False):

    fig = plt.figure(figsize=(10, 10), dpi=100)

    ax = fig.add_subplot(1, 1, 1)
    heatmap, xedges, yedges = np.histogram2d(
        Ztrue.flatten(), Z_est.flatten(), bins=81, range=HEATMAP_RANGE
    )
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if normalisation:
        heatmap = heatmap / np.where(
            heatmap.max(axis=0) == 0, 1, np.nansum(heatmap, axis=0)
        )
    plot_heatmap = ax.imshow(heatmap, extent=extent, origin="lower")
    fig.colorbar(plot_heatmap, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Estimated Depth (m)")
    if title is not None:
        ax.set_title(title)

    ax.grid()
    fig.tight_layout()
    plt.savefig(pathname)
    plt.close(fig)

    return


def build_getTransmission():
    # Only read Excel and build interpolator ONCE
    excel_path = "./data/Map2.xlsx"
    sheet_name = 0
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    angle_values = np.linspace(19.9, 0, num=df.shape[0])
    d_values = np.linspace(12.5, 150, num=df.shape[1])
    transmission_grid = df.to_numpy(dtype=float)

    interpolator = RegularGridInterpolator(
        (angle_values, d_values), transmission_grid, bounds_error=False, fill_value=None
    )

    def getTransmission(angle, d):
        angle = np.asarray(angle)
        d = np.asarray(d)
        orig_shape = angle.shape  # Could be (), (N,), or (M,N)
        # If angle and d have same shape, flatten for interpolation
        points = np.column_stack([angle.ravel(), d.ravel()])
        result = interpolator(points)
        result = result.reshape(orig_shape)
        if result.shape == ():
            return float(result)
        return result

    return getTransmission


def estimate_theta_vector(
    images, metasurface_distances, getTransmission, theta_min=0, theta_max=5
):
    M, N = images.shape
    theta_vector = np.zeros(M, dtype=float)
    for i in range(M):
        image_row = images[i, :]
        # Normalize to remove unknown intensity
        image_row_norm = image_row / np.max(image_row)

        def cost(theta):
            t_vec = getTransmission(
                np.full_like(metasurface_distances, theta, dtype=float),
                metasurface_distances,
            )
            t_vec_norm = t_vec / np.max(t_vec)
            return np.sum((image_row_norm - t_vec_norm) ** 2)

        res = minimize_scalar(cost, bounds=(theta_min, theta_max), method="bounded")
        theta_vector[i] = res.x if res.success else np.nan
    return theta_vector


def simulation_mmdp():
    metasurface_distances = np.linspace(12.5, 150, 30)
    # 1 mm for each pixel of the texture
    full_frame_size = 0.0351  # 35 mm
    sensor_dimension = 351
    pixel_pitch = full_frame_size / sensor_dimension
    point_source_intensity = 255
    getTransmission = build_getTransmission()

    # Texture is a point source

    sensor_coordinates = np.zeros((sensor_dimension, 2))
    sensor_coordinates[:, 1] = (
        np.arange(sensor_dimension) - sensor_dimension // 2
    ) * pixel_pitch

    Z_est_list = []
    Z_true_list = []

    for depth in WORKING_RANGE:
        print(f"Simulating for depth: {depth:.2f} m")
        y_diff = np.abs(sensor_coordinates[:, 1])
        theta = np.rad2deg(np.arctan2(y_diff, depth))
        light_field = np.array(
            [
                np.full(sensor_dimension, depth),
                np.full(sensor_dimension, 0),  # x-coordinate is zero for point source
                theta,
            ]
        ).T

        theta_grid, d_grid = np.meshgrid(theta, metasurface_distances, indexing="ij")
        transmission = getTransmission(theta_grid, d_grid)
        images = point_source_intensity * transmission

        est_theta = estimate_theta_vector(
            images, metasurface_distances, getTransmission, theta_min=0, theta_max=5
        )

        # Print error, real theta, and estimated theta
        # error = np.mean(np.abs(theta_vector - theta))
        # print(error)
        # print(f"Real theta: {theta}, Estimated theta: {theta_vector}")

        theta_rad = np.deg2rad(est_theta)
        z_est = np.where(
            np.abs(np.tan(theta_rad)) > 1e-6,
            np.abs(sensor_coordinates[:, 1]) / np.tan(theta_rad),
            np.nan,
        )

        z_true = np.full_like(z_est, depth)
        Z_est_list.append(z_est)
        Z_true_list.append(z_true)

    plotSingleResult(
        np.array(Z_est_list),
        np.array(Z_true_list),
        "./mmdp.png",
        title="MMDP Optimization",
    )


def main():
    # Test the getTransmission function
    # getTransmission = build_getTransmission()
    # angle = [0, 5, 10.0]
    # d = [10, 20, 30]
    # transmission = getTransmission(angle, d)
    # print(f"Transmission for angle {angle} and distance {d}: {transmission}")

    simulation_mmdp()

    return


if __name__ == "__main__":
    main()
