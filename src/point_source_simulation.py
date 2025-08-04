import pandas as pd
import numpy as np
import pdb
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipy.ndimage import shift as sp_shift
import matplotlib.pyplot as plt
import os
import matplotlib

PIXEL_PITCH = 5.86 * 1e-6
SENSOR_SIZE = 200
# WORKING_RANGE = np.linspace(0.30, 2.70, 81)
# WORKING_RANGE = np.linspace(1000, 10000, 91)
WORKING_RANGE = np.linspace(5, 505, 100)
LAPLACIAN_KERNEL = np.array([-1, 2, -1]) / (PIXEL_PITCH**2)
HEATMAP_RANGE = [
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
]

def set_matplotlib_params():
    matplotlib.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
        "figure.figsize": (10, 10),
    })

set_matplotlib_params()

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


def simulation_mmdp_hack():
    metasurface_distances = np.linspace(12.5, 150, 30)
    # 1 mm for each pixel of the texture
    full_frame_size = 0.0351  # 35 mm
    sensor_dimension = 351
    pixel_pitch = full_frame_size / sensor_dimension
    point_source_intensity = 255
    getTransmission = build_getTransmission()
    photon_per_brightness_level = 10000
    stdNoise = 1 / np.sqrt(photon_per_brightness_level)
    num_repeats = 100

    # Texture is a point source

    sensor_coordinates = np.zeros((sensor_dimension, 2))
    sensor_coordinates[:, 1] = (
        np.arange(sensor_dimension) - sensor_dimension // 2
    ) * pixel_pitch

    Z_est_list = []
    Z_true_list = []

    if os.path.exists("./preCal_images.npy"):
        preCal_images = np.load("./preCal_images.npy")
    else:
        preCal_images = []

        for depth in WORKING_RANGE:
            print(f"Simulating for depth: {depth:.2f} m")
            y_diff = np.abs(sensor_coordinates[:, 1])
            theta = np.rad2deg(np.arctan2(y_diff, depth))

            theta_grid, d_grid = np.meshgrid(
                theta, metasurface_distances, indexing="ij"
            )
            transmission = getTransmission(theta_grid, d_grid)
            images = point_source_intensity * transmission

            preCal_images.append(np.array(images).flatten())

        preCal_images = np.array(preCal_images)
        np.save("./preCal_images.npy", preCal_images)

    mean_error = []
    for depth in WORKING_RANGE:
        y_diff = np.abs(sensor_coordinates[:, 1])
        theta = np.rad2deg(np.arctan2(y_diff, depth))

        theta_grid, d_grid = np.meshgrid(theta, metasurface_distances, indexing="ij")
        transmission = getTransmission(theta_grid, d_grid)
        images = point_source_intensity * transmission
        cleam_images = np.array(images).flatten()

        error = []
        for trial in range(num_repeats):
            noise = (
                np.random.normal(size=cleam_images.shape) * stdNoise * np.sqrt(abs(cleam_images))
            )
            noise_images = cleam_images + noise

            diffs = preCal_images - noise_images[None, :]
            costs = np.sum(diffs**2, axis=1)

            best_idx = np.argmin(costs)
            Z_est = WORKING_RANGE[best_idx]
            error.append(np.abs(Z_est - depth))

            Z_est_list.append(Z_est)
            Z_true_list.append(depth)
        mean_error.append(np.mean(error))

    for error_rates in np.linspace(0.01, 0.10, 10):
        range_length = np.count_nonzero(np.array(mean_error) / np.array(WORKING_RANGE) < error_rates) * (WORKING_RANGE[1] - WORKING_RANGE[0])
        print(f"Effective range for error rate {error_rates}: {range_length:.2f} m")

    plotSingleResult(
        np.array(Z_est_list),
        np.array(Z_true_list),
        "./mmdp.png",
        title="",
    )


def integer_shift_fft(a, b):
    """
    Returns the integer shift that best aligns b to a.
    Positive δ means b is shifted right relative to a.
    """
    # Make sure both are the same length
    n = len(a)
    # FFTs
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    # Cross‐power spectrum
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12  # normalize to unit magnitude to do phase‐only
    # Inverse FFT to get correlation
    corr = np.fft.ifft(R)
    # Find peak
    shift = np.argmax(np.abs(corr))
    # Wrap around into [−n/2, +n/2)
    if shift > n//2:
        shift -= n
    return shift

def simulation_stereo(params):
    sensorDistance = params["sensorDistance"]
    baseline = 2 * params["Sigma"]  # distance between the two cameras
    photon_per_brightness_level = params["photon_per_brightness_level"]
    stdNoise = 1 / np.sqrt(photon_per_brightness_level)
    full_frame_size = 0.0351  # 35 mm
    sensor_dimension = 351
    pixel_pitch = full_frame_size / sensor_dimension
    sensor = np.zeros(sensor_dimension)
    sensor[sensor_dimension // 2] = 1  # Point source at the center

    Z_est_list = []
    Z_true_list = []

    for depth in WORKING_RANGE:
        # Ground-truth disparity
        disparity_m = sensorDistance * baseline / depth
        disparity_px = disparity_m / pixel_pitch

        camera_1 = sensor.copy()
        camera_2 = sp_shift(sensor, disparity_px, order=1, mode='constant', cval=0.0)

        noise = (
                np.random.normal(size=camera_1.shape) * stdNoise * np.sqrt(abs(camera_1))
            )
        camera_1_noisy = camera_1 + noise
        noise = (
                np.random.normal(size=camera_2.shape) * stdNoise * np.sqrt(abs(camera_2))
            )
        camera_2_noisy = camera_2 + noise

        shift = integer_shift_fft(camera_1_noisy, camera_2_noisy)
        print(f"Estimated shift: {shift}, True disparity (px): {disparity_px:.2f}")

        # Convert shift to depth
        depth_est = sensorDistance * baseline / (shift * pixel_pitch)
        Z_est_list.append(depth_est)
        Z_true_list.append(depth)
    
    Z_est_list = np.array(Z_est_list)
    Z_true_list = np.array(Z_true_list)

    plotSingleResult(
        np.array(Z_est_list),
        np.array(Z_true_list),
        "./stereo.png",
        title="Stereo Simulation",
    )


def gaussian_1d(length, sigma, oversampling=1, normalize=True):
    """
    Generate a 1D Gaussian vector of given length, using oversampling for accuracy.
    The output is always 'length', regardless of oversampling.
    """
    if oversampling == 1:
        # Standard case
        x = np.arange(length) - (length - 1) / 2
        g = np.exp(-0.5 * (x / sigma) ** 2)
    else:
        total_len = length * oversampling
        center = (total_len - 1) / 2
        x_hr = np.arange(total_len) - center
        x_hr = x_hr / oversampling
        g_hr = np.exp(-0.5 * (x_hr / sigma) ** 2)
        # Downsample: average each oversampling block
        g = g_hr.reshape(length, oversampling).sum(axis=1)
    if normalize:
        g /= g.sum()
    return g


def simulation_stereo_hack(params):
    sensorDistance = params["sensorDistance"]
    baseline = 2 * params["Sigma"]  # distance between the two cameras
    photon_per_brightness_level = params["photon_per_brightness_level"]
    stdNoise = 1 / np.sqrt(photon_per_brightness_level)
    full_frame_size = 0.0351  # 35 mm
    sensor_dimension = 351
    num_repeats = 100
    pixel_pitch = full_frame_size / sensor_dimension
    sensor = np.zeros(sensor_dimension)
    sensor[sensor_dimension // 2] = 1  # Point source at the center
    gaussian_psf = gaussian_1d(
        sensor_dimension, 10, oversampling=5, normalize=True)
    sensor = np.convolve(sensor, gaussian_psf, mode='same')

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    ax.plot(sensor, label='Image')
    plt.show()

    Z_est_list = []
    Z_true_list = []

    if os.path.exists("./preCal_images_stereo.npy"):
        preCal_images = np.load("./preCal_images_stereo.npy")
    else:
        preCal_images = []

        for depth in WORKING_RANGE:
            print(f"Simulating for depth: {depth:.2f} m")
            # Ground-truth disparity
            disparity_m = sensorDistance * baseline / depth
            disparity_px = disparity_m / pixel_pitch

            camera_1 = sensor.copy()
            camera_2 = sp_shift(sensor, disparity_px, order=1, mode='constant', cval=0.0)

            preCal_images.append(np.array([camera_1, camera_2]).flatten())

        preCal_images = np.array(preCal_images)
        np.save("./preCal_images_stereo.npy", preCal_images)

    mean_error = []
    for depth in WORKING_RANGE:
        # Ground-truth disparity
        disparity_m = sensorDistance * baseline / depth
        disparity_px = disparity_m / pixel_pitch

        camera_1 = sensor.copy()
        camera_2 = sp_shift(sensor, disparity_px, order=1, mode='constant', cval=0.0)

        error = []
        for trial in range(num_repeats):
            noise = (
                    np.random.normal(size=camera_1.shape) * stdNoise * np.sqrt(abs(camera_1))
                )
            camera_1_noisy = camera_1 + noise
            noise = (
                    np.random.normal(size=camera_2.shape) * stdNoise * np.sqrt(abs(camera_2))
                )
            camera_2_noisy = camera_2 + noise
            noise_images = np.array([camera_1_noisy, camera_2_noisy]).flatten()
            diffs = preCal_images - noise_images[None, :]
            costs = np.sum(diffs**2, axis=1)

            best_idx = np.argmin(costs)
            Z_est = WORKING_RANGE[best_idx]
            error.append(np.abs(Z_est - depth))

            Z_est_list.append(Z_est)
            Z_true_list.append(depth)
        mean_error.append(np.mean(error))
    
    for error_rates in np.linspace(0.01, 0.10, 10):
        range_length = np.count_nonzero(np.array(mean_error) / np.array(WORKING_RANGE) < error_rates) * (WORKING_RANGE[1] - WORKING_RANGE[0])
        print(f"Effective range for error rate {error_rates}: {range_length:.2f} m")

    Z_est_list = np.array(Z_est_list)
    Z_true_list = np.array(Z_true_list)

    plotSingleResult(
        np.array(Z_est_list),
        np.array(Z_true_list),
        "./stereo.png",
        title="",
    )


def get_1d_psf(radius,
               length,
               pixel_pitch,
               nsigma=4,
               spike_thresh=0.5,
               oversampling=50):
    sigma = np.abs(float(radius))
    eff_pixel_pitch = pixel_pitch / oversampling  # smaller steps

    # 1) If σ is much smaller than a pixel, just return a delta at center
    if sigma < spike_thresh * pixel_pitch:
        psf = np.zeros(length, dtype=float)
        psf[length // 2] = 1.0
        return psf

    # 2) determine how many pixels cover ±nsigma·σ (in oversampled grid)
    half_support = int(np.ceil(nsigma * sigma / pixel_pitch))
    half_support = min(half_support, length // 2)
    hr_half_support = half_support * oversampling

    # 3) sample x at those integer-pixel offsets (oversampled)
    offsets_hr = np.arange(-hr_half_support, hr_half_support + 1)
    x_hr = offsets_hr * eff_pixel_pitch

    # 4) compute the Gaussian at high resolution
    local_hr = np.exp(-0.5 * (x_hr / sigma) ** 2)
    # Downsample by summing oversampling points for each base pixel
    local = np.add.reduceat(local_hr, np.arange(0, len(local_hr), oversampling))
    local /= local.sum()

    # 5) embed back into full-length array
    psf = np.zeros(length, dtype=float)
    start = (length // 2) - half_support
    psf[start : start + local.size] = local

    return psf


def get_sigma(optical_power, sensor_distance, depth):
    return 1 - sensor_distance * optical_power + sensor_distance / depth


def get_depth_focaltrack(params, Laplacian_I, I_rho):
    a = params["Sigma"] ** 2 * params["sensorDistance"] ** 2 * params["Delta_rho"]
    b = (params["Sigma"] ** 2 * params["sensorDistance"] * params["Delta_rho"]) * (params["sensorDistance"] * params["rho"] - 1)
    c = 1

    depth = (a * Laplacian_I) / (b * Laplacian_I + c * I_rho)

    return depth


def simulation_focaltrack_hack(params):
    rho = params["rho"]
    Sigma = params["Sigma"]
    Delta_rho = params["Delta_rho"]
    sensorDistance = params["sensorDistance"]
    photon_per_brightness_level = params["photon_per_brightness_level"]
    stdNoise = 1 / np.sqrt(photon_per_brightness_level)
    full_frame_size = 0.0351  # 35 mm
    sensor_dimension = 351
    pixel_pitch = full_frame_size / sensor_dimension
    sensor = np.zeros(sensor_dimension)
    sensor[sensor_dimension // 2] = 1  # Point source at the center
    num_repeats = 100
    
    list_Z = []
    list_Z_true = []

    if os.path.exists("./preCal_images_focal_track.npy"):
        preCal_images = np.load("./preCal_images_focal_track.npy")
    else:
        preCal_images = []

        for depth in WORKING_RANGE:
            print(f"Simulating for depth: {depth:.2f} m")
            sigma = get_sigma(rho, sensorDistance, depth)
            psf = get_1d_psf(sigma * Sigma, sensor_dimension, pixel_pitch)
            image = np.convolve(sensor, psf, mode='same')

            sigma_plus = get_sigma(rho + Delta_rho, sensorDistance, depth)
            psf_plus = get_1d_psf(sigma * Sigma, sensor_dimension, pixel_pitch)
            image_plus = np.convolve(sensor, psf, mode='same')

            sigma_minus = get_sigma(rho - Delta_rho, sensorDistance, depth)
            psf_minus = get_1d_psf(sigma * Sigma, sensor_dimension, pixel_pitch)
            image_minus = np.convolve(sensor, psf, mode='same')

            preCal_images.append(np.array([image, image_plus, image_minus]).flatten())

        preCal_images = np.array(preCal_images)
        np.save("./preCal_images_focal_track.npy", preCal_images)

    mean_error = []
    for depth in WORKING_RANGE:
        sigma = get_sigma(rho, sensorDistance, depth)
        psf = get_1d_psf(sigma * Sigma, sensor_dimension, pixel_pitch)
        image = np.convolve(sensor, psf, mode='same')

        sigma_plus = get_sigma(rho + Delta_rho, sensorDistance, depth)
        psf_plus = get_1d_psf(sigma_plus * Sigma, sensor_dimension, pixel_pitch)
        image_plus = np.convolve(sensor, psf_plus, mode='same')

        sigma_minus = get_sigma(rho - Delta_rho, sensorDistance, depth)
        psf_minus = get_1d_psf(sigma_minus * Sigma, sensor_dimension, pixel_pitch)
        image_minus = np.convolve(sensor, psf_minus, mode='same')

        # plt.figure(figsize=(10, 5))
        # ax = plt.subplot(1, 3, 1)
        # ax.plot(image, label='Image')
        # ax = plt.subplot(1, 3, 2)
        # ax.plot(image_plus, label='Image + Delta_rho')
        # ax = plt.subplot(1, 3, 3)
        # ax.plot(image_minus, label='Image - Delta_rho')
        # plt.show()

        print("Radius in pixels:", np.abs(sigma * Sigma / pixel_pitch))

        error = []
        for trial in range(num_repeats):
            noise_image = image + np.random.normal(size=image.shape) * stdNoise * np.sqrt(abs(image))
            noise_image_plus = image_plus + np.random.normal(size=image_plus.shape) * stdNoise * np.sqrt(abs(image_plus))
            noise_image_minus = image_minus + np.random.normal(size=image_minus.shape) * stdNoise * np.sqrt(abs(image_minus))

            noise_images = np.array([noise_image, noise_image_plus, noise_image_minus]).flatten()
            diffs = preCal_images - noise_images[None, :]
            costs = np.sum(diffs**2, axis=1)

            best_idx = np.argmin(costs)
            Z_est = WORKING_RANGE[best_idx]

            error.append(np.abs(Z_est - depth))

            list_Z.append(Z_est)
            list_Z_true.append(depth)
        mean_error.append(np.mean(error))

    for error_rates in np.linspace(0.01, 0.10, 10):
        range_length = np.count_nonzero(np.array(mean_error) / np.array(WORKING_RANGE) < error_rates) * (WORKING_RANGE[1] - WORKING_RANGE[0])
        print(f"Effective range for error rate {error_rates}: {range_length:.2f} m")

    list_Z = np.array(list_Z)
    list_Z_true = np.array(list_Z_true)
    plotSingleResult(
        list_Z,
        list_Z_true,
        "./focal_track.png",
        title="",
    )
        

def main():
    params = {
        "rho": 0.9851,
        "Sigma": 0.0175,
        "Delta_rho": 0.00001,
        "sensorDistance": 0.985,
        "photon_per_brightness_level": 10000,
        "kernelSize": 5,
    }

    # Check focal distance
    # focal_distance = 1 / (params["rho"] - 1 / params["sensorDistance"])
    # pdb.set_trace()

    # Test the getTransmission function
    # getTransmission = build_getTransmission()
    # angle = [0, 5, 10.0]
    # d = [10, 20, 30]
    # transmission = getTransmission(angle, d)
    # print(f"Transmission for angle {angle} and distance {d}: {transmission}")
    # simulation_mmdp_hack()
    simulation_stereo_hack(params)
    simulation_focaltrack_hack(params)

    return


if __name__ == "__main__":
    main()
