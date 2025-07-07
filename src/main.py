import numpy as np
import matplotlib.pyplot as plt
import pdb

PIXEL_PITCH = 5.86 * 1e-6
SENSOR_SIZE = 200
WORKING_RANGE = np.linspace(0.30, 2.70, 81)
LAPLACIAN_KERNEL = np.array([-1, 2, -1]) / (PIXEL_PITCH ** 2)
HEATMAP_RANGE = [
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
]

def get_sine_1d_texture(frequency, amplitude, phase, length, pixel_pitch=PIXEL_PITCH):
    x = np.linspace(-length // 2, length // 2, num=length) * pixel_pitch
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)


def get_1d_psf(radius, length, pixel_pitch=PIXEL_PITCH):
    x = np.linspace(-length // 2, length // 2, num=length) * pixel_pitch
    psf = np.exp(-0.5 * (x / radius) ** 2)
    return psf / np.sum(psf)


def get_sigma(optical_power, sensor_distance, depth):
    return 1 - sensor_distance * optical_power + sensor_distance / depth


def get_depth_focaltrack(params, Laplacian_I, I_rho):
    a = params["Sigma"] ** 2 * params["sensorDistance"] ** 2 * params["Delta_rho"]
    b = (params["Sigma"] ** 2 * params["sensorDistance"] * params["Delta_rho"]) * (params["sensorDistance"] * params["rho"] - 1)
    c = 1

    depth = (a * Laplacian_I) / (b * Laplacian_I + c * I_rho)

    return depth


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


def simulation_focaltrack(params, texture):
    list_Z = []
    list_Z_true = []
    list_Confidence = []

    for depth in WORKING_RANGE:
        sigma = get_sigma(params["rho"], params["sensorDistance"], depth)
        psf = get_1d_psf(sigma * params["Sigma"], SENSOR_SIZE, PIXEL_PITCH)
        image = np.convolve(texture, psf, mode='same')

        sigma_plus = get_sigma(params["rho"] + params["Delta_rho"], params["sensorDistance"], depth)
        psf_plus = get_1d_psf(sigma_plus * params["Sigma"], SENSOR_SIZE, PIXEL_PITCH)
        image_plus = np.convolve(texture, psf_plus, mode='same')

        sigma_minus = get_sigma(params["rho"] - params["Delta_rho"], params["sensorDistance"], depth)
        psf_minus = get_1d_psf(sigma_minus * params["Sigma"], SENSOR_SIZE, PIXEL_PITCH)
        image_minus = np.convolve(texture, psf_minus, mode='same')

        Laplacian_I = np.convolve(image, LAPLACIAN_KERNEL, mode='same')
        I_rho = (image_plus - image_minus) / 2
        depth_estimate = get_depth_focaltrack(params, Laplacian_I, I_rho)
        depth_true = np.full(SENSOR_SIZE, depth)

        print(f"Depth: {depth:.2f} m, Estimated Depth: {depth_estimate.mean():.2f} m")

        list_Z.append(depth_estimate)
        list_Z_true.append(depth_true)
        list_Confidence.append(np.ones(SENSOR_SIZE))

    list_Z = np.array(list_Z)
    list_Z_true = np.array(list_Z_true)
    list_Confidence = np.array(list_Confidence)

    plotSingleResult(
        list_Z, list_Z_true, "./focaltrack.png",
        title="Focal Track", normalisation=False
    )


def shift_texture_physically(texture, shift_px):
    """
    Shift the 1D texture physically using zero-padding.
    Positive shift => right shift; Negative shift => left shift.
    """
    N = len(texture)
    shifted = np.zeros_like(texture)

    if shift_px > 0:
        shifted[shift_px:] = texture[:N - shift_px]
    elif shift_px < 0:
        shifted[:shift_px] = texture[-shift_px:]
    else:
        shifted = texture.copy()

    return shifted

def simulation_stereo(params, texture):
    f = 1 / (params["rho"] - 1 / params["sensorDistance"])  # focal length in meters
    B = 2 * params["Sigma"]  # baseline in meters

    list_Z = []
    list_Z_true = []

    for depth in WORKING_RANGE:
        # Ground-truth disparity
        disparity_m = f * B / depth
        disparity_px = disparity_m / PIXEL_PITCH
        shift = int(np.round(disparity_px))

        # Simulate blur using PSFs (same for left and right for now)
        sigma = get_sigma(params["rho"], params["sensorDistance"], depth)
        radius = sigma * params["Sigma"]
        psf = get_1d_psf(radius, SENSOR_SIZE, PIXEL_PITCH)

        # Generate physically shifted version of texture
        shifted_texture = shift_texture_physically(texture, -shift)

        # Convolve both images with PSF
        blurred_left = np.convolve(texture, psf, mode='same')
        blurred_right = np.convolve(shifted_texture, psf, mode='same')

        # Estimate disparity via integer shift matching
        search_range = 10
        min_loss = float('inf')
        best_shift = 0
        for s in range(-search_range, search_range + 1):
            candidate = shift_texture_physically(blurred_right, s)
            loss = np.mean((blurred_left - candidate) ** 2)
            if loss < min_loss:
                min_loss = loss
                best_shift = s

        estimated_disparity_px = abs(shift + best_shift)
        if estimated_disparity_px == 0:
            Z_est = np.nan
        else:
            Z_est = (f * B) / (estimated_disparity_px * PIXEL_PITCH)

        depth_estimate = np.full(SENSOR_SIZE, Z_est)
        depth_true = np.full(SENSOR_SIZE, depth)

        print(f"[Stereo+PSF] True Z: {depth:.2f} m | Est Z: {Z_est:.2f} m | GT shift: {shift}px | Est shift: {best_shift}px")

        list_Z.append(depth_estimate)
        list_Z_true.append(depth_true)

    list_Z = np.array(list_Z)
    list_Z_true = np.array(list_Z_true)

    plotSingleResult(
        list_Z, list_Z_true, "./stereo.png",
        title="Stereo", normalisation=False
    )


def simulation_mmdp(params, texture):
    ray_offset = params.get("ray_offset", 1.0e-3)  # lenslet-to-sensor distance (meters)
    angles = np.linspace(-0.05, 0.05, params.get("N_angles", 5))  # angular sampling in radians

    def capture_measurement(texture, depth):
        combined = np.zeros_like(texture)
        for theta in angles:
            shift_px = int(np.round((ray_offset * np.tan(theta)) / (depth * PIXEL_PITCH)))
            shifted = shift_texture_physically(texture, shift_px)
            combined += shifted
        return combined / len(angles)

    list_Z = []
    list_Z_true = []

    for depth in WORKING_RANGE:
        observed = capture_measurement(texture, depth)

        # Search for best matching depth
        best_Z = None
        min_loss = float('inf')
        for d_test in WORKING_RANGE:
            pred = capture_measurement(texture, d_test)
            loss = np.mean((observed - pred) ** 2)
            if loss < min_loss:
                min_loss = loss
                best_Z = d_test

        depth_estimate = np.full(SENSOR_SIZE, best_Z)
        depth_true = np.full(SENSOR_SIZE, depth)

        print(f"[MMDP] True Z: {depth:.2f} m | Estimated Z: {best_Z:.2f} m")

        list_Z.append(depth_estimate)
        list_Z_true.append(depth_true)

    list_Z = np.array(list_Z)
    list_Z_true = np.array(list_Z_true)

    plotSingleResult(
        list_Z, list_Z_true, "./mmdp.png",
        title="MMDP", normalisation=False
    )


def main():
    params = {
        "rho": 10.1,
        "Sigma": 0.0025,
        "Delta_rho": 0.06,
        "Delta_Sigma": 0.0010,
        "sensorDistance": 0.1100,
        "photon_per_brightness_level": 240,
        "kernelSize": 5,
        "ray_offset": 1.0e-3,  # distance from microlens or aperture to sensor (in meters)
        "N_angles": 5,         # number of angular samples
    }
    texture = get_sine_1d_texture(1000, 255, 0, SENSOR_SIZE, PIXEL_PITCH)

    simulation_focaltrack(params, texture)
    simulation_stereo(params, texture)
    simulation_mmdp(params, texture)


    return

if __name__ == "__main__":
    main()