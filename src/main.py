import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

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

def transmissionFigure():
    theta = np.linspace(0, 80, 501)
    theta = np.deg2rad(theta)
    metasurface_distance = np.linspace(0, 5, 501)
    frequency = np.pi + np.abs(np.sin(3 * np.pi * metasurface_distance) * np.pi * metasurface_distance)
    transmissions = []
    for f in  frequency:
        transmissions.append(np.abs(np.cos(f * (theta - np.pi/6))))
    transmissions = np.array(transmissions).T

    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
        transmissions, aspect='auto', extent=[0, 5, 0, 80],
        origin='upper', cmap='viridis'
    )
    ax.set_xlabel("Distance (m)")

    plt.show()

def augular_transmission(theta, distance):
    """
    T(theta, d(t))
    Response transmission of the sensor at angle theta and distance d(t).
    For now it is just a raw estimation based on a sinusoidal function.
    """
    frequency = np.pi + np.abs(np.sin(3 * np.pi * distance) * np.pi * distance)
    transmission = np.abs(np.cos(frequency * (theta - np.pi/6)))

    return transmission

def simulation_mmdp(params, texture):
    metasurface_distances = np.linspace(0, 5, 30)
    # 1 mm for each pixel of the texture
    texture_pixiel_size = 1e-3

    texture_coordinates = np.zeros((len(texture), 2))
    texture_coordinates[:, 0] = (np.arange(len(texture)) - len(texture) // 2) * texture_pixiel_size

    sensor_coordinates = np.zeros((SENSOR_SIZE, 2))
    sensor_coordinates[:, 0] = (np.arange(SENSOR_SIZE) - SENSOR_SIZE // 2) * PIXEL_PITCH

    for depth in WORKING_RANGE:
        # Simulate light field at different depths
        # Light field (y, depth, theta)
        light_field = []
        images = np.zeros((SENSOR_SIZE, len(metasurface_distances)))

        for i in range(SENSOR_SIZE):
            y_diff = texture_coordinates[:, 0] - sensor_coordinates[i, 0]
            theta = np.arctan2(y_diff, depth)
            distance_squared = texture_coordinates[:, 0] ** 2 + depth ** 2
            light_field.append(np.array([texture_coordinates[:, 0], np.full(len(texture_coordinates[:, 0]), depth), theta]).T)
            for j, d in enumerate(metasurface_distances):
                transmission = augular_transmission(theta, d)
                images[i, j] += np.sum(texture / distance_squared * transmission)
        light_field = np.array(light_field)
        


def simulation_mmdp(params, texture):
    metasurface_distances = np.linspace(0, 5, 30)

    # Precompute coordinates
    N = texture.size
    tex_coords = (np.arange(N) - N//2) * 1e-3        # 1 mm per texture pixel
    sensor_coords = (np.arange(SENSOR_SIZE) - SENSOR_SIZE//2) * PIXEL_PITCH
    y_diff = tex_coords[None, :] - sensor_coords[:, None]  # shape: (SENSOR_SIZE, N)

    # Loop over true depths
    Z_est_list = []
    Z_true_list = []

    for depth_true in WORKING_RANGE:
        # Simulate measurement stack at this true depth
        images = np.zeros((SENSOR_SIZE, metasurface_distances.size))
        dist2_true = y_diff**2 + depth_true**2
        for j, d in enumerate(metasurface_distances):
            theta_true = np.arctan2(y_diff, depth_true)
            T = augular_transmission(theta_true, d)
            images[:, j] = np.sum(texture[None, :] / dist2_true * T, axis=1)

        # Depth estimation per pixel via Eq. (4) MLE (no regularization)
        def predict_pixel(i, Z_trial):
            dist2 = y_diff[i]**2 + Z_trial**2
            theta = np.arctan2(y_diff[i], Z_trial)
            stack = np.array([
                np.sum(texture / dist2 * augular_transmission(theta, d))
                for d in metasurface_distances
            ])
            return stack

        Z_est = np.zeros(SENSOR_SIZE)
        for i in range(SENSOR_SIZE):
            I_obs = images[i]

            def cost(Z_trial):
                I_pred = predict_pixel(i, Z_trial)
                return np.sum((I_obs - I_pred)**2)

            res = minimize_scalar(cost, bounds=(WORKING_RANGE.min(), WORKING_RANGE.max()), method='bounded')
            Z_est[i] = res.x

        Z_est_list.append(Z_est)
        Z_true_list.append(np.full(SENSOR_SIZE, depth_true))
        print(f"[MMDP] True Z={depth_true:.2f} m, Mean Z_est={Z_est.mean():.2f} m")

    Z_est_arr = np.array(Z_est_list)
    Z_true_arr = np.array(Z_true_list)
    plotSingleResult(Z_est_arr, Z_true_arr, "./mmdp.png", title="MMDP Optimization")


def forward_stack(P, z, d_list):
    """
    Forward model: predict measurement stack I_pred[k,j] for each sensor pixel k and metasurface distance j.
    """
    K = len(z)
    M = len(d_list)
    I_pred = np.zeros((K, M))
    # Precompute coordinates
    tex_coords = (np.arange(K) - K//2) * 1e-3           # 1 mm per texture pixel
    sensor_coords = (np.arange(K) - K//2) * PIXEL_PITCH
    for j, d in enumerate(d_list):
        for k in range(K):
            dx = tex_coords - sensor_coords[k]          # vector over scene points m
            theta = np.arctan2(dx, z)
            falloff = P / (dx**2 + z**2)
            T = augular_transmission(theta, d)
            I_pred[k, j] = np.sum(falloff * T)
    return I_pred

def reconstruct_MMDP(I_obs, d_list, z0, P0, lambda_z=1e-2, lambda_P=1e-2, max_iters=5):
    """
    Alternating minimization to recover depth z and appearance P from observations I_obs.
    """
    K, M = I_obs.shape
    z = z0.copy()
    P = P0.copy()

    for it in range(max_iters):
        # 1) Solve for P (linear least squares with ridge)
        rows = K*M; cols = K
        A = np.zeros((rows, cols))
        b = I_obs.flatten()
        for j, d in enumerate(d_list):
            I_row = j*K
            for k in range(K):
                dx = (np.arange(K)-k) * PIXEL_PITCH
                theta = np.arctan2(dx, z)
                G = 1.0/(dx**2 + z**2) * augular_transmission(theta, d)
                A[I_row + k, :] = G
        P = lsqr(A, b, damp=lambda_P)[0]

        # 2) Solve for z (nonlinear)
        def cost_z(z_vec):
            I_pred = forward_stack(P, z_vec, d_list)
            diff = I_pred - I_obs
            smooth = lambda_z * np.sum((z_vec[1:] - z_vec[:-1])**2)
            return np.sum(diff**2) + smooth

        res = minimize(
            cost_z, z, method='L-BFGS-B',
            bounds=[(WORKING_RANGE.min(), WORKING_RANGE.max())]*K
        )
        z = res.x

        print(f"Iter {it+1}/{max_iters}: cost={res.fun:.2e}, mean(z)={z.mean():.3f} m")

    return z, P

def simulation_mmdp(params, texture):
    """
    1) Simulate MMDP image stack across WORKING_RANGE of true depths
    2) Reconstruct z,P via alternating minimization (Eq. 4 model)
    """
    d_list = np.linspace(0, 5, 30)
    K = SENSOR_SIZE

    Z_est_list = []
    Z_true_list = []

    for depth_true in WORKING_RANGE:
        # simulate measurements I_obs[k,j]
        I_obs = np.zeros((K, len(d_list)))
        tex_coords = (np.arange(K) - K//2) * 1e-3
        sensor_coords = (np.arange(K) - K//2) * PIXEL_PITCH
        for j, d in enumerate(d_list):
            for k in range(K):
                dx = tex_coords - sensor_coords[k]
                theta = np.arctan2(dx, depth_true)
                falloff = texture / (dx**2 + depth_true**2)
                T = augular_transmission(theta, d)
                I_obs[k, j] = np.sum(falloff * T)

        # initialize and reconstruct
        z0 = np.full(K, depth_true)
        P0 = np.ones(K)
        z_est, P_est = reconstruct_MMDP(I_obs, d_list, z0, P0)

        print(f"[MMDP] True Z={depth_true:.2f} m → Mean Z_est={z_est.mean():.2f} m")

        Z_est_list.append(z_est)
        Z_true_list.append(np.full(K, depth_true))

    Z_est_arr = np.array(Z_est_list)
    Z_true_arr = np.array(Z_true_list)
    plotSingleResult(Z_est_arr, Z_true_arr, "./mmdp_new.png", title="MMDP Reconstruction")



def main():
    params = {
        "rho": 10.1,
        "Sigma": 0.0025,
        "Delta_rho": 0.06,
        "Delta_Sigma": 0.0010,
        "sensorDistance": 0.1100,
        "photon_per_brightness_level": 240,
        "kernelSize": 5,
    }
    texture = get_sine_1d_texture(1000, 255, 0, SENSOR_SIZE, PIXEL_PITCH)

    # simulation_focaltrack(params, texture)
    # simulation_stereo(params, texture)
    simulation_mmdp(params, texture)
    # transmissionFigure()


    return

if __name__ == "__main__":
    main()