import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.constants
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.pylab as pl
import Tigger
from matplotlib import rc
rc('text', usetex=True)

def find_closest_power_of_two(number):
    """
    Finds the closest power of two

    :param number: The number to find the power for
    :type number: int

    :returns: The next closest power of two for the number.
    """
    s = 2
    for i in range(0, 15):
        if number < s:
            return s
        else:
            s *= 2

def draw_matrix(matrix):
    """
    Draws the TART visibilities onto two image planes, one for real and one
    for imaginary. It saves this plot into a png file in the plots folder.

    :param matrix: The complex visibilities
    :type matrix: complex 2d numpy Array

    :returns: Nothing
    """
    plt.figure()
    plt.subplot(121)
    plt.set_cmap('viridis')
    plt.imshow(matrix.real)
    plt.title("Real: visibilities")

    plt.subplot(122)
    plt.imshow(matrix.imag)
    plt.title("Imag: visibilities")
    plt.savefig('Plots/Antenna_Visibilities.png', transparent=True)
    plt.close()

def get_B(b_ENU, L):
    """
    Converts the xyz form of the baseline into the coordinate system XYZ

    :param b_ENU: The baseline to convert
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :returns: The baseline in XYZ
    """
    D = math.sqrt(np.sum((b_ENU)**2))
    A = np.arctan2(b_ENU[0],b_ENU[1])
    E = np.arcsin(b_ENU[2]/D)
    B = np.array([D * (math.cos(L)*math.sin(E) - math.sin(L) * math.cos(E)*math.cos(A)),
                D * (math.cos(E)*math.sin(A)),
                D * (math.sin(L)*math.sin(E) + math.cos(L) * math.cos(E)*math.cos(A))])
    return B

def get_lambda(f):
    """
    Gets the wavelength for calculations

    :param f: The frequency of the interferometer
    :type f: float

    :returns: Lambda, wavelength
    """
    c = scipy.constants.c
    lam = c/f
    return lam

def plot_baseline(b_ENU, L, f, h0, h1, dec, name):
    """
    Finds the baseline in XYZ coordinates, then calculates the major axis, minor
    axis and center of the ellipse for the baseline plot, also works out the UV
    coordinates associated with the baseline.


    :param b_ENU: The baseline in xyz format
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param h0: The start hour angle
    :type h0: float

    :param h1: The end hour angle
    :type h1: float

    :param dec: The declination of the center
    :type dec: float

    :param name: The name for the image file
    :type name: str

    :returns: Nothing
    """
    B = get_B(b_ENU, L)
    lam = get_lambda(f)
    h = np.linspace(h0,h1,num=600)*np.pi/12

    X = B[0]
    Y = B[1]
    Z = B[2]

    u = lam**(-1)*(np.sin(h)*X+np.cos(h)*Y)
    v = lam**(-1)*(-np.sin(dec)*np.cos(h)*X+np.sin(dec)*np.sin(h)*Y+np.cos(dec)*Z)

    a = np.sqrt(X**2+Y**2)/lam
    b = a*np.sin(dec)
    v0 = (Z/lam)*np.cos(dec)

    UVellipse(u,v,a,b,v0, name)

def UVellipse(u,v,a,b,v0, name):
    """
    This code is taken from the fundamentals of interferometry notebook.
    It plots an ellipse for the baseline pair and saves it as a png file in
    the Plots folder.

    :param u: The u coordinates associated with the baseline
    :type u: numpy float array

    :param v: The v coordinates associated with the baseline
    :type v: numpy float array

    :param a: The major axis of the ellipse
    :type a: float

    :param b: The major axis of the ellipse
    :type b: float

    :param v0: The center of the ellipse
    :type v0: float

    :param name: The name to apply to the image file that is produced
    :type name: str

    :returns: Nothing
    """
    fig=plt.figure(0, figsize=(8,8))

    e1=Ellipse(xy=np.array([0,v0]),width=2*a,height=2*b,angle=0)
    e2=Ellipse(xy=np.array([0,-v0]),width=2*a,height=2*b,angle=0)

    ax=fig.add_subplot(111,aspect="equal")

    ax.plot([0],[v0],"go")
    ax.plot([0],[-v0],"go")
    ax.plot(u[0],v[0],"bo")
    ax.plot(u[-1],v[-1],"bo")

    ax.plot(-u[0],-v[0],"ro")
    ax.plot(-u[-1],-v[-1],"ro")

    ax.add_artist(e1)
    e1.set_lw(1)
    e1.set_ls("--")
    e1.set_facecolor("w")
    e1.set_edgecolor("b")
    e1.set_alpha(0.5)
    ax.add_artist(e2)

    e2.set_lw(1)
    e2.set_ls("--")
    e2.set_facecolor("w")
    e2.set_edgecolor("r")
    e2.set_alpha(0.5)
    ax.plot(u,v,"b")
    ax.plot(-u,-v,"r")
    ax.grid(True)
    plt.title("UV Coverage", size=26)
    plt.xlabel("u", size=22)
    plt.ylabel("v", size=22)
    ax.tick_params(labelsize=20)
    plt.savefig('Plots/' + name + 'UVCoverage.png', transparent=True)
    plt.close()

def plot_array(antennas, name):
    """
    Plots the array layout for the interferometer.

    :param antennas: The antenna layout array.
    :type antennas: 2d float array

    :param name: The name for the file to be saved in
    :type name: str

    :returns: Nothing
    """
    plt.figure(figsize=(10,10))
    plt.scatter(antennas[:,0], antennas[:,1])
    plt.grid(True)
    plt.xlabel('E-W [m]', size=24)
    plt.ylabel('N-S [m]', size=24)
    plt.title(name + ' Array Layout', size=26)
    ax = plt.gca()
    ax.tick_params(labelsize=22)
    plt.savefig('Plots/' + name + 'AntennaLayout.png', transparent=True)
    plt.close()

def get_visibilities(b_ENU, L, f, h0, h1, model_name, layout):
    """
    Gets the visibilities from the sky model, also calculates the uv_tracks

    :param b_ENU: The baseline in xyz format.
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param h0: The start hour angle
    :type h0: float

    :param h1: The end hour angle
    :type h1: float

    :param model_name: The name of the sky model
    :type model_name: str

    :param layout: The antenna layout
    :type layout: numpy float array

    :returns: The visibilities, the UV tracks, and the center declination as well
              as the center right ascention.
    """
    h = np.linspace(h0,h1,num=600)*np.pi/12
    point_sources, l, m, dec, flux_sources, ra_0 = load_sky_model(model_name)

    plot_sky_model(l*(180/np.pi), m*(180/np.pi), flux_sources, "l [degrees]", "m [degrees]")

    uv, u_d, v_d, uu, vv, uv_tracks = get_uv_and_uv_tracks(b_ENU, L, f, h, dec, point_sources)
    plot_uv_tracks(uv_tracks)

    plot_visibilities(u_d, v_d, uu, vv, point_sources)
    all_uv_tracks, all_uv = get_all_uv_and_uv_tracks(L, f, h, dec, point_sources, layout)

    return all_uv, all_uv_tracks, dec, ra_0

def load_sky_model(model_name):
    """
    Loads the sky model and extracts all the necessary information.

    :param model_name: The name of the sky model to be loaded
    :type model_name: str

    :returns: The point sources from the loaded sky model, the l coordinates of
              the points, the m coordinates of the points, the center declination,
              the flux sources, and the center right ascention
    """
    model = Tigger.load(model_name)
    RA_sources = []
    DEC_sources = []
    Flux_sources = []

    for val in model.sources:
        RA_sources.append(val.pos.ra)
        DEC_sources.append(val.pos.dec)
        Flux_sources.append(val.flux.I)

    RA_sources = np.array(RA_sources)
    DEC_sources = np.array(DEC_sources)
    Flux_sources = np.array(Flux_sources)

    ra_0 = model.ra0
    dec_0 = model.dec0
    ra_0_rad = ra_0 * (np.pi/12)
    dec_0_rad = dec_0 * (np.pi/180)

    RA_rad = RA_sources*(np.pi/12)
    DEC_rad = DEC_sources*(np.pi/180)
    RA_delta_rad = RA_rad-ra_0_rad

    l = np.cos(DEC_rad)*np.sin(RA_delta_rad)
    m = (np.sin(DEC_rad)*np.cos(dec_0_rad)-np.cos(DEC_rad)*np.sin(dec_0_rad)*np.cos(RA_delta_rad))

    point_sources = np.zeros((len(RA_sources),3))
    point_sources[:,0] = Flux_sources
    point_sources[:,1] = l[0:]
    point_sources[:,2] = m[0:]
    dec = dec_0

    return point_sources, l, m, dec_0, Flux_sources, ra_0_rad

def plot_sky_model(l, m, Flux_sources, x, y):
    """
    Plots the sky model from the LSM file and saves it into the plots folder.

    :param l: The l coordinates of the sky model
    :type l: numpy array

    :param m: The m coordinates of the sky model
    :type m: numpy array

    :param Flux_sources: The brightness of the sources
    :type Flux_sources: numpy array

    :param x: The name for the x axis
    :type x: str

    :param y: The name for the y axis
    :type y: str
    """
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    plt.xlabel(x, size=24)
    plt.ylabel(y, size=24)
    ax.tick_params(labelsize=22)
    max_flux = max(Flux_sources)

    if max_flux > 1:
        col = (Flux_sources/max_flux)
    else:
        col = Flux_sources

    colour = []
    for i in col:
        colour.append((i,i,i))

    plt.scatter(l,m,c=colour,s=8)
    ax.set_facecolor('xkcd:black')
    fig.patch.set_alpha(0)
    plt.title("Sky Model", size=26)
    plt.savefig("Plots/SkyModel.png", transparent=False)
    plt.close()

def get_uv_and_uv_tracks(b_ENU, L, f, h, dec, point_sources):
    """
    Gets the UV tracks for the baseline as well as the UV coordinates
    for the visibilities from the sky model file

    :param b_ENU: The baseline in xyz format.
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param h: The hour angle array
    :type h: float array

    :param dec: The declination at the center
    :type dec: float

    :param dec: The point sources for the sky model
    :type dec: numpy array

    :returns: The UV tracks for the baseline and the UV coordinates for the
              sky model
    """
    u_d, v_d = get_uv_tracks(b_ENU, L, f, h, dec)

    step_size = 200
    u = np.linspace(-1*(np.amax(np.abs(u_d)))-10, np.amax(np.abs(u_d))+10, num=step_size, endpoint=True)
    v = np.linspace(-1*(np.amax(abs(v_d)))-10, np.amax(abs(v_d))+10, num=step_size, endpoint=True)

    uu, vv = np.meshgrid(u, v)
    uv_tracks = calculate_uv_tracks(point_sources, u_d, v_d)

    uv = []
    for i in range(len(u_d)):
        uv.append([u_d[i], v_d[i]])

    uv = np.array(uv)

    return uv, u_d, v_d, uu, vv, uv_tracks

def get_uv_tracks(b_ENU, L, f, h, dec):
    """
    Gets the UV tracks for the baseline

    :param b_ENU: The baseline in xyz format.
    :type b_ENU: float array

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param h: The hour angle array
    :type h: float array

    :param dec: The declination at the center
    :type dec: float

    :returns: The UV tracks for the baseline
    """
    B = get_B(b_ENU, L)
    lam = get_lambda(f)

    X = B[0]
    Y = B[1]
    Z = B[2]

    u = lam**(-1)*(np.sin(h)*X+np.cos(h)*Y)
    v = lam**(-1)*(-np.sin(dec)*np.cos(h)*X+np.sin(dec)*np.sin(h)*Y+np.cos(dec)*Z)

    return u, v

def calculate_uv_tracks(point_sources, u, v):
    """
    Calculates the UV tracks using a fourier transform.

    :param point_sources: The point source to plot
    :type point_sources: numpy float array

    :param u: The u coordinates of the uv tracks
    :type u: numpy float array

    :param v: The v coordinates of the uv tracks
    :type v: numpy float array

    :returns: The UV tracks/coordinates for the baseline and point source
    """
    z = np.zeros(u.shape).astype(complex)
    s = point_sources.shape
    for counter in range(0, s[0]):
        A_i = point_sources[counter,0]
        l_i = point_sources[counter,1]
        m_i = point_sources[counter,2]
        z += A_i*np.exp(-1*2*np.pi*1j*((u*l_i)+(v*m_i)))

    return z

def plot_uv_tracks(uv_tracks):
    """
    Plots the UV tracks

    :param uv_tracks: The UV tracks of the baseline
    :type uv_tracks: numpy float array

    :returns: Nothing
    """
    plt.subplot(121)
    plt.plot(uv_tracks.real)
    plt.xlabel("Timeslots", size=18)
    plt.ylabel("Jy", size=18)
    plt.title("Real: sampled visibilities", size=18)

    plt.subplot(122)
    plt.plot(uv_tracks.imag)
    plt.xlabel("Timeslots", size=18)
    plt.title("Imag: sampled visibilities", size=18)
    plt.savefig('Plots/SampledVisibilities.png', transparent=True)

    axc = plt.gca()
    axc.tick_params(labelsize=16)

    plt.close()

def plot_visibilities(u, v, uu, vv, point_sources):
    """
    Plots the visibilities for the baseline chosen.

    :param u: The U coordinates of the UV tracks
    :type u: numpy float array

    :param v: the V coordinates of the UV tracks
    :type v: numpy float array

    :param uu: The U coordinates for the visibilities
    :type uu: numpy float array

    :param vv: The V coordinates for the visibilities
    :type vv: numpy float array

    :param point_sources: The point sources to plot the visibilities for
    :type point_sources: numpy float array
    """
    zz = np.zeros(uu.shape).astype(complex)
    s = point_sources.shape

    for counter in range(0, s[0]):
        A_i = point_sources[counter,0]
        l_i = point_sources[counter,1]
        m_i = point_sources[counter,2]
        zz += A_i*np.exp(-2*np.pi*1j*(uu*l_i+vv*m_i))
    zz = zz[:,::-1]

    plt.figure()
    plt.set_cmap('viridis')
    plt.subplot(121)
    plt.imshow(zz.real,extent=[-1*(np.amax(np.abs(u)))-10, np.amax(np.abs(u))+10,-1*(np.amax(abs(v)))-10, \
                               np.amax(abs(v))+10])

    plt.plot(u,v,"k")
    plt.xlim([-1*(np.amax(np.abs(u)))-10, np.amax(np.abs(u))+10])
    plt.ylim(-1*(np.amax(abs(v)))-10, np.amax(abs(v))+10)
    plt.xlabel("u", size=18)
    plt.ylabel("v", size=18)
    plt.title("Real part of visibilities", size=18)

    plt.subplot(122)
    plt.imshow(zz.imag,extent=[-1*(np.amax(np.abs(u)))-10, np.amax(np.abs(u))+10,-1*(np.amax(abs(v)))-10, \
                               np.amax(abs(v))+10])
    plt.plot(u,v,"k")
    plt.xlim([-1*(np.amax(np.abs(u)))-10, np.amax(np.abs(u))+10])
    plt.ylim(-1*(np.amax(abs(v)))-10, np.amax(abs(v))+10)
    plt.xlabel("u", size=18)
    plt.title("Imaginary part of visibilities", size=18)
    plt.savefig('Plots/Visibilities.png', transparent=True)
    ax = plt.gca()
    ax.tick_params(labelsize=16)
    plt.close()

def get_all_uv_and_uv_tracks(L, f, h, dec, point_sources, layout):
    """
    Gets all of the UV tracks and UV coordinates for all of the baselines

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param dec: The center declination of the view
    :type dec: float

    :param point_sources: All the point sources from the sky model
    :type point_sources: numpy float array

    :param layout: The antenna layout
    :type layout: numpy float array

    :returns: All of the UV tracks and coordinates across all the baselines
    """
    all_uv_tracks = []
    all_uv = []

    for i in range(len(layout)):
        for j in range(i+1, len(layout)):
            b = layout[j] - layout[i]
            uv, u_d, v_d, uu, vv, uv_tracks = get_uv_and_uv_tracks(b, L, f, h, dec, point_sources)
            all_uv.append(uv)
            all_uv_tracks.append(uv_tracks)

            b = layout[i] - layout[j]
            uv, u_d, v_d, uu, vv, uv_tracks = get_uv_and_uv_tracks(b, L, f, h, dec, point_sources)
            all_uv.append(uv)
            all_uv_tracks.append(uv_tracks)

    return all_uv_tracks, all_uv

def get_TART_uv_and_tracks(layout, L, f, visibilities):
    """
    Gets all of the UV tracks and UV coordinates for all of the baselines

    :param layout: The layout of the interferometer
    :type layout: float array

    :param L: The latitude of the interferometer
    :type L: float

    :param f: the frequency of the interferometer
    :type f: float

    :param visibilities: The visibilities from TART
    :type visibilities: complex float array

    :returns: All of the UV tracks and coordinates across all the baselines for TART
    """
    all_uv = []
    all_uv_tracks = []
    for i in range(len(layout)):
        for j in range(i+1, len(layout)):
            b = layout[j] - layout[i]
            u_d, v_d =  get_uv_tracks(b, L, f, 0, L)
            uv = []
            uv.append([u_d, v_d])
            uv = np.array(uv)
            all_uv.append(uv)
            uv_tracks = [visibilities[i][j]]
            all_uv_tracks.append(uv_tracks)

            b = layout[i] - layout[j]
            u_d, v_d =  get_uv_tracks(b, L, f, 0, L)
            uv = []
            uv.append([u_d, v_d])
            uv = np.array(uv)
            all_uv.append(uv)
            uv_tracks = [visibilities[j][i]]
            all_uv_tracks.append(uv_tracks)
    return all_uv, all_uv_tracks


def image(uv, uv_tracks, cell_size, dec_0, res, name, showGrid):
    """
    Calculates the Resolusion from the cell size and the provided resolution,
    it then plots the baseline grid and applies the visibilities to a grid.
    After which it sends the grid off to be transformed and saved to a file.
    It also plots the PSF of the interferometer and scales the image correctly.

    :param uv: The visibilities of the interferometer
    :type uv: numpy complex float array

    :param uv_tracks: The UV coordinates of the baseline
    :type uv_tracks: numpy float array

    :param cell_size: The user supplied cell size for the image
    :type cell_size: float

    :param dec_0: The declination at the center of the view
    :type dec_0: float

    :param res: The user supplied resolution
    :type res: float

    :param name: The name for the output image
    :type name: str

    :param showGrid: Whether or not to show the grid on the image
    :type showGrid: boolean

    :returns: Nothing
    """
    c_s = float(cell_size)
    cell_size_l = c_s
    cell_size_m = c_s

    degrees_l = float(res)
    degrees_m = float(res)

    Nl = int(np.round(degrees_l / cell_size_l))
    Nm = int(np.round(degrees_m / cell_size_m))
    # Nl = find_closest_power_of_two(Nl)
    # Nm = find_closest_power_of_two(Nm)

    rad_d_l = cell_size_l * (np.pi/180)
    rad_d_m = cell_size_m * (np.pi/180)

    gridded, cell_size_error = grid(Nl, Nm, uv_tracks, uv, cell_size_l, cell_size_m)
    img = plt.figure(figsize=(8,8))

    plt.title("Baseline Grid", size=26)
    plt.set_cmap('nipy_spectral')
    im = plt.imshow(np.real(np.abs(gridded)), origin='lower')
    plt.ylabel("v [rad$^{-1}$]",size=24)
    plt.xlabel("u [rad$^{-1}$]", size=24)
    ax = plt.gca()
    ax.tick_params(labelsize=22)
    plt.savefig('Plots/' + name + 'grid.png', transparent=True)
    plt.close()

    # Find the center of the view.
    L = np.cos(dec_0) * np.sin(0)
    M = np.sin(dec_0) * np.cos(dec_0) - np.cos(dec_0) * np.sin(dec_0) * np.cos(0)

    image = fourier_transform_grid(gridded)
    psf = np.ones ((np.array(uv_tracks).shape), dtype=complex)
    psf_grid, cell_size_error = grid(Nl, Nm, psf, uv, cell_size_l, cell_size_m)
    psf_image = fourier_transform_grid(psf_grid)

    scale_factor = psf_image[int(psf_image.shape[0]/2)][int(psf_image.shape[1]/2)]
    image /= scale_factor
    psf_image /= scale_factor

    draw_image(image, Nl, Nm, cell_size_l, cell_size_m, L, M, name + " SkyModel", "l [degrees]", "m [degrees]", cell_size_error, showGrid)
    draw_image(psf_image, Nl, Nm, cell_size_l, cell_size_m, L, M, name + " PSF", "l [degrees]", "m [degrees]", cell_size_error, False)

def grid(Nl, Nm, uv_tracks, uv, cell_size_l, cell_size_m):
    """
    Creates and places the visibilities onto a 2d grid. If more than one visibility
    is at the same position it is divided by the number of visibilities at that position.

    :param Nl: The image size in the l dimension
    :type Nl: int

    :param Nm: The image size in the m dimension
    :type Nm: int

    :param uv_tracks: The uv tracks for the baseline
    :type uv_tracks: numpy float array

    :param uv: The UV coordinates to apply to the grid
    :type uv: numpy float array.

    :param cell_size_l: The cell size in the l direction
    :type cell_size_l: float

    :param cell_size_m: The cell size in the l direction
    :type cell_size_m: float

    :returns: The visibility grid and a boolean containing whether or not the cell
              size prduced an error.
    """
    vis = np.zeros((Nl, Nm), dtype=complex)
    counter = np.zeros((Nl, Nm))
    half_l = int(Nl / 2)
    half_m = int(Nm / 2)
    cell_size_error = False

    for i in range(len(uv)):
        scaled_uv = np.copy(uv[i])
        scaled_uv[:,0] *= np.deg2rad(cell_size_l * Nl)
        scaled_uv[:,1] *= np.deg2rad(cell_size_m * Nm)

        for j in range(len(scaled_uv)):
            y,x = int(np.round(scaled_uv[j][0])), int(np.round(scaled_uv[j][1]))
            x += half_l
            y += half_m
            if not x >= vis.shape[0] and not y >= vis.shape[1] and not x < -vis.shape[0] and not y < -vis.shape[1]:
                vis[x][y] += uv_tracks[i][j]
                counter[x][y] += 1
            else:
                cell_size_error = True

    for i in range(len(vis)):
        for j in range(len(vis[i])):
            if not counter[i][j] == 0:
                vis[i][j] = vis[i][j] / counter[i][j]

    return vis, cell_size_error

def fourier_transform_grid(grid):
    """
    Using the inverse fourier transform, converts the grid into an image

    :param grid: The grid to transform
    :type grid: 2d numpy array

    :returns: An image in matrix form
    """
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid)))
    image = np.abs(image)
    return np.real(image)

def draw_image(image, Nl, Nm, cell_size_l, cell_size_m, L, M, name,
                x_title, y_title, cell_size_error, showGrid):
    """
    Draws the image onto a figure, adds a color map to the side and draws circles
    for the declination.

    :param image: The image matrix to draw
    :type image: 2d Numpy array

    :param Nl: Image size in l direction
    :type Nl: int

    :param Nm: Image size in m direction
    :type Nm: int

    :param cell_size_l: cell size in l direction
    :type cell_size_l: float

    :param cell_size_m: cell size in m direction
    :type cell_size_m: float

    :param L: The L coordinate of the center of the view, used to center the image correctly
    :type L: float

    :param M: The m coordinate of the center of the view, used to center the image correctly
    :type M: float

    :param name: The name of the image to save with
    :type name: str

    :param x_title: The x axis name
    :type x_title: str

    :param y_title: The y axis name
    :type y_title: str

    :param cell_size_error: whether or not the cell size produced an error boolean
    :type cell_size_error: boolean


    :param showGrid: Whether or not to show the grid on the image
    :type showGrid: boolean

    :returns: Nothing
    """
    img = plt.figure(figsize=(9,8))
    plt.title("Reconstructed " + name,size=26)
    plt.set_cmap('nipy_spectral')

    # im_vis = plt.imshow(image, origin='lower', extent=[L - Nl / 2 * cell_size_l, L + Nl / 2 * cell_size_l,
    #                                                     M - Nm / 2 * cell_size_m, M + Nm / 2 * cell_size_m])
    axc = plt.gca()
    im_vis = axc.imshow(image, origin='lower', extent=[L - Nl / 2 * cell_size_l, L + Nl / 2 * cell_size_l,
                                                        M - Nm / 2 * cell_size_m, M + Nm / 2 * cell_size_m])


    if showGrid:
        plt.axvline(x=0,color='k')
        plt.axhline(y=0,color='k')
        for i in range(10,91,10):
            dec = i * np.pi/180
            d_ra = 0
            dec_0 = 0

            l = math.cos(dec) * math.sin(d_ra) * 180/np.pi
            m = (math.sin(dec) * math.cos(dec_0) - math.cos(dec) * math.sin(dec_0) * math.cos(d_ra)) * 180/np.pi
            radius = np.sqrt(l**2 + m**2)
            circ = Circle((0, 0), radius, fill=False, alpha=1,color='k', lw=2)
            axc.add_patch(circ)

    cbr = img.colorbar(im_vis)
    cbr.set_label('Jy per Beam',size=24)
    cbr.ax.tick_params(labelsize=22)
    plt.xlabel(x_title,size=24)
    plt.ylabel(y_title,size=24)
    axc.tick_params(labelsize=22)


    if cell_size_error:
        txt = "INVALID CELL SIZE"
        plt.figtext(0.5, 0.1, txt, wrap=True, horizontalalignment='center', fontsize=25, color='red')

    plt.savefig('Plots/Reconstructed' + name + '.png', transparent=True)
    plt.close()
