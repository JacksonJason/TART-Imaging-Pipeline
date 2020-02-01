import requests

def antenna_layout(loc):
    """
    Sends a request to the TART API in either New Zealand or South Africa
    Retreives the antenna layout.

    :param loc: The location of the telescope
    :type loc: str

    :returns: The antenna layout in JSON format.
    """
    if loc == "1":
        # SU
        r = requests.get("http://146.232.222.105/api/v1/imaging/antenna_positions")
    else:
        # NZ
        r = requests.get("https://tart.elec.ac.nz/signal/api/v1/imaging/antenna_positions")
    if not r.status_code == 200:
        return "Error retreiving antenna positions"
    return r.json()

def get_visibilities(loc):
    """
    Sends a request to the TART API in either New Zealand or South Africa
    Retreives the latest visibilities.

    :param loc: The location of the telescope
    :type loc: str

    :returns: The latest visibilities in JSON format.
    """
    if loc == "1":
        # SU
        r = requests.get("http://146.232.222.105/api/v1/imaging/vis")
    else:
        # NZ
        r = requests.get("https://tart.elec.ac.nz/signal/api/v1/imaging/vis")

    if not r.status_code == 200:
        return "Error retreiving visibilities"
    return r.json()

def get_latitude_and_frequency(loc):
    """
    Sends a request to the TART API in either New Zealand or South Africa
    Retreives the general information about the telescope.

    :param loc: The location of the telescope
    :type loc: str

    :returns: A tuple containing the latitude and frequency of the telescope.
    """
    if loc == "1":
        # SU
        r = requests.get("http://146.232.222.105/api/v1/info")
    else:
        # NZ
        r = requests.get("https://tart.elec.ac.nz/signal/api/v1/info")
    if not r.status_code == 200:
        return "Error retreiving latitude"
    info = r.json()["info"]
    return(info["location"]["lat"], info["operating_frequency"])
