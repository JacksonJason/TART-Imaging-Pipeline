import TARTRequests as TR
import numpy as np
import Utilities as ut
import cherrypy
import json
from jinja2 import Environment, FileSystemLoader
import time
from PIL import Image, ImageDraw
import glob
import shutil
import os
import re
import datetime

env = Environment(loader=FileSystemLoader('html'))
cherrypy.config.update({'server.socket_port': 8099})


class pipeline(object):

    def atoi(self, text):
        """
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        """
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        """
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def make_vis_matrix(self, loc, calibrate):
        """
        Requests the visibilities from the API, then converts that into a 2d
        array of visibilities.

        :param loc: The location of the interferometer, NZ or SA
        :type loc: str

        :returns: A matrix of visibilities
        """
        vis = TR.get_visibilities(loc)
        vis = np.array(vis["data"])

        i, j = self.parse_vis(vis)
        vis_matrix = np.zeros((i + 1, j + 1)).astype(complex)

        gains, phase_offset = TR.get_gains_and_phases()
        if calibrate:
            cherrypy.log("Calibrating")
            for v in vis:
                vis_matrix[v["i"]][v["j"]] = (v["re"] + 1j * v["im"]) * gains[v["i"]] * gains[v["j"]] * np.exp(
                    -1j * (phase_offset[v["i"]] - phase_offset[v["j"]]))
        else:
            for v in vis:
                vis_matrix[v["i"]][v["j"]] = v["re"] + 1j * v["im"]

        cc = vis_matrix.T
        cc = np.conj(cc)

        for i in range(cc.shape[0]):
            cc[i][i] = 0 + 0j

        vis_matrix = vis_matrix + cc

        ut.draw_matrix(vis_matrix)

        return vis_matrix

    def parse_vis(self, vis):
        """
        Finds the maximum i and j postions in the vis array

        :param vis: The visibility to convert
        :type vis: numpy array

        :returns: the x and y max coordinates
        """
        i = 0
        j = 0

        for v in vis:
            if v['i'] >= i:
                i = v['i']

            if v['j'] >= j:
                j = v['j']

        return i + 1, j

    def get_antenna_layout(self, loc):
        """
        Gets the antenna layout and returns it in JSON format.

        :returns: The antenna layout in JSON format.
        """
        layoutJSON = TR.antenna_layout(loc)
        return np.array(layoutJSON)

    def generate(self, cell_size, layout, L, f, visibilities, showGrid):
        """
        Generates the image files, all of the hard processing is done here

        :param cell_size: The cell size for the imaging process
        :type cell_size: str

        :param layout: The antenna layout
        :type layout: array

        :param L: The Latitude of the antenna
        :type L: int

        :param f: The frequency of the antenna
        :type f: int

        :param visibilities: The visibilities from the antenna
        :type visibilities: array

        :param showGrid: The boolean telling us whether to show the grid or
                        not on the images
        :type showGrid: boolean

        :returns: nothing
        """
        ut.plot_array(layout, "TART")

        all_uv, all_uv_tracks = ut.get_TART_uv_and_tracks(layout, L, f, visibilities)

        res = 2 * 180 / np.pi

        ut.image(all_uv, all_uv_tracks, cell_size, 0, res, "TART", showGrid, gif=True)

    @cherrypy.expose
    def generate_custom_graphs(self, input_file=None, sky_model_file=None,
                               baseline=None, cell_size=None, res=None,
                               showGrid=False, add_gauss=False):
        """
        Called by the HTML page when the generate Custom graphs button is pressed.
        The HTML page sends the information entered in the fields to the function and it is
        executed.

        :param input_file: The file that contains the telescope information
        :type input_file: str

        :param sky_model_file: The file that contains the sky model
        :type sky_model_file: str

        :param baseline: The baseline to plot, for information purposes, in the
                         form "x y" where x and y are numbers
        :type baseline: str

        :param cell_size: The cell size for the imaging process
        :type cell_size: str

        :param res: The resolution for the imaging process
        :type res: str

        :param showGrid: Whether or not to show the grid on the image.
        :type showGrid: str

        :returns: nothing
        """
        upload_path = os.path.dirname(__file__)
        if (res != "" and input_file != "" and sky_model_file != ""
                and baseline != "" and cell_size != ""):

            cherrypy.log("Working on Custom Image")

            bl = baseline.split(" ")
            bl_1 = int(bl[0]) - 1
            bl_2 = int(bl[1]) - 1
            input_file = input_file.split("\\")[-1]
            sky_model_file = sky_model_file.split("\\")[-1]
            input_file = os.path.normpath(os.path.join(upload_path, "Antenna_Layouts", input_file))

            with open(input_file) as outfile:
                json_antenna = json.load(outfile)

            custom_layout = np.array(json_antenna['antennas'])
            ut.plot_array(custom_layout, "Custom")
            b = custom_layout[bl_2] - custom_layout[bl_1]
            custom_L = json_antenna['latitude']
            custom_L = (np.pi / 180) * (custom_L[0] + custom_L[1] / 60. + custom_L[2] / 3600.)
            custom_f = json_antenna['frequency']
            custom_f = custom_f * 10 ** 9

            sha = json_antenna['sha']
            eha = json_antenna['eha']
            dec = json_antenna['center_dec']
            dec = dec[0] + dec[1] / 60. + dec[2] / 3600.

            ut.plot_baseline(b, custom_L, custom_f, sha, eha, dec, "CUSTOM")

            add_gauss = add_gauss == "true" if True else False
            uv, uv_tracks, dec_0, ra_0 = ut.get_visibilities(b, custom_L, custom_f, sha, eha, os.path.join(upload_path, "Sky_Models", sky_model_file),
                                                             custom_layout, add_gauss)
            show_grid = showGrid == "true" if True else False
            ut.image(uv, uv_tracks, cell_size, dec_0, res, "CUSTOM", show_grid)

    @cherrypy.expose
    def generate_graphs(self, cell_size=None, loc=None, showGrid=False, calibrate=False):
        """
        Called by the HTML page when the generate Tart graphs button is pressed.
        The HTML page sends the information entered in the fields to the function and it is
        executed.

        :param cell_size: The cell size for the imaging process
        :type cell_size: str

        :param loc: location, which telescope to use, New Zealand or South Africa
        :type loc: str

        :param showGrid: Whether or not to show the grid on the image.
        :type showGrid: str

        :returns: nothing
        """
        cherrypy.log("Working on TART images")

        if cell_size != "" and loc != "":
            cherrypy.log("Retrieiving telescope information")
            if showGrid == "true":
                showGrid = True
            else:
                showGrid = False

            if calibrate == "true":
                calibrate = True
            else:
                calibrate = False

            layout = self.get_antenna_layout(loc)
            L, f = TR.get_latitude_and_frequency(loc)
            visibilities = self.make_vis_matrix(loc, calibrate)

            # Save the most recent visibilities
            location = ''
            if loc == 1:
                location = "ZA"
            else:
                location = "NZ"

            calibrated = ''
            if calibrate:
                calibrated = "Calibrated"
            else:
                calibrated = "Not_Calibrated"

            layout_l_f_visibilties = np.asarray([layout, L, f, visibilities])
            if not os.path.exists('Saved_Visibilities'):
                os.makedirs('Saved_Visibilities')
            fileName = 'Saved_Visibilities' + os.path.sep + 'Visibilities_{:%Y-%m-%d-%H-%M-%S}_'.format(
                datetime.datetime.now()) + "_" + location + "_" + calibrated
            np.save(fileName, layout_l_f_visibilties, allow_pickle=True)

            self.generate(cell_size, layout, L, f, visibilities, showGrid)

        cherrypy.log("Done")

    @cherrypy.expose
    def generate_gif(self, cell_size=None, loc=None, showGrid=False, duration=0):
        """
        Called by the HTML page when the generate Tart GIF button is pressed.
        The HTML page sends the information entered in the fields to the function and it is
        executed. It generates a gif in the backend.

        :param cell_size: The cell size for the imaging process
        :type cell_size: str

        :param loc: location, which telescope to use, New Zealand or South Africa
        :type loc: str

        :param showGrid: Whether or not to show the grid on the image.
        :type showGrid: str

        :param duration: The observation period for the GIF
        :type duration: str

        :returns: nothing
        """
        start_time = round(time.time() * 1000)
        cwd = os.getcwd()
        counter = 0
        if not os.path.exists("".join([cwd, os.path.sep, "GIF", os.path.sep])):
            os.mkdir("".join([cwd, os.path.sep, "GIF", os.path.sep]))

        while round(time.time() * 1000) - start_time < int(duration):
            try:
                begin_minute = round(time.time() * 1000)
                self.generate_graphs(cell_size, loc, showGrid)
                
                shutil.copy("".join([
                    cwd, os.path.sep, "Plots", os.path.sep, "ReconstructedTART Sky Model.png"]), 
                    "".join([cwd, os.path.sep, "GIF", os.path.sep, str(counter) + ".png"])
                    )
                counter += 1

                remaining_ms = 30000 - (round(time.time() * 1000) - begin_minute)
                if remaining_ms > 0:
                    time.sleep(remaining_ms / 1000)
                cherrypy.log(str(counter / 2) + " Minutes into GIF")

            except Exception as e:
                cherrypy.log(e)

        frames = []
        images = [f for f in glob.glob("".join([cwd,  os.path.sep, "GIF", os.path.sep, "*.png"]), recursive=False)]
        images.sort(key=self.natural_keys)

        for im in images:
            new_frame = Image.open(im)
            frames.append(new_frame)
            counter += 1

        frames[0].save("".join([cwd,  os.path.sep, "GIF", os.path.sep, "observation_period.gif"]), format='GIF', append_images=frames[1:], save_all=True,
                       duration=200, loop=0)

        dir = os.listdir("".join([cwd, os.path.sep, "GIF"]))
        for item in dir:
            if item.endswith(".png"):
                os.remove("".join([cwd, os.path.sep, "GIF", os.path.sep, item]))
        cherrypy.log("Done")

    @cherrypy.expose
    def use_saved_visibilities(self, cell_size=None, file_name=None, showGrid=False):
        """
        Called by the HTML page when the Saved Visibilities Generationbutton is pressed.
        The HTML page sends the file selected to the function and it is
        executed.

        :param file_name: The name of the file for the imaging process
        :type file_name: str

        :returns: nothing
        """
        cherrypy.log("Working on TART images from file")
        if cell_size != "":
            if showGrid == "true":
                showGrid = True
            else:
                showGrid = False
            print(showGrid)
            cherrypy.log("Loading data from file")
            upload_path = os.path.dirname(__file__)
            file_name = file_name.split("\\")[-1]
            file_name = os.path.normpath(os.path.join(upload_path, file_name))
            file_name = 'Saved_Visibilities' + os.path.sep + file_name

            layout_l_f_visibilties = np.load(file_name, allow_pickle=True)
            layout = layout_l_f_visibilties[0]
            L = layout_l_f_visibilties[1]
            f = layout_l_f_visibilties[2]
            visibilities = layout_l_f_visibilties[3]

            self.generate(cell_size, layout, L, f, visibilities, showGrid)
        cherrypy.log("Done")

    @cherrypy.expose
    def index(self):
        """
        Tells Cherrypy where to start
        :returns: the rendered template from Jinga
        """
        tmpl = env.get_template('index.html')
        return tmpl.render(target='Imaging pipeline')


if __name__ == '__main__':
    cherrypy.log("Started")
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './Plots'
        }  # This sets the locations of images and such for cherrypy
    }
    cherrypy.quickstart(pipeline(), '/', conf)
