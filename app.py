import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash_extensions import WebSocket
from dash import dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import time
import edgeiq
from helpers import *
from sample_writer import *
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from websocket import create_connection
import itertools
import json
from collections import deque
from Autoannotate import *
from copy import deepcopy

app = Flask(__name__, template_folder='./templates/')
socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(app, logger=socketio_logger, engineio_logger=socketio_logger)
SAMPLE_RATE = 25
SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.FileVideoStream("costcoVideo.mp4", play_realtime=True)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@socketio.on('write_data')
def write_data():
    controller.start_writer()
    socketio.sleep(0.05)
    controller.update_text('Data Collection Started')
    #file_name = file_set_up("video", SESSION)

    # with edgeiq.VideoWriter(output_path=file_name, fps=SAMPLE_RATE, codec='H264') as video_writer:
    #     if SAMPLE_RATE > video_stream.fps:
    #         raise RuntimeError(
    #             "Sampling rate {} cannot be greater than the camera's FPS {}".
    #             format(SAMPLE_RATE, video_stream.fps))

    print('Data Collection Started')
        # while True:
        #     t_start = time.time()
        #     frame = controller.cvclient.video_frames.popleft()
        #     video_writer.write_frame(frame)
        #     t_end = time.time() - t_start
        #     t_wait = (1 / SAMPLE_RATE) - t_end
        #     if t_wait > 0:
        #         time.sleep(t_wait)
        #     time.sleep(0.01)
        #     if controller.is_writing() == False:
        #         controller.update_text('Data Collection Ended')
        #         print('Data Collection Ended')
        #         break

        # socketio.sleep(0.01)

@socketio.on('stop_writing')
def stop_writing():
    print('stop signal received')
    controller.stop_writer()
    controller.complete_annotations()
    controller.update_text('Data Collection Stopped')
    socketio.sleep(0.01)


@socketio.on('close_app')
def close_app():
    print('Closing App...')
    controller.close_writer()
    controller.close()


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    file = os.path.join(".", get_file(filename))
    return send_file(file, as_attachment=True)

@app.route('/videos', methods=['GET'])
def videos():
    videos = {}
    files = get_all_files()
    if files:
        for f in files:
            videos[f] = (os.path.join(os.path.sep, get_file(f)))
    return render_template('videos.html', videos=videos)

@app.route('/analytics', methods=['GET'])
def analytics():
    return render_template('analytics.html')

@app.route('/view_video/<filename>', methods=['GET'])
def view_video(filename):
    file = os.path.join(os.path.sep, get_file(filename))
    if '.jpeg' in file:
        return render_template('view_video.html', image=file, filename=filename)
    else:
        return render_template('view_video.html', video=file, filename=filename)

@app.route('/delete/<filename>', methods=['GET'])
def delete(filename):
    file = os.path.join(".", get_file(filename))
    if file is not None:
        delete_file(file)
    return redirect(url_for('videos'))

ZONES = [
            'incoming_traffic_cumulative',
            'outgoing_traffic_cumulative',
        ]

ZONE_COMBOS = [(source, sink) for source, sink in itertools.combinations(
    ZONES, 2)]
print(ZONE_COMBOS)
ZONE_COMBO_NUM = len(ZONE_COMBOS)
INCOMING_X = ["Morning","Midday","Evening"]
OUTGOING_X = ["Morning","Midday","Evening"]
INCOMING_Y = [0,0,0]
OUTGOING_Y = [0,0,0]
INCOMING_CARRY_OVER = [0,0]
OUTGOING_CARRY_OVER = [0,0]
INCOMINGDATACOUNT= 0

LAST_VALUES = [0] * len(ZONES)

dash_app = dash.Dash(
    __name__,
    server=app,
    assets_folder="./static",
    url_base_pathname='/analytics/dash/',
    external_stylesheets=[dbc.themes.LUX]
)

dash_app.layout = html.Div([
    WebSocket(
            id='input',
            url='wss://analytics.alwaysai.co?projectId=2fc349e5-ec2c-4a8c-af07-994ed84b5706&apiKey=u3zd2QZD-7kg9jWPwUsCeqNRVnx5bSyJF4~8p@T'
        ),
    dcc.Tabs(id='tabs-example-graph', value='tab-2-example-graph', children=[
        dcc.Tab(
            label='Traffic Flow',
            value='tab-1-example-graph',
            children=[
                dcc.Graph(id='graph',),
            ]),
        dcc.Tab(
            label='Current Occupancy',
            value='tab-2-example-graph',
            id='metric',
            children=[
                dbc.Container(
                    fluid=True,
                    style={'width': '100%', 'height': '10px', 'margin': '0 auto 0 auto'},
                    children=[
                        dbc.Row(
                            # style={'height': '100px', 'position': 'relative'},
                            children=[
                                dbc.Col(
                                    # style={'height': '50px'},
                                    width=12,
                                    children=[
                                        dcc.Graph(id='total')
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            style={'margin': '10px'},
                            children=[
                                dbc.Col(
                                    width=True,
#                                     style={'height': '10px'},
                                    children=[
                                        dbc.Card(
                                            style={'height': '230px'},
                                            color='rgba(48, 230, 223)',
                                            children=[
                                                dbc.CardBody(
                                                    style={'height': '100%', 'width': 'auto'},
                                                    children=[
                                                        dcc.Graph(style={'height': '100%', 'margin': 0, 'padding': 0},id='fig-1')
                                                    ]
                                                )
                                            ],
                                        )
                                    ]
                                ),
                                dbc.Col(
                                    width=True,
                                    # style={'height': '10px'},
                                    children=[
                                        dbc.Card(
                                            style={'height': '230px'},
                                            color='rgba(240, 28, 4)',
                                            children=[
                                                dbc.CardBody(
                                                    style={'height': '100%', 'width': 'auto'},
                                                    children=[
                                                        dcc.Graph(style={'height': '100%', 'margin': 0, 'padding': 0},id='fig-2')
                                                    ]
                                                )
                                            ],
                                        )
                                    ]
                                )

                            ]
                        ),
                    ]
                )
            ]
        ),
    ]),
    dcc.Interval(
        id='timer',
        interval=1000,  # in milliseconds
        n_intervals=0)
])


@dash_app.callback(
    [Output('graph', 'figure'),
     Output('total', 'figure'),
     Output('fig-1', 'figure'),
     Output('fig-2', 'figure')],
    [Input('input', 'message')])
def display_sankey(value):
    global INCOMING
    global OUTGOING
    global INCOMINGDATACOUNT
    global INCOMING_CARRY_OVER
    global OUTGOING_CARRY_OVER
    # replace this will real values
    #raw_data = ws.recv()
    if value and value is not None:
        raw_data = value['data']
        print('raw_data ', raw_data)
        json_data = json.loads(raw_data)
        global LAST_VALUES
        values = [0] * len(ZONES)
        occupancy_results = json_data


        global LAST_VALUES
        values = [0] * len(ZONES)
        occupancy_results = json_data

        for label, count in occupancy_results.items():
            if label != "timestamp" and label != 'project_id' and label != 'device_id' and label != 'custom_results' and label != 'incoming_traffic' and label != 'outgoing_traffic':
                values[ZONES.index(label)] = count
                INCOMINGDATACOUNT += 1
                print(f"INCOMINGDATACOUNT: {INCOMINGDATACOUNT}")


        total_fig = go.Figure(
                        go.Indicator(
                            mode='number+delta',
                            gauge={'shape': 'bullet'},
                            delta={'reference': sum(LAST_VALUES)},
                            value=sum(values),
                            # domain={'x': [0, 1], 'y': [0.0, 0.25]},
                            title={'text': 'Total', 'font': {'size': 60}},
                            number={'font': {'size': 84}}
                        )
                    )

        fig1 = go.Figure(
                    go.Indicator(
                            mode='number',
                            gauge={'shape': 'bullet'},
                            delta={'reference': LAST_VALUES[0]},
                            value=values[0],
                            # domain={'x': [0.0, 0.45], 'y': [0.0, 0.25]},
                            # title_font_color='#f2303a',
                            title={'text': 'Incoming Vehicles', 'font': {'size': 20}},
                            number={'font': {'size': 60}}
                    )
                )

        fig2 = go.Figure(
                    go.Indicator(
                        mode='number',
                        gauge={'shape': 'bullet'},
                        delta={'reference': LAST_VALUES[1]},
                        value=values[1],
                        # domain={'x': [0.5, 1], 'y': [0.0, 0.25]},
                        # title_font_color='#c92c56',
                        title={'text': 'Outgoing Vehicles', 'font': {'size': 20}},
                        number={'font': {'size': 60}}
                    )
                )
        if INCOMINGDATACOUNT < 3000:
            INCOMING_Y[0] = values[0]
            OUTGOING_Y[0] = values[1]
            INCOMING_CARRY_OVER[0] = values[0]
            OUTGOING_CARRY_OVER[0] = values[1]
        if INCOMINGDATACOUNT > 3000 and INCOMINGDATACOUNT < 6000:
            INCOMING_Y[1] = values[0] - INCOMING_CARRY_OVER[0]
            OUTGOING_Y[1] = values[1] - OUTGOING_CARRY_OVER[0]
            INCOMING_CARRY_OVER[1] = values[0] - INCOMING_CARRY_OVER[0]
            OUTGOING_CARRY_OVER[1] = values[1] - OUTGOING_CARRY_OVER[0]
        if INCOMINGDATACOUNT > 6000:
            INCOMING_Y[2] = values[0] - INCOMING_CARRY_OVER[1]
            OUTGOING_Y[2] = values[1] - OUTGOING_CARRY_OVER[1]

        fig = go.Figure()
        incoming_trace = go.Bar(
            x=INCOMING_X,
            y=INCOMING_Y,
            name='Incoming Vehicles',
            text=INCOMING_Y,
            textposition='auto',
            marker=dict(
            color='rgba(0, 0, 246, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )
        )
        outgoing_trace = go.Bar(
            x=OUTGOING_X,
            y=OUTGOING_Y,
            name='Outgoing Vehicles',
            text=OUTGOING_Y,
            textposition='auto',
            marker=dict(
            color='rgba(246, 0, 0, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    )
        )
        fig.add_trace(incoming_trace)
        fig.add_trace(outgoing_trace)
        fig.update_layout(barmode='group', font_size=30)
        fig.update_traces(opacity=0.5)
        fig.update_xaxes(title_text='Time of Day')
        fig.update_yaxes(nticks=5,title_text='Vehicles')


        LAST_VALUES = deepcopy(values)

        return fig, total_fig, fig1, fig2
    raise PreventUpdate


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler, Lila Mullany, and Dalton Varney
        Copyright alwaysAI, Inc. 2022

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self.Ended = False
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.writer = SampleWriter()
        self.all_frames = deque()
        self.video_frames = deque()
        self.dataset_name = "annotated_data"
        self.auto_annotator = AutoAnnotator(confidence_level=0.5, overlap_threshold=0.3, labels=['car'], markup_image=False)
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self
    """
    @param objects (dictionary) From our Object Tracker.  Keys are the id of each object
    values are the predictions of each object
    @return 
    """    
    def average_time_for_lane(self):
        #First obtain how to obtain time for an object.

        #Then sort objects based on the zone they are in.  Use Zone object to get list of
        #objects in a particular zone

        #One you have the time of an object and the lane it populated take that group and 
        #find the average time of that section. (Mean function).  With this you can find
        #the total average as well
        
        return self

    def run(self):
        print("Starting Up")
        global car_count
        lane_one_car_count = 0
        lane_two_car_count = 0
        lane_three_car_count = 0
        lane_four_car_count = 0
        lane_five_car_count = 0

        obj_detect = edgeiq.ObjectDetection(
                "alwaysai/yolo_v3")
        obj_detect.load(engine=edgeiq.Engine.DNN)
        zones = edgeiq.ZoneList("zone_config.json")

        print("Engine: {}".format(obj_detect.engine))
        print("Accelerator: {}\n".format(obj_detect.accelerator))
        print("Model:\n{}\n".format(obj_detect.model_id))
        print("Labels:\n{}\n".format(obj_detect.labels))

        def object_enters(object_id, prediction):
            print("{}: {} enters".format(object_id, prediction.label))


        def object_exits(object_id, prediction):
            print("{} exits".format(prediction.label))
        tracker = edgeiq.KalmanTracker(deregister_frames=12, max_distance=250, min_inertia=1,enter_cb=object_enters,exit_cb=object_exits)
        #tracker = edgeiq.KalmanTracker(deregister_frames=8, max_distance=50, min_inertia=1,enter_cb=object_enters,exit_cb=object_exits)
        fps = edgeiq.FPS()
        objects_in_inzone = []
        objects_in_outzone = []
        objects_in_laneOne = []
        objects_in_laneTwo = []
        objects_in_laneThree = []
        objects_in_laneFour = []
        objects_in_laneFive = []

        video_stream.start()
        # Allow Webcam to warm up
        socketio.sleep(1.0)
        self.auto_annotator.make_directory_structure(self.dataset_name)
        self.fps.start()

        # loop detection
        while True:
            frame = video_stream.read()
            ogframe = deepcopy(frame)
            results = obj_detect.detect_objects(frame, confidence_level=.1)
            predictions = edgeiq.filter_predictions_by_label(results.predictions, ['car','truck'])
            filtered_predictions = []
            for item in predictions:
                has_duplicate = False
                if item.label == 'truck':
                    item.label = 'car'
                for item_comp in predictions:
                    if item_comp.box.start_x == item.box.start_x and item_comp.box.start_y == item.box.start_y:
                        pass
                    else:
                        if (item_comp.box.compute_overlap(item.box)) > 0.1:
                            has_duplicate = True
                            print("Found duplicate")
                if has_duplicate != True:
                    filtered_predictions.append(item)
                    # Generate text to display on streamer
            text = ["Model: {}".format(obj_detect.model_id)]    
            text.append(
                    "Inference time: {:1.3f} s".format(
                        results.duration))
            text.append("Objects:")
            objects = tracker.update(filtered_predictions)
            """
            laneOne = zones.get_zone("laneOne")
            laneTwo = zones.get_zone("laneTwo")
            laneThree = zones.get_zone("laneOne")
            laneFour = zones.get_zone("laneOne")
            laneFive = zones.get_zone("laneOne")
            print(laneOne.get_results_for_zone(objects))
            """
            
            objects_in_zone = [objects_in_laneOne, objects_in_laneTwo, objects_in_laneThree, objects_in_laneFour, objects_in_laneFive]
            lane_Names = ["laneOne", "laneTwo", "laneThree", "laneFour", "laneFive"]
            lane_cumulative_names = ["lane_one_cumulative", "lane_two_cumulative", "lane_three_cumulative", "lane_four_cumulative", "lane_five_cumulative"]
            lane_car_counts = [lane_one_car_count, lane_two_car_count, lane_three_car_count, lane_four_car_count, lane_five_car_count]
            zone_dictionary = {}
            for i in range(5):
                lane = zones.get_zone(lane_Names[i])
                for key, value in objects.items():
                    if lane.check_object_detection_prediction_within_zone(value):
                        if key not in objects_in_zone[i]:
                            lane_car_counts[i] += 1
                            objects_in_zone[i].append(key)
                
                zone_dictionary[lane_cumulative_names[i]] = lane_car_counts[i]
                zone_dictionary[lane_Names[i]] = len(lane.get_results_for_zone(objects))
           
                    
                """
                if laneTwo.check_object_detection_prediction_within_zone(value):
                    if key not in objects_in_outzone:
                        lane_two_car_count += 1
                        objects_in_outzone.append(key)
                """

            text.append(f"Lane One Car Count: {lane_car_counts[0]}")
            text.append(f"Lane Two Car Count: {lane_car_counts[1]}")
            text.append(f"Lane Three Car Count: {lane_car_counts[2]}")
            text.append(f"Lane Four Car Count: {lane_car_counts[3]}")
            text.append(f"Lane Five Car Count: {lane_car_counts[4]}")
            

            #frame = zones.markup_image_with_zones(frame, ['laneOne'], show_labels=False, fill_zones=True, alpha=0.3)
            frame = zones.markup_image_with_zones(frame, lane_Names, show_labels=False, color = (150,150,0), fill_zones=True, alpha=0.3)
            frame = edgeiq.markup_image(
                    frame, filtered_predictions, show_labels=True,
                    show_confidences=False, colors=[(255,255,255)])
            text.append(self.writer.text)
            text.append('\n')
            text.append('\n')

            # enqueue the frames
            """
            zone_dictionary = {}
            zone_dictionary['lane_one_cumulative'] = lane_one_car_count
            zone_dictionary['lane_two_cumulative'] = lane_two_car_count
            zone_dictionary['laneOne'] = len(laneOne.get_results_for_zone(objects))
            zone_dictionary['laneTwo'] = len(laneTwo.get_results_for_zone(objects))
            """
            edgeiq.publish_analytics(zone_dictionary)
            self.all_frames.append(frame)
            if self.writer.write == True:
                self.video_frames.append(ogframe)
                frame2 = deepcopy(ogframe)
                (annotation_xml, frame2, image_name, annotationText) = self.auto_annotator.annotate(frame2, results.predictions)
                self.auto_annotator.write_image(annotation_xml, frame2, image_name)
                start = time.time()
                self.auto_annotator.image_index += 1
            self.send_data(frame, text)

            self.fps.update()

            if self.Ended:
                video_stream.stop()
                break

    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=640, height=480, keep_scale=True)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)#,
                        #'data': get_all_files()
                    })
            socketio.sleep(0.0001)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.writer.close

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()

class Controller(object):
    def __init__(self):
        self.write = False
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('alwaysAI Dashboard on http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.cvclient.Ended = True
        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()
        self.fps.stop()

    def complete_annotations(self):
        self.cvclient.auto_annotator.zip_annotations(self.cvclient.dataset_name)
        print("Zipped Dataset")

    def close_writer(self):
        self.cvclient.writer.write = False
        self.cvclient.writer.close = True

    def start_writer(self):
        self.cvclient.writer.write = True

    def stop_writer(self):
        self.cvclient.writer.write = False
        self.cvclient.writer.close = True

    def is_writing(self):
        return self.cvclient.writer.write

    def update_text(self, text):
        self.cvclient.writer.text = text

controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        print("Program Complete - Thanks for using alwaysAI")
        controller.close()
