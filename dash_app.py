
import dash
from dash import dcc, html
import plotly.graph_objects as go
import zmq
import json
import threading
import plotly
import argparse
import subprocess
import os


def create_dash_app():
    """
    Create the dash app with 3 plot layout: 3d plot, error curves and the final fit.

    return: the dash app
    """

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(children=[
                    dcc.Graph(id='3d-plot', style={'display': 'inline-block'}),
                    dcc.Graph(id='error-curves', style={'display': 'inline-block'}),
                    dcc.Graph(id='final-fit', style={'display': 'inline-block'})
                ])
        ]
    )

    return app


def create_zmq_socket():
    """
    Set up the ZeroMQ socket for receiving data from 
    the optimization script.
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # Bind to the address where the optim script sends data
    socket.bind("tcp://127.0.0.1:5555")
    # Subscribe to all topics
    socket.setsockopt(zmq.SUBSCRIBE, b"") 

    return socket


def create_thread(socket,app):
    """
    Create a background thread to continuously update the figure.

    :param socket: the zmq socket
    :param app: the dash app
    """
    thread = threading.Thread(target=update_figure, args=(socket,app)) #,outq))
    thread.daemon = True
    thread.start()


def update_figure(socket,app):
    """
    Update the dash app figures with the data 
    received from the optimization script.

    :param socket: the zmq socket
    :param app: the dash app
    """
    while True:
        try:
            msg = socket.recv_string()
            data = json.loads(msg)
            fig = plotly.io.from_json(data)
            type_of_plot = fig.data[0].meta["dash_plot"]
            if type_of_plot == "3d-plot":
                app.layout['3d-plot'].figure = fig
            elif type_of_plot == "error-curves":
                app.layout['error-curves'].figure = fig
            elif type_of_plot == "final-fit":
                app.layout['final-fit'].figure = fig
            # return fig
        except Exception as e:
            print("Error while updating figure:", e)


def run_dash_app_as_subprocess(port):
    """
    Run dahs app in a subprocess.
    :param port: port to run the dash app on. 
    The visualization should be available at localhost:<port>
    If running on a remote server, make sure to forward the port 
    locally.
    """
    dash_app_process = subprocess.Popen(["python", 
                                        "dash_app.py", 
                                        "--port", 
                                        f"{port}"],
                                        stderr=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL,
                                        # text=True
                                        )
    dash_app_process_pid = dash_app_process.pid

    return dash_app_process, dash_app_process_pid


def terminate_dash_app_subprocess(dash_app_process,dash_app_process_pid):
    """
    Terminate the dash app subprocess asking nicely first, then forcefully.
    
    :param dash_app_process: the dash app subprocess
    :param dash_app_process_pid: the pid of the dash app subprocess
    """
    
    # ask nicely first
    dash_app_process.kill()
    outs, errs = dash_app_process.communicate()
    if outs:
        print("Dash app not terminated - outs:")
        print(outs)
    if errs:
        print("Dash app not terminated -  errors:")
        print(errs)
    
    # if dash app is still running, kill it forcefully
    dash_app_is_still_running = dash_app_process.poll() is None
    if dash_app_is_still_running:
        print("Forcefully terminating dash app.")
        os.kill(dash_app_process_pid, 0)
    print("Dash app terminated.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    app = create_dash_app()
    socket = create_zmq_socket()
    create_thread(socket,app)

    app.run_server(debug=False, host='localhost', port=args.port)