#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import dash
import dash_bootstrap_components as dbc


class BaseApp(object):
    """
    Base class for applications.
    """

    def __init__(self, name):
        super(BaseApp, self).__init__()

        # Create basic app
        self.app = dash.Dash(
            name=name, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.app.layout = dbc.Container(
            children=[
                dbc.Alert(
                    children="No app content has been added!",
                    color="primary"),
                ],
            style={'marginTop': '15em'},)

    def start_application(self, debug=False):
        self.app.run_server(debug=debug)


# For simple debugging the app can be launched by executing the python file.
if __name__ == "__main__":
    app = BaseApp(name='test')
    app.start_application(debug=True)
