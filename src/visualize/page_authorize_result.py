from dash import dcc, html
from dash.dependencies import Input, Output


class AuthrizeResultVisualizer:
    def __init__(self, result: bool):
        self.result = result
        self.init_id_data()
        self.layout = self._create_layout()

    def get_layout(self):
        return self.layout

    def init_id_data(self):
        self.shared_store_id = "auth-result-store"
        self.interval_id = "auth-result-update"

    def register_callbacks(self, app):

        @app.callback(
            Output(self.shared_store_id, "data"),
            [Input(self.interval_id, "n_intervals")],
        )
        def update_data_store(n):
            return {"auth_result": self.result}

        @app.callback(
            Output("auth-result", "children"),
            Output("auth-result", "className"),
            [Input(self.shared_store_id, "data")],
        )
        def update_auth_result(data):
            if data["auth_result"]:
                return (
                    html.Span(
                        [
                            html.I(className="fas fa-check-circle me-2"),
                            "Authentication Succeeded",
                        ]
                    ),
                    "text-success display-4",
                )
            else:
                return (
                    html.Span(
                        [
                            html.I(className="fas fa-times-circle me-2"),
                            "Authentication Failed",
                        ]
                    ),
                    "text-danger display-4",
                )

    def _create_layout(self):
        return html.Div(
            style={
                "height": "100vh",
                "padding": "20px",
                "box-sizing": "border-box",
                "overflow": "hidden",
            },
            children=[
                html.Div(
                    id="auth-result",
                    className="my-3",
                    style={
                        "position": "absolute",
                        "top": "50%",  # 縦の中央
                        "left": "50%",  # 横の中央
                        "transform": "translate(-50%, -50%)",  # 正確に中央に移動
                        "text-align": "center",
                        "width": "100%",  # 幅を100%に設定してテキストを中央揃え
                    },
                ),
                dcc.Store(id=self.shared_store_id, data=self.result),
                dcc.Interval(id=self.interval_id, interval=1000, n_intervals=0),
            ],
        )
