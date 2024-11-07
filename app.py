import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

from plots_and_metrics import (
    calculate_metrics_comparison,
    calculate_metrics_single,
    plot_comparison,
    plot_single,
)

### UI ###
app_ui = ui.page_fluid(
    ui.h2("Wasserstein distance calculator"),
    ui.input_select(
        "user_choice",
        "I want to:",
        choices=["Evaluate a single model", "Compare two models"],
    ),
    ui.output_ui("conditional_ui"),  # Placeholder for conditional UI
)


### Server ###
def server(input: Inputs, output: Outputs, session: Session):

    @output
    @render.ui
    def conditional_ui():
        "ui to let the user choose to evaluate or compare a model."
        if input.user_choice() == "Evaluate a single model":
            return ui.div(
                ui.h3("Evaluating a single model"),
                ui.markdown(
                    """
                            Upload the observed values and predicted values in separate csv files (no headers) below:
                            """
                ),
                ui.input_file(
                    "observations", "Observations", accept=[".csv"], multiple=False
                ),
                ui.input_file(
                    "predictions_model_1",
                    "Predictions",
                    accept=[".csv"],
                    multiple=False,
                ),
                ui.output_text_verbatim("calc_result_single"),
                ui.output_plot("data_plot_single", width="1000px", height="1000px"),
            )
        if input.user_choice() == "Compare two models":
            return ui.div(
                ui.h3("Comparing two models"),
                ui.markdown(
                    """
                            Upload the observed values and predicted values of both models in separate csv files (no headers) below:
                            """
                ),
                ui.input_file(
                    "observations", "Observations", accept=[".csv"], multiple=False
                ),
                ui.input_file(
                    "predictions_model_1",
                    "Predictions Model 1",
                    accept=[".csv"],
                    multiple=False,
                ),
                ui.input_file(
                    "predictions_model_2",
                    "Predictions Model 2",
                    accept=[".csv"],
                    multiple=False,
                ),
                ui.markdown(
                    """
                            Three different plots are displayed. The top one gives a histogram of the data points along with 
                            estimated probability density functions of both the observations and both models. These are estimated using
                            a Gaussian Kernel Density estimator using the Silvestor bandwidth rule. The two figures below provide
                            the emperical CDFs of both models compared to the emperical CDF of the observations. The area between the two, the 
                            Wasserstein distance, is reported for both models with an accompanying 90% CI (calculated using a bootstrap of size 10,000). The ratio
                            of the two models is also reported, also with an accompanying 90% CI made from the same bootstrap samples. A value of 1 indicates equal performance, a value below
                            1 indicates model 1 is superior, a value above 1 indicates model 2 is superior. 

                            Lastly, the predicted-to-observed geometric mean ratio (that is, the geometric mean of the predictions divided by the geometric mean of
                            the observations) is provided for both models with an accompanying 90% CI. This CI is calculated parametrically (see https://doi.org/10.1007/s40262-023-01326-3 
                            for details).
                            """
                ),
                ui.output_text_verbatim("calc_result_comparison"),
                ui.output_plot("data_plot_comparison", width="1800px", height="2500px"),
            )

    @reactive.Calc
    def load_observations():
        "Load the observations from a user-supplied csv file."
        file_info = input.observations()
        if file_info is None:
            return None

        df1 = pd.read_csv(file_info[0]["datapath"])
        observations = df1.to_numpy()[:, 0]
        return observations

    @reactive.Calc
    def load_predictions_model_1():
        "Load data from user supplied file."
        file_info = input.predictions_model_1()
        if file_info is None:
            return None

        df2 = pd.read_csv(file_info[0]["datapath"])
        predictions = df2.to_numpy()[:, 0]
        return predictions

    @reactive.Calc
    def load_predictions_model_2():
        "Load data from user supplied file."
        file_info = input.predictions_model_2()
        if file_info is None:
            return None
        df2 = pd.read_csv(file_info[0]["datapath"])
        predictions = df2.to_numpy()[:, 0]
        return predictions

    @output
    @render.text
    def calc_result_single():
        "Calculate the metrics for the evaluation of a single model"
        observations = load_observations()
        model_1_predictions = load_predictions_model_1()
        # Check if both files are loaded
        if observations is None or model_1_predictions is None:
            return "Please upload both files to see the result."

        # Perform a simple calculation (e.g., sum of values in both files times multiplier)
        GMR, CI_GMR, W1, CI_W1 = calculate_metrics_single(
            model_1_predictions=model_1_predictions, observations=observations
        )
        return f"""
        Geometric-mean ratio: {GMR} {CI_GMR} \n
        Wasserstein distance: {W1} {CI_W1}"""

    @output
    @render.plot
    def data_plot_single():
        "Provide the plot for a single model evaluation."
        observations = load_observations()
        model_1_predictions = load_predictions_model_1()
        # Check if both files are provided
        if observations is None or model_1_predictions is None:
            return
        plot = plot_single(
            observations=observations, predictions=model_1_predictions, xlabel="Value"
        )
        return plot

    @output
    @render.text
    def calc_result_comparison():
        "Calculate the metrics for the comparison of two models."
        observations = load_observations()
        model_1_predictions = load_predictions_model_1()
        model_2_predictions = load_predictions_model_2()
        # Check if both files are provided
        if (
            observations is None
            or model_1_predictions is None
            or model_2_predictions is None
        ):
            return "Please upload all three files to see the result."

        # Calculate the relevant metrics and CIs
        GMR_1, CI_GMR_1, GMR_2, CI_GMR_2, W_1, CI_W_1, W_2, CI_W_2, ratio, CI_ratio = (
            calculate_metrics_comparison(
                observations=observations,
                model_1_predictions=model_1_predictions,
                model_2_predictions=model_2_predictions,
                output=True,
            )
        )
        return f"""
        Geometric-mean ratio model 1: {GMR_1} {[float(CI_GMR_1[0]), float(CI_GMR_1[1])]} \n
        Geometric-mean ratio model 2: {GMR_2} {[float(CI_GMR_2[0]), float(CI_GMR_2[1])]} \n
        Wasserstein distance model 1: {W_1} {[float(CI_W_1[0]), float(CI_W_1[1])]} \n
        Wasserstein distance model 2: {W_2} {[float(CI_W_2[0]), float(CI_W_2[1])]} \n
        Ratio:                        {ratio} {[float(CI_ratio[0]), float(CI_ratio[1])]}"""

    @output
    @render.plot
    def data_plot_comparison():
        observations = load_observations()
        model_1_predictions = load_predictions_model_1()
        model_2_predictions = load_predictions_model_2()
        # Check if both files are provided
        if (
            observations is None
            or model_1_predictions is None
            or model_2_predictions is None
        ):
            return
        plot = plot_comparison(
            observations=observations,
            model_1_predictions=model_1_predictions,
            model_2_predictions=model_2_predictions,
            xlabel="Value",
        )
        return plot


### App ###
app = App(app_ui, server)
