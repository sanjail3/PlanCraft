import base64
import io
import uuid
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.io as pio
from langchain.tools import StructuredTool
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from PIL import Image
from pydantic.v1 import BaseModel, Field, ValidationError
from scipy import ndimage

from .python_tool import get_python_lc_tool

PLOT_INIT_CODE = """
import pandas as pd
import plotly.graph_objects as go
"""


def is_plot_empty(fig):
    # Export the figure as an image
    img_bytes = pio.to_image(fig, format="png")

    # Open the image using PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Convert image to numpy array
    img_array = np.array(img)

    # Convert to grayscale
    gray_img = np.mean(img_array, axis=2)

    # Apply threshold to separate potential data points from background
    threshold = np.mean(gray_img) * 0.9  # Adjust this value if needed
    binary_img = gray_img < threshold

    # Remove small objects (noise, axis labels, etc.)
    cleaned_img = ndimage.binary_opening(binary_img, structure=np.ones((3, 3)))

    # Count non-background pixels
    non_background_pixels = np.sum(cleaned_img)

    # Calculate the percentage of non-background pixels
    total_pixels = cleaned_img.size
    non_background_percentage = (non_background_pixels / total_pixels) * 100

    # If less than 1% of the image is non-background, consider it empty
    # Adjust this threshold if needed
    return non_background_percentage < 1


def average_trace_width(fig):
    widths = []
    for trace in fig.data:
        if hasattr(trace, "width") and trace.width is not None:
            widths.append(trace.width)

    if widths:
        return sum(widths) / len(widths)
    else:
        return None


def check_and_fix_plot_width(fig):
    if not is_plot_empty(fig):
        return fig

    # Get the average trace width
    average_width = average_trace_width(fig)
    if average_width is None:
        return fig

    # Set the width of all traces to none
    for trace in fig.data:
        trace.width = None

    return fig


def update_fig_axis_range(fig):
    if fig.layout.yaxis.tickformat and "%" in fig.layout.yaxis.tickformat:
        fig = fig.update_layout(yaxis=dict(range=[0, 1]))

    if fig.layout.yaxis.tickmode == "linear":
        fig = fig.update_layout(yaxis=dict(tickmode="auto"))

    fig = fig.update_layout(yaxis=dict(autorange=True), xaxis=dict(autorange=True))
    return fig


def is_valid_variable_name(name):
    try:
        exec(f"{name} = 1")
        return True
    except SyntaxError:
        return False


def escape_curly_braces(string: str):
    return string.replace("{", "{{").replace("}", "}}")


def convert_fig_to_resized_bytes(fig, format="jpg", size=(512, 512)):
    # Convert the figure to image bytes
    img_bytes = pio.to_image(fig, format=format)

    # Open the image using PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Resize the image to the specified size while preserving the aspect ratio
    img.thumbnail(size)

    # Save the resized image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG" if format.lower() == "jpg" else format.upper())
    return img_byte_arr.getvalue()


class IssueTypes(str, Enum):
    NO_ISSUE = "NO_ISSUE"
    EMPTY_PLOT = "EMPTY_PLOT"
    OVERLAPPING_TRACES = "OVERLAPPING_TRACES"
    CLUSTERED_AXIS = "CLUSTERED_AXIS"
    COLOR_PROBLEM = "COLOR_PROBLEM"
    WRONG_PLOT = "WRONG_PLOT"
    OTHER_ISSUE = "OTHER_ISSUE"


class PlotIssue(BaseModel):
    """
    Tool to represent an issue found in a plot
    """

    issue_type: IssueTypes = Field(description="The type of issue found in the plot")
    issue_description: str = Field(
        description="Description of the issue found in the plot"
    )


@traceable
def get_plot_issue(fig, plot_instruction):
    system_message = """
Identify the issue in the plot, call the PlotIssue with the issue type and description.
Plot is geberated using plotly

Instructions:
* OVERLAPPING_TRACES -> If only one trace is visible in the plot among all the traces
* COLOR PROBLEM -> If the colors of the plot are black (happens if the color setting logic is wrong), or all the legends have the same color
* WRONG PLOT -> If the plot type is different from plot instruction
* OTHER_ISSUE -> For any other issue identified not falling in the above categories

* If labels are not visible the it is not an issue
* traces are overlapping, but visible, it is not an issue
* For line plot ignore the overlapping traces issue (overlapping is expected in line plot)
"""
    # Use the new function in the get_plot_issue function
    img_byte_arr = convert_fig_to_resized_bytes(fig)

    image_path = f"image_test/plot_{str(uuid.uuid4())}.jpg"
    with open(image_path, "wb") as f:
        f.write(img_byte_arr)

    human_message = HumanMessage(
        content=[
            {"text": plot_instruction},
            {"image": {"format": "jpeg", "source": {"bytes": img_byte_arr}}},
        ]
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            human_message,
        ]
    )

    chain = prompt_template | ChatBedrock(
        # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs=dict(temperature=0),
        beta_use_converse_api=True,
    ).with_structured_output(PlotIssue)

    result = chain.invoke({})
    return result


class PlotCodeResponse(BaseModel):
    code: str = Field(description="Python code to generate the plot")
    plot_variable: str = Field(
        description="Name of the variable containing the plot object"
    )


class PlotGenerator:
    def __init__(self, source_python_tool: Callable, plot_folder: Path, max_retries=3):
        self.source_python_tool = source_python_tool
        self.plot_folder = Path(plot_folder)
        self.source_local = self.source_python_tool.func.__self__.locals
        self.max_retries = max_retries

    def generate_plot(self, plot_instruction, variable_name, query=None):

        if query:
            raise NotImplementedError("Query is not supported yet")

        if not is_valid_variable_name(variable_name):
            return {
                "observation": f"{variable_name} is not a valid variable name",
                "metadata": {
                    "error": True,
                },
            }
        if variable_name not in self.source_local:
            return {
                "observation": f"{variable_name} not found in the python environment",
                "metadata": {
                    "error": True,
                },
            }

        df = self.source_local[variable_name]

        if not isinstance(df, pd.DataFrame):
            return {
                "observation": f"{variable_name} is not a pandas dataframe",
                "metadata": {
                    "error": True,
                },
            }

        plot_system_prompt = """
Write python code to generate the plot using plotly

pre-executed code:
>>>
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv(csv_path)
<<< 

df.sample:
----
{df_sample}

df.dtype
----
{df_dtype}

df.nunique
----
{df_nunique}

len(df)
----
{df_len}


execute python code to create plotly plot using the instruction provided

In the plot_instruction, if user is refering specific variable for the dataframe, still use df as the variable name

Plotly Instructions Code Generation:
>>>
* Take extra care in making the plot aesthetically pleasing
* Always preser light theme for the charts
* If there are multiple traces, make sure to have a legend and labels for each trace, and it should display the label on hover
* Make sure x axis is not crouwded with labels (don't put tickmode linear for large number of data points)
* Never put ID columns on y-axis
* Never set the colors manually
* Only create the plot object do not show the plot (i.e do not execute .show())
* If y axis is percentage, use tickmode with \% and set the range to [0, 1] (eg. fig.update_yaxes(tickmode='linear', range=[0, 1]))
    * Do not do range [0, 100] as the api has a bug with this range
* Do not update the width of the traces as it can make the plot invisible if the width is too small, eg. do not do fig,update_traces(width=0.8)
* do not set hovermode='x unified' unless asked
* For overlapping bar plots (or anythoer plot) make sure to set alpha so that the overlapping is visible
* Do not attempt to create the variable `df` again (it already exists), start execution after the pre-executed code.
* ALways represents percentages in 0 to 1 in dataframe, if the percentage is in range 0 to 100, convert it to 0 to 1 before plotting
<<<
return JSON output to 2 keys `code` & `plot_variable`

plot_variable: name of python variable containing the plot object

use the PlotCodeResponse tool to generate the output code & plot_variable as tool input
"""

        df_len = len(df)
        sample_size = min(df_len, 10)
        plot_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", plot_system_prompt),
                ("human", "{input}"),
            ]
        ).partial(
            df_sample=df.sample(sample_size).to_string(),
            df_len=df_len,
            df_dtype=df.dtypes.to_csv(header=False),
            df_nunique=df.nunique().to_string(),
        )

        chain = plot_prompt_template | ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            model_kwargs=dict(temperature=0),
            # beta_use_converse_api=True,
        ).with_structured_output(PlotCodeResponse)

        plot_gen_result = None
        plot_code_result = None
        error_str = None
        for _ in range(self.max_retries):
            if plot_gen_result:
                code = escape_curly_braces(plot_gen_result.code)
                plot_instruction += f"\nPrevious Code:\n {code}\n\nError: {error_str}\nFix the error and try again"
            error_str = None
            try:
                plot_gen_result: PlotCodeResponse = chain.invoke(
                    {"input": plot_instruction}
                )
            except ValidationError as e:
                error_str = str(e)
                continue

            plot_py_tool = get_python_lc_tool(
                init_code=PLOT_INIT_CODE, reject_plot=False
            )
            plot_py_tool.func.__self__.locals["df"] = self.source_local[variable_name]
            plot_code_result = plot_py_tool.invoke({"code": plot_gen_result.code})
            if plot_code_result.get("metadata", {}).get("error"):
                error_str = escape_curly_braces(plot_code_result["observation"])
                continue

            fig = plot_py_tool.func.__self__.locals[plot_gen_result.plot_variable]
            fig = update_fig_axis_range(fig)
            fig = check_and_fix_plot_width(fig)

            if is_plot_empty(fig):
                error_str = "Plot is empty, please update the code to fix it"
                continue

            plot_issue_result = get_plot_issue(fig, plot_instruction)

            if plot_issue_result.issue_type != IssueTypes.NO_ISSUE:
                error_str = plot_issue_result.issue_description
                if error_str == IssueTypes.OVERLAPPING_TRACES:
                    error_str += "This can happen due to fixed y axis range, try to make it dynamic or have transparancy in the plot or ignore some part of the instruction to improve the plot"
                continue

            if not error_str:
                break
        else:
            if not fig:
                return {
                    "observation": "Failed to generate plot please simplify the instruction or the data",
                    "metadata": {
                        "error": True,
                    },
                }

        # fig = plot_py_tool.func.__self__.locals[plot_gen_result.plot_variable]
        image_path = self.plot_folder / f"{str(uuid.uuid4())}.jpg"
        pio.write_image(fig, image_path, format="jpg")

        return {
            "observation": None,
            "metadata": {
                "image_path": image_path,
                "plot_gen_code": plot_gen_result.code,
                "plotly_fig": fig,
            },
        }


class PlotGeneratorInputs(BaseModel):
    plot_instruction: str = Field(
        description="instruction of the kind of plot we are looking to generate"
    )
    variable_name: str = Field(
        description="which variable to use as source table to the python environment"
    )


class PlotGeneratorInputsWithQuery(BaseModel):
    query: str = Field(
        description="what question we are trying to answer from the plot"
    )


def get_plot_gen_lc_tool(
    source_python_tool: Callable, plot_folder: Path, with_query=True
):
    plot_generator = PlotGenerator(
        source_python_tool=source_python_tool, plot_folder=plot_folder
    )
    description = "Tool to generate plot of a dataframe using the instruction provided"
    return StructuredTool.from_function(
        plot_generator.generate_plot,
        name="plot_generator",
        description=description,
        args_schema=PlotGeneratorInputsWithQuery if with_query else PlotGeneratorInputs,
    )
