import ast
import json
import re
import sys
import threading
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Union

import astor
import numpy as np
import pandas as pd
from langchain.tools import StructuredTool
from loguru import logger
from pydantic.v1 import BaseModel, Field
from rapidfuzz import fuzz


class PandasSdtoutHandler:
    @staticmethod
    def pandas_json_serializer(obj: Any) -> Dict[str, Union[str, List, Dict]]:
        """
        Serializes various pandas and numpy objects into JSON-compatible dictionaries.

        Parameters:
        obj (Any): The object to serialize. This can be a pandas Timestamp, Timedelta, Period, Interval,
                   Categorical, DataFrame, Series, or a numpy integer or floating point number.

        Returns:
        Dict[str, Union[str, List, Dict]]: A dictionary representing the serialized form of the input object.
                                            The dictionary contains a "__type__" key indicating the type of the
                                            object and a "value" key containing the serialized value. Additional
                                            keys may be present depending on the type of the object.
        """
        if isinstance(obj, pd.Timestamp):
            return {"__type__": "Timestamp", "value": obj.isoformat()}
        elif isinstance(obj, pd.Timedelta):
            return {"__type__": "Timedelta", "value": str(obj)}
        elif isinstance(obj, pd.Period):
            return {"__type__": "Period", "value": str(obj)}
        elif isinstance(obj, pd.Interval):
            return {"__type__": "Interval", "value": str(obj)}
        elif isinstance(obj, pd.Categorical):
            return {
                "__type__": "Categorical",
                "value": obj.to_list(),
                "categories": obj.categories.to_list(),
                "ordered": obj.ordered,
            }
        elif isinstance(obj, pd.DataFrame):
            return {"__type__": "DataFrame", "value": obj.to_dict(orient="split")}
        elif isinstance(obj, pd.Series):
            return {
                "__type__": "Series",
                "value": obj.to_list(),
                "name": obj.name,
                "index": obj.index.to_list(),
            }
        elif isinstance(obj, np.integer):
            return {"__type__": "np.integer", "value": int(obj)}
        elif isinstance(obj, np.floating):
            return {"__type__": "np.floating", "value": float(obj)}
        return {"__type__": "other", "value": str(obj)}

    @staticmethod
    def pandas_json_deserializer(obj: Dict[str, Any]) -> Any:
        """
        Deserialize a JSON object into corresponding pandas or numpy objects.

        This function checks for a special key "__type__" in the input dictionary to determine
        the type of object to deserialize. It supports deserialization of various pandas and
        numpy types including Timestamp, Timedelta, Period, Interval, Categorical, DataFrame,
        Series, np.integer, and np.floating.

        Parameters:
        obj (Dict[str, Any]): The JSON object to deserialize.

        Returns:
        Any: The deserialized pandas or numpy object, or the original object if no special type is found.
        """
        if "__type__" not in obj:
            return obj

        obj_type = obj["__type__"]
        if obj_type == "Timestamp":
            return pd.Timestamp(obj["value"])
        elif obj_type == "Timedelta":
            return pd.Timedelta(obj["value"])
        elif obj_type == "Period":
            return pd.Period(obj["value"])
        elif obj_type == "Interval":
            return obj["value"]  # Keep it as a string
        elif obj_type == "Categorical":
            return pd.Categorical(
                obj["value"], categories=obj["categories"], ordered=obj["ordered"]
            )
        elif obj_type == "DataFrame":
            return pd.DataFrame.from_dict(obj["value"], orient="split")
        elif obj_type == "Series":
            return pd.Series(obj["value"], name=obj["name"], index=obj["index"])
        elif obj_type == "np.integer":
            return np.int64(obj["value"])
        elif obj_type == "np.floating":
            return np.float64(obj["value"])
        elif obj_type == "other":
            return obj["value"]
        return obj

    @staticmethod
    def extract_serialized_df_str(serialized_str: str) -> List[Dict[str, Any]]:
        """
        Extracts and deserializes DataFrame or Series objects from a serialized string.

        This function processes a string containing serialized DataFrame or Series objects
        wrapped in <serialized_df> tags. It splits the string into blocks, deserializes the
        DataFrame or Series objects, and returns a list of dictionaries representing the
        extracted data.

        Args:
            serialized_str (str): The input string containing serialized DataFrame or Series objects.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a block
            of extracted data. Each dictionary contains:
                - "type" (str): The type of the block, either "text", "dataframe", or "series".
                - "data" (Any): The extracted data, either a string for text blocks or a DataFrame/Series for data blocks.
                - "total_rows" (int, optional): The total number of rows in the DataFrame/Series (only for data blocks).
                - "trimmed" (bool, optional): Whether the DataFrame/Series was trimmed (only for data blocks).
        """
        # Extract the serialized DataFrame string
        serialized_str = serialized_str.strip()
        split_blocks = re.split(
            r"(<serialized_df>.*?</serialized_df>)", serialized_str, flags=re.DOTALL
        )
        split_blocks = [block.strip() for block in split_blocks]

        result_blocks = []
        for block in split_blocks:
            if not block.startswith("<serialized_df>"):
                result_blocks.append(
                    {
                        "type": "text",
                        "data": block,
                    }
                )
            if block.startswith("<serialized_df>"):
                serialized_str = re.sub(
                    r"^<serialized_df>|</serialized_df>$", "", block
                )
                data = json.loads(
                    serialized_str,
                    object_hook=PandasSdtoutHandler.pandas_json_deserializer,
                )
                df = pd.DataFrame(data["data"])
                for col, dtype in data["data_types"].items():
                    df[col] = df[col].astype(dtype)
                result_blocks.append(
                    {
                        "type": "series" if data["is_series"] else "dataframe",
                        "data": df,
                        "total_rows": data["total_rows"],
                        "trimmed": data["trimmed"],
                    }
                )
        return result_blocks

    @staticmethod
    def blocks_to_sdtout_str(
        blocks: List[Dict[str, Any]], max_rows=10, max_cols=40
    ) -> str:
        """
        Converts a list of blocks containing text, DataFrame, or Series data into a formatted string.

        Args:
            blocks (List[Dict[str, Any]]): A list of dictionaries where each dictionary represents a block of data.
                Each block must have a "type" key which can be "text", "dataframe", or "series".
                - For "text" type, the block should contain a "data" key with the text content.
                - For "dataframe" type, the block should contain "data" (a DataFrame object), "total_rows" (int), and "trimmed" (bool).
                - For "series" type, the block should contain "data" (a Series object), "total_rows" (int), and "trimmed" (bool).
            max_rows (int, optional): The maximum number of rows to display for DataFrame and Series objects. Defaults to 10.
            max_cols (int, optional): The maximum number of columns to display for DataFrame and Series objects. Defaults to 40.

        Returns:
            str: A formatted string representation of the blocks.
        """
        result = []
        for block in blocks:
            if block["type"] == "text":
                result.append(block["data"])
            elif block["type"] == "dataframe":
                result.append(
                    f"DataFrame (total_rows={block['total_rows']}, trimmed={block['trimmed']})"
                )
                result.append(
                    block["data"].to_string(max_rows=max_rows, max_cols=max_cols)
                )
            elif block["type"] == "series":
                result.append(
                    f"Series (total_rows={block['total_rows']}, trimmed={block['trimmed']})"
                )
                result.append(
                    block["data"]
                    .squeeze(axis=1)
                    .to_string(dtype=True, name=True, max_rows=max_rows)
                )
        return "\n\n".join(result)

    @staticmethod
    def serializer_str_to_stdout_str(
        serialized_str: str, max_rows=10, max_cols=40
    ) -> str:
        """
        Converts a serialized string to a formatted string for standard output.

        This function deserializes the serialized DataFrame or Series objects in the input string,
        converts them to a formatted string, and returns the result.

        Args:
            serialized_str (str): The input string containing serialized DataFrame or Series objects.
            max_rows (int, optional): The maximum number of rows to display for DataFrame and Series objects. Defaults to 10.
            max_cols (int, optional): The maximum number of columns to display for DataFrame and Series objects. Defaults to 40.

        Returns:
            str: A formatted string representation of the deserialized DataFrame or Series objects.
        """
        blocks = PandasSdtoutHandler.extract_serialized_df_str(serialized_str)
        return PandasSdtoutHandler.blocks_to_sdtout_str(
            blocks, max_rows=max_rows, max_cols=max_cols
        )

    @staticmethod
    def get_serialized_df_str(df: pd.DataFrame | pd.Series, keep_rows=30) -> str:
        """
        Serializes a pandas DataFrame or Series into a JSON string with additional metadata.

        Parameters:
        df (pd.DataFrame | pd.Series): The DataFrame or Series to be serialized.
        keep_rows (int, optional): The number of rows to keep in the serialized output.
                                   If the DataFrame has more rows than this value,
                                   it will keep the top half and bottom half of the specified rows.
                                   Defaults to 30.

        Returns:
        str: A JSON string containing the serialized DataFrame or Series along with metadata:
             - "data": The data of the DataFrame or Series in dictionary form.
             - "columns": List of column names.
             - "data_types": Dictionary of column data types.
             - "is_series": Boolean indicating if the input was a Series.
             - "trimmed": Boolean indicating if the DataFrame was trimmed.
             - "total_rows": The total number of rows in the original DataFrame or Series.
        """
        is_series = False
        trimmed = False
        total_rows = len(df)
        if isinstance(df, pd.Series):
            is_series = True
            df = df.to_frame()
        # if index dtype is object, convert to string
        if df.index.dtype == "O":
            df.index = df.index.astype(str)
        if len(df) > keep_rows:
            # keep top keep_rows/2 and bottom keep_rows/2 rows
            df = pd.concat([df.head(keep_rows // 2), df.tail(keep_rows // 2)])
            trimmed = True
        serialized_df_dict = {
            "data": df.to_dict(),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
            "is_series": is_series,
            "trimmed": trimmed,
            "total_rows": total_rows,
        }
        return f"<serialized_df>{json.dumps(serialized_df_dict, default=PandasSdtoutHandler.pandas_json_serializer)}</serialized_df>"


def get_exec_traceback(code_string, exc_info):
    """
    Generates a formatted traceback string from the given exception information.

    Args:
        code_string (str): The original code string where the exception occurred.
        exc_info (tuple): A tuple containing exception type, exception value, and traceback object, output of sys.exc_info()

    Returns:
        str: A formatted string representing the traceback, only including the lines of user's code_string where the exception occurred.
    """
    exc_type, exc_value, exc_traceback = exc_info

    tb_lines = ["Traceback:"]
    for frame_summary in traceback.extract_tb(exc_traceback):
        # Check if the frame is from the user's code_string
        if frame_summary.filename == "<string>":
            lineno = frame_summary.lineno

            # Get the line from our original string
            lines = code_string.splitlines()
            line = lines[lineno - 1] if 0 < lineno <= len(lines) else "Unknown"

            tb_lines.append(f"    {line.strip()}")

    tb_lines.append(f"{exc_type.__name__}: {exc_value}")

    return "\n".join(tb_lines)


def convert_triple_quoted_to_single_line(code):
    """
    Converts triple-quoted strings in the given Python code to single-line strings.

    This function parses the provided Python code, identifies triple-quoted strings,
    and replaces newline characters within those strings with the escape sequence '\\n'.
    The transformed code is then returned as a string.

    Args:
        code (str): The Python code containing triple-quoted strings to be converted.

    Returns:
        str: The transformed Python code with triple-quoted strings converted to single-line strings.
    """
    tree = ast.parse(code)

    class TripleQuoteTransformer(ast.NodeTransformer):
        def visit_Str(self, node):
            # Check if the node is a string constant
            if isinstance(node, ast.Constant) and isinstance(node.s, str):
                # Replace newlines with the escape sequence
                new_lines_str = node.s.replace("\n", "\\n")
                node.s = new_lines_str
            return node

    transformer = TripleQuoteTransformer()
    transformed_tree = transformer.visit(tree)

    # Convert the transformed AST back to source code
    return astor.to_source(transformed_tree)


# code syntax check
def is_correct_syntax(code: str) -> bool:
    """
    Checks if the given code string has correct Python syntax.

    Args:
        code (str): The Python code as a string to be checked.

    Returns:
        bool: True if the code has correct syntax, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def sanitize_python_code(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & python from start
    # query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    # query = re.sub(r"(\s|`)*$", "", query)
    return query
    # return query.replace("\\n", "\n")


INSECURE_LIBRARIES = {
    "subprocess",
    "os",
    "eval",
    "pickle",
    "shlex",
    "sys",
    "tempfile",
    "popen",
    "pipe",
    "fcntl",
    "termios",
    "ctypes",
    "socket",
    "http.server",
    "http.client",
    "multiprocessing",
    "imp",
    "imp.load_source",
    "requests",
    "xml.etree.ElementTree",
    "urllib.request",
    "urllib.urlopen",
    "urllib.urlretrieve",
}

INSECURE_FUNCTIONS = {
    "exec",
    "eval",
    "pickle.loads",
    "input",
    "__import__",
    "os.system",
    "sh",
    "system",
    "getpass",
    "fileinput",
    "imaplib",
    "poplib",
    "smtplib",
    "telnetlib",
    "ftplib",
    "http.client",
    "http.server",
    "sqlite3",
    "mysql.connector",
    "paramiko",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.check_output",
    "subprocess.getstatusoutput",
    "os.popen",
    "os.spawn",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.environ",
}


def extract_modules_functions(code: str):
    """
    Extracts all modules and functions from the given code.

    This function parses the provided code to identify all imported modules and called functions.
    It returns the functions with their full names if they are methods of a module.

        code (str): The code to extract modules and functions from.

        tuple: A tuple containing two sets:
            - The first set contains the names of the imported modules.
            - The second set contains the names of the called functions, with methods prefixed by their module names.

    Example:
        code = '''
        os.path.join('a', 'b')
        print('hello')
        '''
        modules, functions = extract_modules_functions(code)
        # modules: {'os', 'sys'}
        # functions: {'os.path.join', 'print'}
    """
    # Extracts all modules and functions from the code
    modules = set()
    functions = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # if node.func.value is instance of ast.Call it does not have attribute id, (so node.func.value.id will throw attribute error)
                if isinstance(node.func.value, ast.Name):
                    functions.add(f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node.func, ast.Name):
                functions.add(node.func.id)
    return modules, functions


def is_code_insecure(code: str) -> bool:
    """
    Analyzes the provided code to determine if it uses any insecure modules or functions.

    This function extracts the modules and functions used in the given code and checks
    them against predefined sets of insecure libraries and functions. If any insecure
    modules or functions are detected, a warning is logged and the function returns True.

    Args:
        code (str): The source code to be analyzed.

    Returns:
        bool: True if the code contains insecure modules or functions, False otherwise.
    """
    modules, functions = extract_modules_functions(code)
    insecure_modules = modules.intersection(INSECURE_LIBRARIES)
    insecure_functions = functions.intersection(INSECURE_FUNCTIONS)
    if insecure_modules or insecure_functions:
        logger.warning(
            f"Insecure code detected: {insecure_modules or insecure_functions}"
        )
        return True
    return False


INIT_CODE = """
import nltk
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import statsmodels
import re
import collections

"""


def get_csv_df_repr(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to its CSV string representation.

    Parameters:
    df (pd.DataFrame): The DataFrame to be converted to CSV format.

    Returns:
    str: A string containing the CSV representation of the DataFrame, without the index.
    """
    return df.to_csv(index=False)


def get_init_variables() -> Dict[str, Any]:
    """
    Initializes and returns a dictionary of commonly used libraries and settings for pandas.

    This function sets various display options for pandas DataFrames and Series,
    and modifies their string representation methods to use a custom JSON serializer.
    It also imports and returns a set of commonly used libraries.

    Returns:
        Dict[str, Any]: A dictionary containing initialized libraries and settings.
    """
    import collections
    import re

    import nltk
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import sklearn
    import statsmodels

    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)

    # Override the default string representation of pandas objects
    pd.DataFrame.__repr__ = PandasSdtoutHandler.get_serialized_df_str
    pd.DataFrame.__str__ = PandasSdtoutHandler.get_serialized_df_str
    pd.Series.__repr__ = PandasSdtoutHandler.get_serialized_df_str
    pd.Series.__str__ = PandasSdtoutHandler.get_serialized_df_str

    return {
        "np": np,
        "pd": pd,
        "sklearn": sklearn,
        "statsmodel": statsmodels,
        "st": st,
        "nltk": nltk,
        "re": re,
        "collections": collections,
    }


code_exec_lock = threading.Lock()


class PythonTool:
    def __init__(
        self,
        locals: Dict = None,
        globals: Dict = None,
        sanitize_input=True,
        init_code=None,
        reject_plot=True,
        reject_dataframe_creation=False,
        attempt_dataframe_creation_removal=True,
    ):
        self.locals = locals or {}
        self.globals = globals or {}
        self.sanitize_input = sanitize_input
        self.reject_plot = reject_plot
        self.reject_dataframe_creation = reject_dataframe_creation
        self.attempt_dataframe_creation_removal = attempt_dataframe_creation_removal
        self.locals.update(get_init_variables())
        self.execution_history = INIT_CODE
        if init_code:
            self.execute_code(init_code)
            self.update_execution_history(init_code)

    def execute_code(self, code):
        """
        Executes the provided Python code string in a controlled environment.

        This method parses the provided code, separates it into all but the last expression and the last expression,
        and then executes them in sequence. The output and any return value from the last expression are captured and returned.
        If an error occurs during execution, it attempts to handle it gracefully and returns the error information.

        Args:
            code (str): The Python code to be executed.

        Returns:
            dict: A dictionary containing the following keys:
            - "std_out" (str): The standard output captured during code execution.
            - "ret" (Any): The return value of the last expression, if successfully evaluated.
            - "error" (bool): A flag indicating whether an error occurred during execution.
        """
        # Ensure thread safety by acquiring the lock
        with code_exec_lock:
            # Parse the code into an AST (Abstract Syntax Tree)
            tree = ast.parse(code)

            # Separate the code into all but the last expression
            code_before_last_expr = ast.Module(body=tree.body[:-1], type_ignores=[])
            code_before_last_expr_str = ast.unparse(code_before_last_expr)

            # Isolate the last expression
            last_expr = ast.Module(body=[tree.body[-1]], type_ignores=[])
            last_expr_str = ast.unparse(last_expr)

            # Create a buffer to capture standard output
            io_buffer = StringIO()
            try:
                # Redirect standard output to the buffer
                with redirect_stdout(io_buffer):
                    # Execute all but the last expression
                    exec(ast.unparse(code_before_last_expr), self.locals, self.locals)
                    try:
                        # Use eval for the last expression and capture its return value
                        ret = eval(last_expr_str, self.globals, self.locals)
                        return {
                            "std_out": io_buffer.getvalue(),
                            "ret": ret,
                            "error": False,
                        }
                    except Exception as e:
                        # If eval fails, log the warning and retry with exec
                        logger.warning(
                            f"eval failed with error {e}, retrying with exec"
                        )
                        try:
                            # Execute the last expression
                            exec(last_expr_str, self.globals, self.locals)
                            return {
                                "std_out": io_buffer.getvalue(),
                                "error": False,
                            }
                        except Exception as e:
                            # Capture and return the traceback if exec fails
                            traceback_str = get_exec_traceback(
                                last_expr_str, sys.exc_info()
                            )
                            return {
                                "std_out": io_buffer.getvalue() + "\n" + traceback_str,
                                "error": True,
                            }

            except Exception:
                # Capture and return the traceback if the initial exec fails
                traceback_str = get_exec_traceback(
                    code_before_last_expr_str, sys.exc_info()
                )
                return {
                    "std_out": io_buffer.getvalue() + "\n" + traceback_str,
                    "error": True,
                }
            finally:
                # Close the buffer
                io_buffer.close()

    def update_execution_history(self, code: str) -> None:
        """
        Updates the execution history with the provided code.

        Args:
            code (str): The code to be appended to the execution history.
        """
        self.execution_history += "\n" + code

    def create_dateframe_in_code(self, code: str) -> bool:
        """
        Checks if the provided code contains any blacklisted pandas functions
        related to DataFrame creation.

        Args:
            code (str): The code to be analyzed.

        Returns:
            bool: True if any blacklisted functions are detected in the code,
              False otherwise.
        """
        blacklisted_functions = {
            "pd.read_csv",
            "pd.read_excel",
            "pd.read_sql",
            "pd.DataFrame",
            "pd.read_sql_query",
        }
        _, detected_functions = extract_modules_functions(code)
        if detected_functions.intersection(blacklisted_functions):
            return True
        return False

    def plot_logic_in_code(self, code: str) -> bool:
        """
        Checks if the provided code contains any plotting logic.

        This method analyzes the code to detect the use of blacklisted plotting libraries
        such as matplotlib, seaborn, and plotly. If any of these libraries are detected,
        it returns True, indicating that the code contains plotting logic.

        Args:
            code (str): The Python code to be analyzed.

        Returns:
            bool: True if the code contains plotting logic, False otherwise.
        """
        blacklisted_modules = {"matplotlib", "seaborn", "plotly"}
        detected_modules, _ = extract_modules_functions(code)
        # Filter submodules from detected_modules
        detected_modules = {module.split(".")[0] for module in detected_modules}
        if detected_modules.intersection(blacklisted_modules):
            return True
        return False

    def get_sql_dataframe_variables(self) -> List[str]:
        """
        Retrieve a list of variable names from the local scope that are either
        DataFrame objects or have names starting with "sql_result_df_".

        Returns:
            List[str]: A list of variable names that match the criteria.
        """
        return [
            var
            for var in self.locals
            if var.startswith("sql_result_df_")
            or isinstance(self.locals[var], pd.DataFrame)
        ]

    def filter_sql_dataframe_variables(self, code: str) -> str:
        """
        Filters out specific SQL dataframe variables from the provided code.

        This method identifies and removes variables that match the pattern
        'sql_result_df_{idx}' where {idx} is a digit, and the variable is
        assigned using the pandas `read_sql_query` function. The method
        ensures that these variables are not present in the final code string.

        Args:
            code (str): The input code string that may contain SQL dataframe
                variables.

        Returns:
            str: The filtered code string with specific SQL dataframe variables
             removed.
        """
        # Todo: Use llm to generate the code
        # Extremely HACKY
        sql_dataframe_variables = self.get_sql_dataframe_variables()
        if not any(var in code for var in sql_dataframe_variables):
            return code
        code = convert_triple_quoted_to_single_line(code)
        # remove all the variables that are pandas dataframe name is of pattern sql_result_df_{idx}
        pattern = r"sql_result_df_\d+\s*=\s*pd\.read_sql_query\((?:.|\s)*?\)\s*\n"
        filtered_code = re.sub(pattern, "", code)
        return filtered_code

    def __call__(self, code: str) -> Dict[str, Any]:
        """
        Executes the provided Python code string in a controlled environment.

        This method sanitizes the input code, checks for insecure imports or function calls,
        and optionally filters out code that creates dataframes or plots. It then executes
        the code and returns the output along with metadata.

        Args:
            code (str): The Python code to be executed.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
            - "observation" (str): The standard output captured during code execution.
            - "metadata" (Dict[str, Any]): Metadata about the execution, including any errors.
        """
        if self.sanitize_input:
            code = sanitize_python_code(code)

        execution_code = code
        metadata = {}

        try:
            # Check for insecure imports or function calls
            if is_code_insecure(code):
                metadata["error"] = True
                return {
                    "observation": "Due to security reasons, we cannot execute this code (insecure imports or function calls)",
                    "metadata": metadata,
                }

            # Check for plotting logic
            if self.reject_plot and self.plot_logic_in_code(code):
                metadata["error"] = True
                return {
                    "observation": "Code creates a plot, which is not allowed, please use the plot generator tool",
                    "metadata": metadata,
                }

            # Check for dataframe creation
            if self.reject_dataframe_creation and self.create_dateframe_in_code(code):
                logger.warning("Code creates a dataframe")
                sql_dataframe_variables = self.get_sql_dataframe_variables()
                filtered_code = self.filter_sql_dataframe_variables(code)
                if not is_correct_syntax(filtered_code):
                    logger.warning("Code syntax error after filtering")
                if (
                    not self.attempt_dataframe_creation_removal
                    or not is_correct_syntax(filtered_code)
                ):
                    return {
                        "observation": f"Code creates a dataframe, which is not allowed, please use one of the following pre-existing dataframe variables: {sql_dataframe_variables}, do not initialize/re-initialize it as it already exists",
                        "metadata": {"error": True},
                    }
                execution_code = filtered_code
                if filtered_code != code:
                    logger.warning("Code transformed due to dataframe creation")

            metadata["executed_code"] = execution_code
            execute_result = self.execute_code(execution_code)
            metadata["raw_std_log"] = execute_result["std_out"]
            metadata["output_blocks"] = PandasSdtoutHandler.extract_serialized_df_str(
                execute_result["std_out"]
            )
            if execute_result["error"]:
                metadata["error"] = True
                std_out = PandasSdtoutHandler.serializer_str_to_stdout_str(
                    execute_result["std_out"]
                )

                if "matplotlib" in std_out:
                    std_out += "\n\nMatplotlib plots are not allowed. Please use the plot generator tool instead."
                return {
                    "observation": std_out,
                    "metadata": metadata,
                }
            ret = execute_result.get("ret")
            self.update_execution_history(code)
            metadata["ret"] = ret
            str_log = PandasSdtoutHandler.serializer_str_to_stdout_str(
                execute_result["std_out"]
            )
            if isinstance(ret, pd.DataFrame):
                metadata["table"] = ret
            observation = str_log
            if ret is not None:
                if isinstance(ret, (pd.DataFrame, pd.Series)):
                    observation += ret.to_string(max_rows=10)
                else:
                    observation += str(ret)
            return {"observation": observation, "metadata": metadata}
        except Exception as e:
            metadata["error"] = True
            return {
                "observation": f"{e.__class__.__name__}: {e}",
                "metadata": metadata,
            }


def find_matching_values(
    value: str, column_name: str, df: pd.DataFrame, top_n: int = 5
) -> List[str]:
    """
    Find the closest matching values in a specified column of a DataFrame using fuzzy matching.

    This function uses the `rapidfuzz` library to perform partial ratio fuzzy matching
    between the input value and the unique values in the specified column of the DataFrame.
    It returns the top N closest matches.

    Args:
        value (str): The value to match against the column values.
        column_name (str): The name of the column to search for matching values.
        df (pd.DataFrame): The DataFrame containing the column to search.
        top_n (int): The number of top matches to return. Default is 5.

    Returns:
        List[str]: A list of the top N closest matching values from the specified column.
    """
    # Extract unique values from the specified column
    uniques = pd.Series(df[column_name].unique())

    # Compute fuzzy matching scores for each unique value
    uniques_scores = uniques.apply(
        lambda x: fuzz.partial_ratio(x.lower(), value.lower())
    )

    # Set the index of the scores to the unique values
    uniques_scores.index = uniques

    # Sort the scores in descending order and return the top N matches
    return uniques_scores.sort_values(ascending=False).head(top_n).index.to_list()


class ColumnMatchFinder:
    def __init__(self, source_python_tool: PythonTool):
        """
        Initialize the ColumnMatchFinder with a source PythonTool.

        Args:
            source_python_tool (PythonTool): The PythonTool instance to use for accessing the local environment.
        """
        self.source_python_tool = source_python_tool
        self.source_local = self.source_python_tool.func.__self__.locals

    def match_column(
        self, value: str, column_name: str, df_name: str
    ) -> Dict[str, Any]:
        """
        Match a value against a column in a specified DataFrame using fuzzy matching.

        Args:
            value (str): The value to match against the column values.
            column_name (str): The name of the column to search for matching values.
            df_name (str): The name of the DataFrame variable in the local environment.

        Returns:
            Dict[str, Any]: A dictionary containing the observation and metadata about the matching process.
        """
        if df_name not in self.source_local:
            return {
                "observation": f"{df_name} not found in the python environment",
                "metadata": {
                    "error": True,
                },
            }

        df = self.source_local[df_name]
        if not isinstance(df, pd.DataFrame):
            return {
                "observation": f"{df_name} is not a pandas dataframe",
                "metadata": {
                    "error": True,
                },
            }

        if column_name not in df.columns:
            return {
                "observation": f"{column_name} not found in the dataframe",
                "metadata": {
                    "error": True,
                },
            }

        matched_values = find_matching_values(value, column_name, df)
        return {
            "observation": str(matched_values),
            "metadata": {
                "error": False,
            },
        }


class PythonInputs(BaseModel):
    """Python inputs."""

    code: str = Field(description="code snippet to run(can be multiline)")


def get_python_lc_tool(init_code: str = None, **kwargs) -> StructuredTool:
    """
    Create a StructuredTool instance for executing Python code in a controlled environment.

    This function initializes a PythonTool with the provided initial code, and additional arguments.
    It then creates and returns a StructuredTool that can be used to execute Python commands.

    Args:
        init_code (str, optional): Initial Python code to execute upon tool creation. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the PythonTool.

    Returns:
        StructuredTool: A StructuredTool instance configured to execute Python commands.
    """
    py_tool = PythonTool(init_code=init_code, **kwargs)
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
    )
    return StructuredTool.from_function(
        py_tool.__call__,
        name="python_env",
        description=description,
        args_schema=PythonInputs,
    )


class MatchInput(BaseModel):
    value: str = Field(description="value to match")
    column_name: str = Field(description="column to match on")
    df_name: str = Field(description="dataframe which has the column")


def get_matching_values_tool(py_tool):
    description = (
        "Find matching values in a column of a dataframe. "
        "This tool uses fuzzy matching to find the closest matches to the input value."
    )
    match_finder = ColumnMatchFinder(py_tool)
    return StructuredTool.from_function(
        match_finder.match_column,
        name="find_matching_values",
        description=description,
        args_schema=MatchInput,
    )
