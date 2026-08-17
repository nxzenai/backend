from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from ..exceptions import EDAInvalidRequest, EDAUnsupportedFile


def load_dataframe(
    path: Path, extension: str, *, nrows: int | None = None
) -> pd.DataFrame:
    try:
        if extension == ".csv":
            frame = pd.read_csv(path, nrows=nrows)
        elif extension == ".xlsx":
            frame = pd.read_excel(path, engine="openpyxl", nrows=nrows)
        elif extension == ".xls":
            frame = pd.read_excel(path, engine="xlrd", nrows=nrows)
        else:
            raise EDAUnsupportedFile()
    except EmptyDataError as exc:
        raise EDAInvalidRequest("The uploaded file is empty.") from exc
    except (ParserError, UnicodeDecodeError, ValueError, OSError, ImportError) as exc:
        raise EDAInvalidRequest(
            "The file could not be read as a valid tabular document. Check its format and encoding."
        ) from exc

    if len(frame.columns) == 0:
        raise EDAInvalidRequest("The file does not contain a header row.")
    if frame.empty:
        raise EDAInvalidRequest("The file contains headers but no data rows.")
    return frame
