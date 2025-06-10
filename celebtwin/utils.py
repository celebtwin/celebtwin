import sys
import traceback

import pdb
from decorator import decorator


@decorator
def post_mortem(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(exc_tb)
        raise
