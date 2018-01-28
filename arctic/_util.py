from warnings import warn

from pandas import DataFrame
from pandas.util.testing import assert_frame_equal
from pymongo.errors import OperationFailure
import logging

logger = logging.getLogger(__name__)

def as_sorted(dframe):
    """ Checks that dframe is monotonically increasing and sorts it if not

    This method should be preferred over dframe.as_sorted as it dramatically outperforms it
      for the case that dframe is monotonically  decreasing

    Args:
      dframe: DataFrame

    Returns:
      sorted_drame: DataFrame

    Raises:
      warning: if dframe is not monotonically increasing
    """
    if dframe.index.is_monotonic_increasing:
        return dframe
    else:
        warn("DataFrame is not monotonic increasing and must be sorted - may incur performance penalty")
        if dframe.index.is_monotonic_decreasing:
            dframe.index = dframe.index[::-1]
        else:
            dframe = dframe.sort_index()
        return dframe


def indent(s, num_spaces):
    s = s.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    return s


def are_equals(o1, o2, **kwargs):
    try:
        if isinstance(o1, DataFrame):
            assert_frame_equal(o1, o2, kwargs)
            return True
        return o1 == o2
    except Exception:
        return False


def enable_sharding(arctic, library_name, hashed=True, key='symbol'):
    """
    Enable sharding on a library

    Parameters:
    -----------
    arctic: `arctic.Arctic` Arctic class

    library_name: `basestring` library name

    hashed: `bool` if True, use hashed sharding, if False, use range sharding
            See https://docs.mongodb.com/manual/core/hashed-sharding/,
            as well as https://docs.mongodb.com/manual/core/ranged-sharding/ for details.

    key: `basestring` key to be used for sharding. Defaults to 'symbol', applicable to
         all of Arctic's built-in stores except for BSONStore, which typically uses '_id'.
         See https://docs.mongodb.com/manual/core/sharding-shard-key/ for details.
    """
    c = arctic._conn
    lib = arctic[library_name]._arctic_lib
    dbname = lib._db.name
    library_name = lib.get_top_level_collection().name
    try:
        c.admin.command('enablesharding', dbname)
    except OperationFailure as e:
        if 'already enabled' not in str(e):
            raise
    if not hashed:
        logger.info("Range sharding '" + key + "' on: " + dbname + '.' + library_name)
        c.admin.command('shardCollection', dbname + '.' + library_name, key={key: 1})
    else:
        logger.info("Hash sharding '" + key + "' on: " + dbname + '.' + library_name)
        c.admin.command('shardCollection', dbname + '.' + library_name, key={key: 'hashed'})
