from utils import *

sql = """
SELECT *
FROM AH_USE_LOG_BYMINUTE_LABELED_copy
"""


df = get_table_from_db(sql)



# 3min 20s 걸림