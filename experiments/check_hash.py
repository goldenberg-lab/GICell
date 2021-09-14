import pandas as pd
from funs_support import hash_hp

lr, p, batch = 0.001, 64, 6

df_slice = pd.DataFrame({'lr':lr, 'p':p, 'batch':batch},index=[0])
try:
    code_hash1 = hash_hp(df_slice, method='hash_array')
except:
    code_hash1 = None
try:
    code_hash2 = hash_hp(df_slice, method='hash_pandas_object')
except:
    code_hash2 = None

print('hash1: %s\nhash2: %s' % (code_hash1, code_hash2))
