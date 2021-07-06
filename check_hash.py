import pandas as pd

lr, p, batch = 0.001, 64, 6

df_slice = pd.DataFrame({'lr':lr, 'p':p, 'batch':batch},index=[0])
df_slice = df_slice.loc[0].reset_index().rename(columns={'index':'hp',0:'val'})
hp_string = pd.Series([df_slice.apply(lambda x: x[0] + '=' + str(x[1]), 1).str.cat(sep='_')])
try:
    code_hash1 = pd.util.hash_array(hp_string)[0]
except:
    code_hash1 = None
try:
    code_hash2 = pd.util.hash_pandas_object(hp_string)[0]
except:
    code_hash2 = None

print('hash1: %s\nhash2: %s' % (code_hash1, code_hash2))
