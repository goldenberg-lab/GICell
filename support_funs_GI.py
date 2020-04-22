import sys, os
import pandas as pd
from zipfile import ZipFile
import shutil

def stopifnot(cond):
    if not cond:
        sys.exit('error!')

# Function to parse the zipped file
def zip_points_parse(fn, dir, valid_cells):
    valid_files = ['Points ' + str(k + 1) + '.txt' for k in range(6)]
    with ZipFile(file=fn, mode='r') as zf:
        names = pd.Series(zf.namelist())
        stopifnot(names.isin(valid_files).all())
        zf.extractall('tmp')
    # Loop through and parse files
    holder = []
    for pp in names:
        s_pp = pd.read_csv(os.path.join(dir, 'tmp', pp), sep='\t', header=None)
        stopifnot(s_pp.loc[0, 0] == 'Name')
        cell_pp = s_pp.loc[0, 1].lower()
        stopifnot(cell_pp in valid_cells)
        df_pp = pd.DataFrame(s_pp.loc[3:].values.astype(float), columns=['x', 'y'])
        stopifnot(df_pp.shape[0] == int(s_pp.loc[2, 1]))  # number of coords lines up
        df_pp.insert(0, 'cell', cell_pp)
        holder.append(df_pp)
    df = pd.concat(holder).reset_index(drop=True)
    shutil.rmtree('tmp', ignore_errors=True)  # Get rid of temporary folder
    return df
