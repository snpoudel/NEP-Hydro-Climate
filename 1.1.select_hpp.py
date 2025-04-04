import pandas as pd
import numpy as np

#read all hydropower projects listed in doed database 
# #http://www.doed.gov.np/license/54
df = pd.read_csv('Hydropower_Table.csv')

#filter projects with capacity less than equal to 10 MW
df = df[df['Capacity(MW)'] <= 10]
df = df.reset_index(drop=True)

#filter project that has COD(Date of Commencement) of last last 10yrs
df[['COD_month', 'COD_day', 'COD_year']] = df['COD'].str.split('/', expand = True)
df['COD_year'] = df['COD_year'].astype(np.float32)
df = df[df['COD_year'] > 2070]

#save filtered list of hydropower projects
df.to_csv('filtered_hpp.csv', index=False)