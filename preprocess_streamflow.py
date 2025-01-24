import numpy as np
import pandas as pd

############################################################################################################
#428 is the station number for Mardi############################################################################################################
year = list(range(1974,1996))
total_df = pd.DataFrame()
for i in year:
    start_year = i
    i = i - 1900 #the year is in 1900 format
    # Read the text file with streamflow data into a pandas DataFrame
    file_path = f'data/streamflow/428/AQ428_{i}.TXT'  # Replace with the path to your txt file

    # Define the columns for the DataFrame
    columns = ['Day'] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Read the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=10, skipfooter=4, engine='python', quotechar='"')
    df = df.drop(columns=['Day'])

    # Create a DataFrame to store the streamflow data
    total_streamflow = []
    #iterate though each month(which is a column) and append the streamflow to the list
    for month in df.columns:
        total_streamflow += df[month].tolist()
    #remove nan values
    total_streamflow = [x for x in total_streamflow if not np.isnan(x)]

    streamflow = pd.DataFrame()
    #add the streamflow to the streamflow dataframe
    streamflow['Streamflow'] = total_streamflow
    #remove nan values
    streamflow = streamflow.dropna()
    #add all dates for a year to the dataframe
    dates = pd.date_range(start=f'1/1/{start_year}', end=f'12/31/{start_year}')
    streamflow['Date'] = dates

    total_df = pd.concat([total_df, streamflow])

#save to csv
total_df.to_csv('data/streamflow/428streamflow.csv', index=False)



############################################################################################################
#440 is the station number for Chepe############################################################################################################
year = list(range(1964,2000))
total_df = pd.DataFrame()
for i in year:
    start_year = i
    i = i - 1900 #the year is in 1900 format
    # Read the text file with streamflow data into a pandas DataFrame
    file_path = f'data/streamflow/440/AQ440_{i}.TXT'  # Replace with the path to your txt file

    # Define the columns for the DataFrame
    columns = ['Day'] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Read the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=10, skipfooter=4, engine='python', quoting=3, )
    df = df.drop(columns=['Day'])

    # Create a DataFrame to store the streamflow data
    total_streamflow = []
    #iterate though each month(which is a column) and append the streamflow to the list
    for month in df.columns:
        total_streamflow += df[month].tolist()
    #remove nan values
    total_streamflow = [x for x in total_streamflow if not np.isnan(x)]

    streamflow = pd.DataFrame()
    #add the streamflow to the streamflow dataframe
    streamflow['Streamflow'] = total_streamflow
    #remove nan values
    streamflow = streamflow.dropna()
    #add all dates for a year to the dataframe
    dates = pd.date_range(start=f'1/1/{start_year}', end=f'12/31/{start_year}')
    streamflow['Date'] = dates[:len(streamflow)]

    total_df = pd.concat([total_df, streamflow])

year = list(range(2000,2010))
for i in year:
    start_year = i
    i = i - 2000 #the year is in 1900 format
    # Read the text file with streamflow data into a pandas DataFrame
    file_path = f'data/streamflow/440/AQ440_0{i}.TXT'  # Replace with the path to your txt file

    # Define the columns for the DataFrame
    columns = ['Day'] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Read the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=10, skipfooter=4, engine='python', quoting=3, )
    df = df.drop(columns=['Day'])
    
    # Create a DataFrame to store the streamflow data
    total_streamflow = []
    #iterate though each month(which is a column) and append the streamflow to the list
    for month in df.columns:
        total_streamflow += df[month].tolist()
    #remove nan values
    total_streamflow = [x for x in total_streamflow if not np.isnan(x)]

    streamflow = pd.DataFrame()
    #add the streamflow to the streamflow dataframe
    streamflow['Streamflow'] = total_streamflow
    #remove nan values
    streamflow = streamflow.dropna()
    #add all dates for a year to the dataframe
    dates = pd.date_range(start=f'1/1/{start_year}', periods=len(streamflow), freq='D')
    streamflow['Date'] = dates

    total_df = pd.concat([total_df, streamflow])
#save to csv
total_df.to_csv('data/streamflow/440streamflow.csv', index=False)



############################################################################################################
#650 is the station number for Khimti############################################################################################################
year = list(range(1968,2007))
total_df = pd.DataFrame()
for i in year:
    start_year = i
    # i = i - 1900 #the year is in 1900 format
    # Read the text file with streamflow data into a pandas DataFrame
    file_path = f'data/streamflow/650/Q{i}.TXT'  # Replace with the path to your txt file

    # Define the columns for the DataFrame
    columns = ['Day'] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Read the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=10, skipfooter=4, engine='python', quoting=3, )
    df = df.drop(columns=['Day'])

    # Create a DataFrame to store the streamflow data
    total_streamflow = []
    #iterate though each month(which is a column) and append the streamflow to the list
    for month in df.columns:
        total_streamflow += df[month].tolist()
    #remove nan values
    total_streamflow = [x for x in total_streamflow if not np.isnan(x)]

    streamflow = pd.DataFrame()
    #add the streamflow to the streamflow dataframe
    streamflow['Streamflow'] = total_streamflow
    #remove nan values
    streamflow = streamflow.dropna()
    #add all dates for a year to the dataframe
    dates = pd.date_range(start=f'1/1/{start_year}', end=f'12/31/{start_year}')
    streamflow['Date'] = dates[:len(streamflow)]

    total_df = pd.concat([total_df, streamflow])

#save to csv
total_df.to_csv('data/streamflow/650streamflow.csv', index=False)




############################################################################################################
#728 is the station number for Khimti############################################################################################################
year = list(range(1983,2007))
total_df = pd.DataFrame()
for i in year:
    start_year = i
    # i = i - 1900 #the year is in 1900 format
    # Read the text file with streamflow data into a pandas DataFrame
    file_path = f'data/streamflow/728/AQ728_{i}.TXT'  # Replace with the path to your txt file

    # Define the columns for the DataFrame
    columns = ['Day'] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Read the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=10, skipfooter=4, engine='python', quoting=3, )
    df = df.drop(columns=['Day'])

    # Create a DataFrame to store the streamflow data
    total_streamflow = []
    #iterate though each month(which is a column) and append the streamflow to the list
    for month in df.columns:
        total_streamflow += df[month].tolist()
    #remove nan values
    total_streamflow = [x for x in total_streamflow if not np.isnan(x)]

    streamflow = pd.DataFrame()
    #add the streamflow to the streamflow dataframe
    streamflow['Streamflow'] = total_streamflow
    #remove nan values
    streamflow = streamflow.dropna()
    #add all dates for a year to the dataframe
    dates = pd.date_range(start=f'1/1/{start_year}', end=f'12/31/{start_year}')
    streamflow['Date'] = dates[:len(streamflow)]

    total_df = pd.concat([total_df, streamflow])

#save to csv
total_df.to_csv('data/streamflow/728streamflow.csv', index=False)