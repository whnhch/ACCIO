import pandas as pd
import json
import os

def get_pivot(data: pd.DataFrame, param: dict)->pd.DataFrame:
    return data.pivot_table(index=param['index'], columns=param['column'], values=param['value'], aggfunc=param['aggfunc'])

def data_to_text(data: pd.DataFrame, nrows:int=10, ncols:int=10, add_sep:bool=True)->str:
    
    if data.shape[0] > ncols:
        data = data.iloc[:, :ncols]
    if data.shape[0] > nrows:
        data = data.iloc[:nrows, :]
        
    linear_data=[]
    special_token=''
    if add_sep: special_token= " [SEP] " 

    for column in data.columns:
        column_values = ' '.join([str(value) for value in data[column]])
        linear_data.append(f"{column} {column_values} {special_token}")

    linear_data = ''.join(linear_data)
    return linear_data

def pivot_to_text(pivot: pd.DataFrame, nrows:int=10, ncols:int=10, add_sep:bool=True)->str:
    if pivot.shape[0] > ncols:
        pivot = pivot.iloc[:, :ncols]
    if pivot.shape[0] > nrows:
        pivot = pivot.iloc[:nrows, :]

    special_token=''
    if add_sep: special_token= " [SEP] " 
    
    linear_pivot = []
    for column in pivot.columns:
        column_values = ' '.join([str(value) for value in pivot[column]])
        linear_pivot.append(f"{column} {column_values} {special_token}")

    linear_pivot = ''.join(linear_pivot)
    return linear_pivot

def get_data_pivot_pair(data: pd.DataFrame, param: dict, nrows:int=10, ncols:int=10, add_sep:bool=True)->tuple:
    pivot=get_pivot(data, param)
    
    linear_data=data_to_text(data, nrows, ncols, add_sep)
    linear_pivot=pivot_to_text(pivot, nrows, ncols, add_sep)
    
    return linear_data, linear_pivot

def data_preprocessing(path:str, folder_names:list):
    result={'data':[],'pivot':[]}
    for folder_path in folder_names:
        data_csv_path = os.path.join(path+folder_path, 'data.csv')
        param_json_path = os.path.join(path+folder_path, 'param.json')
        try:
            df = pd.read_csv(data_csv_path, index_col=None)
            # the first columns are empty somehow
            df = df.iloc[:, 1:]
            with open(param_json_path, 'r') as json_file:
                param_data = json.load(json_file)
            
            data, pivot = get_data_pivot_pair(df, param_data, nrows=10, ncols=10)
            result['data'].append(data)
            result['pivot'].append(pivot)
        except:
            continue
    
    with open('utils.dataset.json', 'w') as json_file:
        json.dump(result, json_file)
        
if __name__ == "__main__":
    folder_names = [folder for folder in os.listdir('../data/pivot_data_csv') if os.path.isdir(os.path.join('../data/pivot_data_csv', folder))]
    data_preprocessing('../data/pivot_data_csv/',folder_names) 