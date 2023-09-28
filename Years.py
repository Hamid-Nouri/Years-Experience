#!/usr/bin/env python
##-------------------
from langdetect import detect
from googletrans import Translator
from bs4 import BeautifulSoup
from datetime import datetime
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm 
from word2number import w2n
import pandas as pd
import pickle
import numpy as np
import requests
import re
import os
import glob
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

##-----------------------
start_time = datetime.now()  
def read_csv(file_name):
    df = pd.read_csv(file_name, sep=',')  
    return df

if __name__ == '__main__':
    my_file_name = 'Jobs.csv' ##--- write the name of you file here
    original_data = [f'{my_file_name}']  

    cpus = mp.cpu_count()
    with mp.Pool(processes=cpus) as pool:
        result = pool.map(read_csv, original_data)  
    pool.close()
    pool.join()

    data = pd.concat(result)
data.rename(columns={'search_country': 'country'}, inplace=True)
data['id'] = range(0, len(data))
data = data[['id'] + [col for col in data if col != 'id']]
end_time = datetime.now()  
print('Total time for reading file(s) taken: {}'.format(end_time - start_time))
##----------
start_time = datetime.now()
data['search_time'] = pd.to_datetime(data['search_time'])
data['date'] = data['search_time'].dt.date
data['time'] = data['search_time'].dt.time
##--- Reading the training set
with open('train_data.pkl', 'rb') as file:
    train_data = pickle.load(file)
duplicates_count = train_data.duplicated().value_counts()
train_data = train_data.drop_duplicates() 
###----- Split the DataFrame into equal-sized as the training set
if len(data) > len(train_data) :
    num_splits = len(data) // len(train_data)
    additional_splits = len(data) % len(train_data)
    sub_dataframes = []
    for i in range(num_splits):
        start_idx = i * len(train_data)
        end_idx = (i + 1) * len(train_data)
        sub_data = data.iloc[start_idx:end_idx]
        sub_dataframes.append(sub_data)
    if additional_splits > 0:
            start_idx = (len(train_data) * (num_splits))-1
            end_idx = len(data)
            sub_data = data.iloc[start_idx:end_idx]
            sub_dataframes.append(sub_data)
if len(data) <= len(train_data) :
    data.to_csv(original_data[0])
    len_dif = abs(len(data)-len(train_data))
    len_not_mentioned = train_data[['job_title_clean','level']].groupby('level').count().iloc[1, 0]
    if len_not_mentioned > len_dif : 
        data.to_csv(original_data[0], index=False)
        rows_to_remove = train_data[train_data['level'] == 'Not Mentioned'].sample(n=len_dif, random_state=42)
        train_data_filtered = train_data[~train_data.index.isin(rows_to_remove.index)]
if len(data) > len(train_data) :
    output_directory = "./"
    for index, sub_data in enumerate(sub_dataframes):
        filename = f"sub_dataframe_{index + 1}.csv"
        full_path = f"{output_directory}/{filename}"
        sub_data.to_csv(full_path, index=False)  # Set index=False to exclude row numbers
        print(f"Saved {filename} to {output_directory}")
    end_time = datetime.now()  # Record the end time
    print('Total time for Time spliting taken: {}'.format(end_time - start_time))
###-----------------
directory = './'
if len(data) <= len(train_data) :
    file_list = [original_data[0]]
if len(data) > len(train_data) :
    all_files = os.listdir(directory)
    file_list = [filename for filename in all_files if filename.startswith('sub_dataframe_') and filename.endswith('.csv')]
for filename in sorted(file_list):
    print(filename)
    sub_data = pd.read_csv(filename, sep=',')
    start_time_total = datetime.now()
    start_time = datetime.now()
    with open('train_data.pkl', 'rb') as file:
        train_data = pickle.load(file)
    if len(data) < len(train_data) :
        len_dif = abs(len(data)-len(train_data))
        len_not_mentioned = train_data[['job_title_clean','level']].groupby('level').count().iloc[1, 0]   
        if len_not_mentioned > len_dif :
            len_dif = abs(len(data)-len(train_data))
            rows_to_remove = train_data[train_data['level'] == 'Not Mentioned'].sample(n=len_dif, random_state=42)
            train_data_filtered = train_data[~train_data.index.isin(rows_to_remove.index)]
    all_job_title =train_data['job_title_clean'].unique().tolist()
    senior_job_titles = ['Senior Data Analyst','Senior Data Engineer','Senior Data Scientist']
    set1 = set(all_job_title)
    set2 = set(senior_job_titles)
    Non_senior_job_titles = list(set1 - set2)
    train_data = train_data[['job_title_clean', 'trsl', 'level']].drop_duplicates()
    ##---
    sub_data['date'] = pd.to_datetime(sub_data['date'])
    min_date = sub_data['date'].min()
    max_date = sub_data['date'].max()
    date_difference = max_date - min_date
    print("Minimum Date:", min_date)
    print("Maximum Date:", max_date)
    print("Date Difference:", date_difference)
    sub_data['date'] = pd.to_datetime(sub_data['date'])
    columns_to_check = ['job_description', 'job_title_clean', 'company_name', 'country']
    sub_data['date_diff'] = (sub_data.groupby(columns_to_check)['date']
                              .transform('max') - sub_data['date']).dt.days
    print("Number of Total job:", len(sub_data))
    ###------- information about duplicates
    # duplicates_info = []
    # for index, row in sub_data.iterrows():
    #     duplicate_group = sub_data[
    #         (sub_data['job_description'] == row['job_description']) &
    #         (sub_data['job_title_clean'] == row['job_title_clean']) &
    #         (sub_data['company_name'] == row['company_name']) &
    #         (sub_data['country'] == row['country'])
    #     ]
    #     if len(duplicate_group) > 1:
    #         duplicates_info.append({
    #             'original_row': row,
    #             'duplicate_rows': duplicate_group.to_dict(orient='records'),
    #         })
    # for idx, duplicate_info in enumerate(duplicates_info):
    #     print(f"Duplicate Group {idx + 1}:")
    #     print("Original Row:")
    #     print(duplicate_info['original_row']['id'])
    #     print("Duplicate Rows:")
    #     for duplicate_row in duplicate_info['duplicate_rows']:
    #         print(duplicate_row['id'])
    #     print('====================================')
    filtered_data = sub_data[sub_data['date_diff'] <= 90]
    filtered_data = filtered_data.drop_duplicates(subset=columns_to_check, keep='first')
    filtered_data = filtered_data.reset_index(drop=True)
    num_duplicates = len(sub_data) - len(filtered_data)
    num_non_duplicates = len(filtered_data)
    print("Number of Duplicates Removed:", num_duplicates)
    print("Number of Non-Duplicates Remaining:", num_non_duplicates)
    print("Number of Missing:",  len(sub_data) - (num_duplicates + num_non_duplicates))
    rows_to_remove = train_data[train_data['level'] == 'Not Mentioned'].sample(n=num_duplicates, random_state=42)
    train_data_filtered = train_data[~train_data.index.isin(rows_to_remove.index)]
    train_data = train_data_filtered.reset_index(drop=True)
    sub_data = filtered_data[['id','job_title_clean','company_name','country','job_description','date_diff']]
    end_time = datetime.now()  # Record the end time
    print('Total time for Time cleaning taken: {}'.format(end_time - start_time))
    ##------ Detecting and trsnlating
    start_time = datetime.now()  # Record the start time
    def language_detection(text):
        try:
            src_lang = detect(text)
            return src_lang
        except:
            return "Unknown"
    if __name__ == '__main__':
        cpus = mp.cpu_count()
        with mp.Pool(processes=cpus) as pool:
            result = pool.map(language_detection, sub_data['job_description'].tolist())
        pool.close()
        pool.join()
        sub_data['language'] = result
    def apply_translation(row):
        if row['language'] != 'en':
            translated_text = translate_large_text(row['job_description'], row['language'], 'en')
            return translated_text
        else:
            return row['job_description']
    def translate_large_text(text, src_lang, dest_lang):
        try:
            translator = Translator()
            translated_text_parts = []
            chunk_size = 300  # You can adjust this value based on your needs

            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                translated_chunk = translator.translate(chunk, src=src_lang, dest=dest_lang).text
                translated_text_parts.append(translated_chunk)

            return ' '.join(translated_text_parts)
        except:
            return "Cannot be translated"
    if __name__ == '__main__':
        dataset = [row for _, row in sub_data.iterrows()]
        cpus = mp.cpu_count()
        with mp.Pool(processes=cpus) as pool:
            result = pool.map(apply_translation, dataset)
        pool.close()
        pool.join()
        sub_data['trsl'] = result
    count = sub_data['trsl'].str.count('Cannot be translated').sum()
    not_trsl = sub_data[sub_data['trsl'].str.contains('Cannot be translated')]
    print(f"Number of entries with 'Cannot be translated': {count}")
    sub_data = sub_data[~sub_data['trsl'].str.contains('Cannot be translated', case=False, na=False)]
    rows_to_remove = train_data[train_data['level'] == 'Not Mentioned'].sample(n=count, random_state=42)
    train_data_filtered = train_data[~train_data.index.isin(rows_to_remove.index)]
    train_data = train_data_filtered.reset_index(drop=True)
    if train_data.shape[0] == sub_data.shape[0]:
        print('data set and training set have the same length, you can calculate the accuracy.')
    if train_data.shape[0] != sub_data.shape[0]:
        print('data set and training set don"'"t have the same length, you can"'"t calculate the accuracy.')
    end_time = datetime.now()  # Record the end time
    print('Total time for Detecting language of job descriptions and Transating taken: {}'.format(end_time - start_time))
    ##----------- Getting level and years
    start_time = datetime.now()  
    tfidf_vectorizer = TfidfVectorizer(max_features=100000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['trsl'])
    y_train = train_data['level']
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    unlabeled_data = sub_data[['id','job_title_clean', 'company_name', 'country', 'trsl','date_diff']]
    X_unlabeled_tfidf = tfidf_vectorizer.transform(unlabeled_data['trsl'])
    predicted_labels = clf.predict(X_unlabeled_tfidf)
    unlabeled_data['predicted_level'] = predicted_labels
    if len(train_data) == len(predicted_labels):
        ground_truth_labels = train_data['level']
        if ground_truth_labels is not None:
            accuracy = accuracy_score(ground_truth_labels, predicted_labels) * 100
            print(f'Accuracy of of predicting the level: {accuracy:.2f}%')
        else:
            print('Ground truth labels are not available for unlabeled data.')
    ##----------------
    mask = (unlabeled_data['predicted_level'] == 'Junior') & (unlabeled_data['job_title_clean'].isin(senior_job_titles))
    unlabeled_data.loc[mask, 'predicted_level'] = 'Senior'
    mask = (unlabeled_data['predicted_level'] == 'Not Mentioned') & (unlabeled_data['job_title_clean'].isin(senior_job_titles))
    unlabeled_data.loc[mask, 'predicted_level'] = 'Senior'
    Final_Senior_job_titles = unlabeled_data[(unlabeled_data['predicted_level'] == 'Senior')
                  &(unlabeled_data['job_title_clean'].isin(senior_job_titles))]
    Final_Senior_Non_job_titles = unlabeled_data[(unlabeled_data['predicted_level'] == 'Senior')
                  &(unlabeled_data['job_title_clean'].isin(Non_senior_job_titles))]
    Final_Senior = pd.concat([Final_Senior_job_titles, Final_Senior_Non_job_titles])
    # Define the list of words to search for
    words_to_search = ['to senior', 'to the senior', 'with other Senior', 'Senior 40',
        'with... senior', 'with a senior', 'our Senior', 'with senior', 'become a great Senior',
        'awareness on senior management level','update senior', 'and senior' ]
    condition = Final_Senior['trsl'].str.contains('|'.join(words_to_search), case=False)
    Final_Senior.loc[condition, 'predicted_level'] = 'Not Mentioned'
    def process_row(row):
        patterns = [r'(.{0,25}\b(year).{0,5}\b)', r'(.{0,25}\b(years).{0,25}\b)']
        job_description = row['trsl']
        extracted_text = []
        lines = job_description.split('\n') 
        for line in lines:
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    extracted_text.append(match.group())
        if len(extracted_text) != 0 :
            filter_criteria = [
             'yearly','vest','during', 'duration', 'per year', 'people', 'day','year old', 'years old','upon years',
            'over the','just over','next','next couple', 'shy of', 'in under', 'customers', 'celebrating', 'in year',
            'out of','of the last','completion of','secondment','insurance','on','within','new year',
            'under','residency','turn','the past','predicted','contract','terms','will','the next',
            'lasts','b2b','growth','enhanced','increasing','long','initial','awards','award','range',
            'fixed-term','plan','our','record checks','clearance','grown','started up','celebrated',
            'in years', 'after', 'each year', 'consecutive', 'month', 'every', 'post', 'a year',
            'for up to','of those','years in','bank','global','enrolled','workplaces','served',
            'full time','temporary','sc','check','eea','1st','2nd','3rd','4th','5th','calendar','csp','weeks','week',
            'deliver', 'remote', 'length','this year','fixed term','almost','€','anniversary','lasting',
            'period','learning','ago','company','including','alternation','alternating','by','preparing',
            'children','engagement','we','employer','labeling','built','tf','countries','00','limited',
            'the first','the year','time','startup','. over','assignment','appointment','employment',
            'work at','grow','development','less than','left','role','program','this is a','possibility',
            'left','combines','team','salary','abroad','might','thriving','for more than','average','storage',
            '.com','.ms',' /', ' . ','000', '$', '£','%', '...']

            cleaned_text_list = [item for item in extracted_text if all(criteria not in item.lower() for criteria in filter_criteria)]
            ##------ Extracting numbers
            new_patterns = [r'-?\d+\.*\d*',r'Experience:\d+']
            numbers = []
            for pattern in new_patterns:
                for text in cleaned_text_list:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if pattern == r'Experience:\d+':
                            experience_value = int(match.split(":")[1])
                            numbers.append(experience_value)
                        else:
                            try:
                                number = float(match)
                                if abs(int(number)) <= 10:
                                    numbers.append(abs(int(number)))
                            except ValueError:
                                pass

            sorted_numbers = sorted(numbers)
            if len(sorted_numbers) == 0:
                return 'Not Mentioned'
            else:
                return min(sorted_numbers)
        else:
            return 'Not Mentioned'

    final_years_column = []
    for record in Final_Senior.to_dict(orient='records'):
        result = process_row(record)
        final_years_column.append(result)
    Final_Senior['years'] = final_years_column
    # ####------------------------------
    Final_Junior = unlabeled_data[(unlabeled_data['predicted_level'] == 'Junior')]
    final_years_column = []
    for record in Final_Junior.to_dict(orient='records'):
        result = process_row(record)
        final_years_column.append(result)

    Final_Junior['years'] = final_years_column
    # ##------------
    Final_Not_Mentioned = unlabeled_data[(unlabeled_data['predicted_level'] == 'Not Mentioned')
                  &(unlabeled_data['job_title_clean'].isin(Non_senior_job_titles))]
    final_years_column = []
    for record in Final_Not_Mentioned.to_dict(orient='records'):
        result = process_row(record)
        final_years_column.append(result)
    Final_Not_Mentioned['years'] = final_years_column
    ###-------------------------------------------
    Final_level_data = pd.concat([Final_Senior, Final_Junior,Final_Not_Mentioned])
    count_not_mentioned = len(Final_level_data[Final_level_data['years'] == 'Not Mentioned'])
    count_mentioned = len(Final_level_data[Final_level_data['years'] != 'Not Mentioned'])
    summation = count_not_mentioned + count_mentioned
    print(f"{count_not_mentioned} jobs will be removed from data because the 'years' in the job description are not mentioned.")
    print(f"Your data now has {count_mentioned} jobs where the 'years' are mentioned.")
    print(f"Missing : {len(Final_level_data) - summation}")
    Final_data = Final_level_data[Final_level_data['years'] != 'Not Mentioned'][['id','trsl','job_title_clean','company_name','country','predicted_level','years']]
    Final_data.rename(columns={'predicted_level': 'level'}, inplace=True)
    ##---------
    predicted_senior_data = Final_data[Final_data['level'] == 'Senior']  # Filter for 'Senior' level
    for index, row in predicted_senior_data.iterrows():
        job_description = row['trsl']
        if 'senior' not in job_description.lower() and 'junior' not in job_description.lower():
            Final_data.at[index, 'level'] = 'Not Mentioned'
    ##---------
    for index, row in Final_data[Final_data['level'] == 'Not Mentioned'].iterrows():
        job_description = row['trsl']
        if 'a senior' in job_description.lower():
            Final_data.at[index, 'level'] = 'Senior'
        if 'a junior' in job_description.lower():
            Final_data.at[index, 'level'] = 'Junior'
    total_rows = len(Final_data)
    duplicates_count = Final_data[['trsl', 'job_title_clean', 'company_name']].duplicated().value_counts()
    non_identical_count = total_rows - duplicates_count.get(True, 0)
    print(f"Identical Rows: {duplicates_count.get(True, 0)}")
    print(f"Non-Identical Rows (Unique): {non_identical_count}")
    check_data = Final_data
    Final_data = Final_data[['id','job_title_clean','company_name','country','level','years']]
    job_titles_to_check = ['Senior Data Analyst', 'Senior Data Engineer', 'Senior Data Scientist']
    condition = Final_data['job_title_clean'].isin(job_titles_to_check)
    Final_data.loc[condition, 'level'] = 'Senior'                         
    end_time = datetime.now()  # Record the end time
    print('Total time for Detecting level and year taken: {}'.format(end_time - start_time))
    ##------- Saving all the data
    if len(file_list) == 1 :
        output_filename1 = "Level.csv"
    else : 
        output_filename1 = filename.replace("sub_dataframe", "Level")
        output_filename1 = output_filename1.replace(' ', '_')

    Final_data.to_csv(output_filename1, index_label='id', encoding='utf-8-sig')
    file_size = os.path.getsize(output_filename1)
    file_size_bytes = os.path.getsize(output_filename1)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print("Your data saved as '{}' with Size of : {:.2f} MB at {}".format(output_filename1,file_size_mb,datetime.now()))
    end_time_total = datetime.now()
    print('Total time taken: {}'.format(end_time_total - start_time_total))
    print('\n********************************\n')

csv_files = glob.glob(f"Level*.csv")
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
# Get the current date and format it as yyyy-mm-dd
current_date = datetime.now().strftime("%Y-%m-%d")
combined_data = pd.concat(dfs, ignore_index=True)

filename = f'Final-Level-{current_date}.csv'
combined_data.to_csv(filename, index=False, encoding='utf-8-sig')

print(f'Combined data saved as "{filename}"')
file_size_bytes = os.path.getsize(filename)
file_size_mb = file_size_bytes / (1024 * 1024)
print(f'Size of "{filename}": {file_size_mb:.2f} MB')
##-------
def remove_csv_files(directory, naming_pattern):
    all_files = os.listdir(directory)
    files_to_remove = [filename for filename in all_files if filename.startswith(naming_pattern) and filename.endswith('.csv')]
    if not files_to_remove:
        print("No files matching the pattern to remove.")
        return
    for filename in files_to_remove:
        filepath = os.path.join(directory, filename)
        print(f"Removing: {filepath}")
        os.remove(filepath)
directory = './' 
naming_pattern = 'sub_dataframe_'
remove_csv_files(directory, naming_pattern)
naming_pattern = 'Level'
remove_csv_files(directory, naming_pattern)



import os
import shutil

# Define the target filename
target_filename = "Years_Level.csv"

# List all files in the directory
files_in_directory = os.listdir()

# Filter and keep only the files you want to remove
files_to_remove = [filename for filename in files_in_directory if filename.startswith("Final*")]

# Check if the target filename already exists
if os.path.exists(target_filename):
    print(f"'{target_filename}' already exists. Skipping merge and deletion.")
else:
    # Merge the existing files into the target file
    combined_data = pd.concat([pd.read_csv(file) for file in files_to_remove], ignore_index=True)
    combined_data.to_csv(target_filename, index=False, encoding='utf-8-sig')
    print(f"'{target_filename}' created successfully.")

    # Remove the existing files
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"'{file}' removed successfully.")
        else:
            print(f"'{file}' does not exist and was not removed.")

