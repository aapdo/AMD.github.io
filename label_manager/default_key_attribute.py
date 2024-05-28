import pandas as pd
import mat4py
from decimal import Decimal, getcontext



def generate_attribute_dict(dir_path: str, dataset: str):

    mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
    mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int).drop(columns='age').drop(columns='image_index') -1

    print( (mat_attribute_train.sum()/len(mat_attribute_train)) * 100)

    mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
    mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int).drop(columns='age').drop(columns='image_index') - 1

    print(mat_attribute_test.sum()/len(mat_attribute_train))

    mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)

    print(mat_attribute.sum()/len(mat_attribute))

    key_attribute = list(mat_attribute.keys())
    if 'age' in key_attribute:
        key_attribute.remove('age')

    return key_attribute


key_attribute = generate_attribute_dict('/root/amd/label_manager/dataset/Market-1501-v15.09.15/market_attribute.mat', "market_attribute")

# 기존 연구에서 사용하는 키 ATTRIBUTE의 순서
print(key_attribute)
target_csv_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/train_attribute.csv'
# Setting the precision to 50 decimal places
getcontext().prec = 50

# Reading the CSV file into a DataFrame
df = pd.read_csv(target_csv_path, index_col=0).drop(columns='age')

# Subtracting 1 from each value in the dataframe
df_adjusted = df - 1

# Calculating the percentage with high precision
percentage_values_precise = (df_adjusted.sum() / len(df_adjusted)) * 100

# Converting to Decimal for higher precision
percentage_values_precise_decimal = percentage_values_precise.apply(lambda x: Decimal(x))
percentage_values_precise_rounded = percentage_values_precise_decimal.apply(lambda x: round(x, 16))
result_df = pd.DataFrame(percentage_values_precise_rounded, index=key_attribute, columns=['Percentage'])

print(result_df)


result_df.to_csv('/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/attribute_percent.csv')

# Displaying the high precision percentage values
# print(percentage_values_precise_decimal)