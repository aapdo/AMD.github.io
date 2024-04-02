#필터링된 갤러리와 10%해상도의 신원별 쿼리 사용하여 train_net 돌리게 해주는 파이썬 코드

import subprocess
import os 
import shutil
import csv

project_path = "/home/workspace"
# 실행하려는 파이썬 스크립트 파일명을 지정합니다.
a=os.listdir(f"{project_path}/datasets/query")

row_res_folder = f"{project_path}/datasets/query" #쿼리 복사 해올곳
query_folder = f"{project_path}/datasets/Market1501/query_copy" #쿼리 복사 해둘곳 

filter_folder  = f"{project_path}/datasets/gallery"
gal_folder = f"{project_path}/datasets/Market1501/bounding_box_test" #갤러리 복사 해둘곳 


for i in a:
    csv_data=[]
    if(int(i)>=1170):
        try:
            #쿼리 복사
            if os.path.exists(query_folder):
                shutil.rmtree(query_folder)  # 대상 폴더를 삭제합니다.
            shutil.copytree(f"{row_res_folder}/{i}", query_folder)  # 원본을 대상 폴더로 복사합니다.

            #갤러리 복사
            if os.path.exists(gal_folder):
                shutil.rmtree(gal_folder)  # 대상 폴더를 삭제합니다.
            shutil.copytree(f"{filter_folder}/Filter_{i}/total", gal_folder)  # 원본을 대상 폴더로 복사합니다.

            #인물번호 및 이미지 수 저장
            f=open('/home/workspace/output/output_test.csv','a',encoding='utf-8',newline='')
            wr=csv.writer(f)
            csv_data.append(f"{i}_{len(os.listdir(query_folder))}")
            wr.writerow(csv_data)
            f.close()

            #train_net.py 실행 
            script_file = "/home/workspace/projects/InterpretationReID/train_net.py"
            result = subprocess.run(["python", script_file])


        except Exception as e:
            print(f"An error occurred: {str(e)}")

# for i in a: # 쿼리폴더별 하나씩 
#     query_row=f"{a}/{i}" #낮은 해상도 쿼리 경로
#     query="/home/workspace/datasets/Market1501/query" #쿼리의 폴더로 
#     shutil.copytree(query_row,query)


