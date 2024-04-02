import scipy.io
import numpy as np
import torch.nn.functional as F
import time
import os

person_compare=1 # 이전 사람과 달라지는지 비교하기 위한 변수
per_count=0 #

count_list=[]
id_list=[]
TP_list=[]
FP_list=[]
TN_list=[]
FN_list=[]
F1_list=[]

start_time=time.time()
mat_filelist=os.listdir('/home/workspace/projects/InterpretationReID/10_row_matlab')
gal_label_mat= scipy.io.loadmat('/home/workspace/projects/InterpretationReID/gallery_label.mat')
person_color_mat=scipy.io.loadmat('/home/workspace/projects/InterpretationReID/person_color.mat')
gal_label=gal_label_mat['gallery_label']

up_color_dict={18:"black",19:"blue",20:"gray",21:"green",22:"purple",23:"red",24:"white",25:"yellow",0:"unknown"} #상의색 8개
down_color_dict={4:"black",5:"blue",6:"brown",7:"gray",8:"green",9:"pink",10:"purple",11:"white",12:"yellow",0:"unknown"} #하의색 9개

for try_num,mat_file in enumerate(mat_filelist):
    result = scipy.io.loadmat(f'/home/workspace/projects/InterpretationReID/20_row_matlab/{mat_file}')

    dist_total=result['dist_total'] #쿼리와 갤러리 간 전체 거리값 
    dist_att=result['dist_att'] #dist_fake_stack belong to 0.0~1.0, 쿼리와 갤러리 간 속성별 거리값

    person_num=int(mat_file.split('_')[0])

    if (person_num!=person_compare): #만약 비교대상 인물이 달라진다면 다음 인물 속성으로 넘어가도록 작성
        per_count+=1
        person_compare=person_num

    query_up=up_color_dict[person_color_mat['upcolor'][0][per_count]] #상의색 구하기
    query_down=down_color_dict[person_color_mat['downcolor'][0][per_count]] # 하의색 구하기

    #print(query_up,query_down)

    count=0
    count_gal=0
    gal_num=[]

    gal_label_dict={} # 신원별 이미지 리스트가 들어있음. 
    for index, value in enumerate(gal_label.tolist()[0]):
        if value in gal_label_dict:
            gal_label_dict[value].append(index)
        else:
            gal_label_dict[value]=[index]

    #confusion matrix
    TP = FP = TN = FN = 0

    #max가맞음
    gal_count=-1
    person_gal_id=0


    for index,i in enumerate(dist_att):
        gal_list=i #첫번째 갤러리에 대한 속성별 거리 

        up_color_att=gal_list[18:26] # 상의색 속성 거리 저장
        down_color_att=gal_list[4:13] # 하의색 속성 거리 저장

        up_max=max(up_color_att) #상의색 속성중 가장 큰 거리
        down_max=max(down_color_att) # 하의색 속성중 가장 큰 거리

        up_max_index=np.where(gal_list==up_max)[0] #상의 색 중 가장 큰 거리값의 인덱스
        down_max_index=np.where(gal_list==down_max)[0] #하의 색 중 가장 큰 거리값의 인덱스

        if(len(up_max_index)==1):
            gal_fake_up=up_color_dict[up_max_index[0]] #모델이 예측한 갤러리 상의색 
        else:
            idx=[index for index, value in enumerate(up_max_index) if 18 <= value <= 25][0]
            gal_fake_up=up_color_dict[up_max_index[idx]]
        if(len(down_max_index)==1):
            gal_fake_down=down_color_dict[down_max_index[0]] #모델이 예측한 갤러리 하의색
        else:
            idx=[index for index, value in enumerate(down_max_index) if 4 <= value <= 12][0]
            gal_fake_down=down_color_dict[down_max_index[idx]]
        # gal_fake_up=up_color_dict[up_max_index[-1]]
        # gal_fake_down=down_color_dict[down_max_index[-1]]

        for x in gal_label_dict: #x 최댓값 = 1501
            if(index in gal_label_dict[x]):
                if(person_gal_id!=x): #다음 인물로 넘어간다면?
                    person_gal_id=x # 다음 인물로 넘겨주고
                    gal_count+=1 # 갤러리 카운트 +1
                if(person_gal_id!=0):
                    gal_real_up=up_color_dict[person_color_mat['upcolor'][0][gal_count]]
                    gal_real_down=down_color_dict[person_color_mat['downcolor'][0][gal_count]] # [0][0] -> [0][?] 로 바꿔야함!!!!!!
                    break
                else:
                    gal_real_up='unknown'
                    gal_real_down='unknown'
                    break
        #필터링 구간 
        if(query_up=='unknown'): #일단 실험 
            query_up=gal_fake_up
        if(query_down=='unknown'):
            query_down=gal_fake_down

        if(query_up==gal_fake_up and query_down==gal_fake_down): #상, 하의 정답 비교해서 완전히 일치하는 갤러리 인덱스 확인     #필터링 안한것
            gal_num.append(count_gal) # 잔여 갤러리 목록 추가 
            count+=1#필터링된 갤러리 이미지 개수를 세기위한 카운트
            if(query_up==gal_real_up and query_down==gal_real_down): #실제 
                TN+=1
            else:
                FP+=1
        else: # 필터링 했다. 
            if(query_up==gal_real_up and query_down==gal_real_down): #
                FN+=1
            else:
                TP+=1
        count_gal+=1 #갤러리에서 몇번째 갤러리 인물인지 세기 위함
        
    count_person=0
    gal_filter_dict={} #신원별 남아있는 갤러리 수 확인을 위한 딕셔너리
    for i in gal_label_dict: #i는 인물넘버를 의미함
        for j in gal_num:
            if(j in gal_label_dict[i]):
                count_person+=1
                gal_filter_dict[i]=count_person
        count_person=0

    percision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * percision * recall / (percision + recall)
    
    end_time=time.time()
    print(f'{try_num+1}번 쿼리({mat_file}) 결과 -','F1-score:', F1_score, f'up:{query_up}, down:{query_down}, 시간경과: {round(end_time-start_time,2)}초, {round(round(end_time-start_time,2)/60,2)}분')
    print(f"신원별 남아있는 이미지 수: {gal_filter_dict}")
    print("True Positive (TP):", TP)
    print("False Positive (FP):", FP)
    print("True Negative (TN):", TN)
    print("False Negative (FN):", FN)

    count_list.append(count)
    id_list.append(len(gal_filter_dict))
    TP_list.append(TP)
    FP_list.append(FP)
    TN_list.append(TN)
    FN_list.append(FN)
    F1_list.append(F1_score)

    #Rank-1, Rank-5, Rank-10 계산
    rank1 = rank5 = rank10 = 0 #Rank-1, Rank-5, Rank-10 초기화

    #필터링된 갤러리의 총 개수
    gallery_num = len(gal_num)
    # print(q_person_num)

    #필터링된 갤러리 이미지의 거리 리스트
    filtered_gallery_dist = []
    for i in range(gallery_num):
        filtered_gallery_dist.append(dist_total[0][gal_num[i]-1].reshape(-1))
    filtered_gallery_dist = np.array(filtered_gallery_dist)
    # print(filtered_gallery_dist[0])

    #필터링된 갤러리 이미지의 거리가 작은 것 부터 인덱스를 나열
    index_list = []
    for i in range(gallery_num):
        dist_index = sorted(range(len(filtered_gallery_dist)), key=lambda i:filtered_gallery_dist[i]) #제일 짧은 거리부터 인덱스 나열
        index_list.append(gal_num[dist_index[i]]) #거리의 인덱스를 기반으로 원래 갤러리에서 해당되는 인덱스로 매칭
    index_list = np.array(index_list)
    # print(index_list)

    #인덱스를 기반으로 인물 번호, 즉 신원을 확인
    g_person_num = []
    for i in range(len(gal_num)):
        for gal_id in gal_label_dict:
            if(index_list[i] in gal_label_dict[gal_id]):
                g_person_num.append(gal_id)
    # print(len(g_person_num))

    # Rank-1, Rank-5, Rank-10 계산
    if g_person_num[0] == person_num:
        rank1 += 1
    if person_num in g_person_num[:5]:
        rank5 += 1
    if person_num in g_person_num[:10]:
        rank10 += 1

    rank_1=[]
    rank_5=[]
    rank_10=[]

    rank_1.append(rank1)
    rank_5.append(rank5)
    rank_10.append(rank10)

    # Rank-1, Rank-5, Rank-10 출력
    # rank_1 = rank1 / len(q_person_num)
    # rank_5 = rank5 / len(q_person_num)
    # rank_10 = rank10 / len(q_person_num)

    print(f"rank-1: {rank1}, rank-5: {rank5}, rank-10: {rank10}\n")

    

end_time=time.time()
elapsed_time = end_time - start_time
print(f"총 소요 시간: {elapsed_time}초")
print(f"필터링된 갤러리 이미지 개수: {sum(count_list)/len(count_list)}")
print(f"필터링된 잔여 신원 개수: {sum(id_list)/len(id_list)}") #신원 개수 구하기
print("True Positive (TP):", sum(TP_list)/len(TP_list))
print("False Positive (FP):", sum(FP_list)/len(FP_list))
print("True Negative (TN):", sum(TN_list)/len(TN_list))
print("False Negative (FN):", sum(FN_list)/len(FN_list))
print('F1-score:', sum(F1_list)/len(F1_list))
print('Rank@1:%f, Rank@5:%f, Rank@10:%f' %(sum(rank_1)/len(rank_1), sum(rank_5)/len(rank_5), sum(rank_10)/len(rank_10)))