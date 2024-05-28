import os
from PIL import Image

def reduce_image_resolution_multiple_for_folder(input_folder, output_base_folder):
    # 입력 폴더의 모든 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 각 이미지 파일에 대해 해상도를 낮춘 여러 버전 생성
    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        img = Image.open(input_image_path)
        original_width, original_height = img.size

        # 10% ~ 90%까지 해상도를 낮춘 이미지 저장
        for scale_factor in range(10, 100, 10):
            new_width = int(original_width * (scale_factor / 100))
            new_height = int(original_height * (scale_factor / 100))
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # 저장할 폴더 경로 생성
            scale_folder = os.path.join(output_base_folder, f"query_{scale_factor}")
            if not os.path.exists(scale_folder):
                os.makedirs(scale_folder)

            output_image_path = os.path.join(scale_folder, image_file)
            resized_img.save(output_image_path)
            print(f"Image saved to {output_image_path}")

input_folder = '/root/amd/label_manager/dataset/Market-1501-v24.05.21_junk_false/query'
output_base_folder = '/root/amd/label_manager/dataset/Market-1501-v24.05.21_junk_false/resolution'

# 함수 호출 (실제 경로 사용 필요)
reduce_image_resolution_multiple_for_folder(input_folder, output_base_folder)
