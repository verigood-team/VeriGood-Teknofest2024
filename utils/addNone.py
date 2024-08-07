import os

folder_path = r'C:\Users\furkan\Masaüstü\Projects\Python\Project1'

for filename in os.listdir(folder_path):
    if filename.endswith('Hepsiburada.txt'):  
        file_path = os.path.join(folder_path, filename)
        output_file_path = os.path.join(folder_path, f'{filename}.modified')

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        modified_lines = [line.replace("Hepsiburada", "Hepsi Burada Express") for line in lines]

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(modified_lines)

        print(f"{filename} dosyası işlendi. İşlenmiş dosya: {output_file_path}")

print("Tüm dosyalar işlendi.")