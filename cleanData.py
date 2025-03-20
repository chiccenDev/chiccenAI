import json
import os
import time

data_head = r"C:\Users\Josh\Desktop\Discord etc\Data\chiccen 2024 December\messages"

def CollateFiles(head):
    message_files = []
    for root, _, files in os.walk(head):
        for file in files:
            if ("messages.json" not in file):
                continue
            else:
                message_files.append(os.path.join(root, file))
    return message_files

def CleanJsons(list):
    file_count = 0
    message_count = 0
    responses = ""
    for file in list:
        os.system("cls")
        print("DEBUG: reading from file: {}".format(file))
        #time.sleep(1)
        file_count += 1
        try:
            with open(file, "r", encoding="utf-8") as file:
                data = json.load(file)
                for message in data:
                    print("Scanning {} file(s) for {} message(s)".format(file_count, message_count))
                    if (message["Attachments"] != "") or (message["Contents"] == ""):
                        continue
                    message_count += 1
                    responses += message["Contents"] + "\n"
        except Exception as e:
            print(f"Error reading {file}: {e}")
            time.sleep(1)
            continue
    return responses

time_start = time.time()
output_path = r"C:\Users\Josh\Desktop\programming\AI\DeepSeek\chiccenAI\trainingdata.txt"
if (os.path.exists(output_path)):
    os.remove(output_path)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(CleanJsons(CollateFiles(data_head)))
time_end = time.time()
runtime = time_end - time_start

print("Finished in {} minutes and {} seconds!".format(runtime // 60, runtime % 60))
time.sleep(5)