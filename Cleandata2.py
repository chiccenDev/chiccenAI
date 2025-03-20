import os
messages = []
count = 0

with open(r"C:\Users\Josh\Desktop\programming\AI\DeepSeek\chiccenAI\trainingdata2.txt", 'r', encoding='utf-8') as f:
    raw_messages = f.readlines()
    for raw_message in raw_messages:
        #os.system("cls")
        print("Scanning {} messsages....".format(count))
        if "@" in raw_message:
            continue
        count += 1
        messages.append(raw_message)

os.system("cls")
print("Scanned {} messages!".format(count))

with open(r"C:\Users\Josh\Desktop\programming\AI\DeepSeek\chiccenAI\trainingdata3.txt", 'w', encoding='utf-8') as f:
    for message in messages:
        f.write(message)