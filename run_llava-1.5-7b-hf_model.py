# source:
# https://huggingface.co/llava-hf/llava-1.5-7b-hf
import os
import csv
import shutil
import requests
from glob import glob
from PIL import Image
import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


# Python program main() function 
# Defining main function 
def main():
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True
            )

    processor = AutoProcessor.from_pretrained(model_id)

    directory='/images'
    files = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.jpg'))]

    os.makedirs('/RESULTS/FIRE_IMAGES/', mode=0o777, exist_ok=True)
    RESULTS=[]
    #directory='/images'
    counter=0
    #for FILE in os.listdir(directory):
    for FILE in files:
        user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
        image_file=FILE
        #image_file=os.path.join(directory, FILE)
        response=run_llava(model, processor, user_prompt, image_file)
        print(FILE)
        print(response[-1:])
        RESULTS.append(FILE + ', ' + response[-1:])
        if int(response[-1])==1:
            shutil.copy(image_file, '/RESULTS/FIRE_IMAGES/')
        #counter += 1
        #if counter > 100:
        #    break

    os.chmod("/RESULTS/FIRE_IMAGES/", 0o777)
    with open('/RESULTS/output.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(RESULTS)
      

def run_llava(model, processor,
              user_prompt='Can you explain the image in detail? Please, use at least 100 words to explain it.',
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"

    #raw_image = Image.open(requests.get(image_file, stream=True).raw)
    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    start_t = time.time()
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    end_t = time.time()
    #print(processor.decode(output[0][2:], skip_special_tokens=True))
    #print('Time elapsed: ', end_t-start_t) 
    return processor.decode(output[0][2:], skip_special_tokens=True)



# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

