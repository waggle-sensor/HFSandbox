# source
# https://huggingface.co/llava-hf/vip-llava-13b-hf
import os
import csv
import shutil
import requests
from glob import glob
from PIL import Image
import time
import torch
from transformers import AutoProcessor, VipLlavaForConditionalGeneration


def main():
    model_id = "llava-hf/vip-llava-13b-hf"

    model = VipLlavaForConditionalGeneration.from_pretrained(
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
    BAG_OF_WORDS=[]
    #directory='/images'
    counter=0
    #for FILE in os.listdir(directory):
    for FILE in files:
        user_question="Is there smoke or not in the image?"
        #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
        user_prompt='Please generate the number 1 if there is smoke in the image, otherwise generate the number 0. Only one number has to be generated in your response.'

        image_file=FILE
        #image_file=os.path.join(directory, FILE)
        response=run_llava(model, processor, user_question, user_prompt, image_file)
        print(FILE)
        #print(response)
        pattern='###Assistant:'
        print(response[-1:])
        RESULTS.append(FILE + ', ' + response[-1:])
        if int(response[-1])==1: # FIRE!!!
            if not os.path.isfile(image_file):
                shutil.copy(image_file, '/RESULTS/FIRE_IMAGES/')

            user_question="An expert inspected the image and claimed to see a wildfire in it."
            #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
            user_prompt='Please, generate a detailed explanation of why the expert thinks this way.'
            response=run_llava(model, processor, user_question, user_prompt, image_file).split(pattern, 1)[1]
            BAG_OF_WORDS.append(image_file)
            BAG_OF_WORDS.append(response)
            print('--------------->>>>>>>>>>>>>>>>>>')
            print(response)
            #break
            

        counter += 1
        if counter > 1000:
            break

    os.chmod("/RESULTS/FIRE_IMAGES/", 0o777)
    with open('/RESULTS/output.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(RESULTS)
      

    with open('/RESULTS/bag_of_words.csv','w') as bow_file:
        wr = csv.writer(bow_file, dialect='excel')
        wr.writerow(BAG_OF_WORDS)






def run_llava(model, processor,
              user_question='What are these?',
              user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    #prompt = f"{user_prompt}.###Human: <image>\n{user_question}###Assistant:"
    prompt = f"{user_question}.###Human: <image>\n{user_prompt}###Assistant:"

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

