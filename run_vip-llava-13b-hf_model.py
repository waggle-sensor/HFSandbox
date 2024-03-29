# source
# https://huggingface.co/llava-hf/vip-llava-13b-hf
import os
import csv
import shutil
import requests
from random import shuffle
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
    shuffle(files)

    os.makedirs('/RESULTS/FIRE_IMAGES/', mode=0o777, exist_ok=True)
    #RESULTS=[]
    #BAG_OF_WORDS=[]
    #NEUTRAL_BAG_OF_WORDS=[]
    #CHECKPOINT=[]
    CHECKPOINT, RESULTS, BAG_OF_WORDS, NEUTRAL_BAG_OF_WORDS = load_checkpoint()
    #checkpoint_path='/RESULTS/checkpoint.txt'
    #if os.path.isfile(checkpoint_path):
    #    with open(checkpoint_path) as file:
    #        CHECKPOINT = [line.rstrip() for line in file]

    counter=0
    for FILE in files:
        if FILE not in CHECKPOINT:
            user_question="Is there smoke or not in the image?"
            #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
            user_prompt='Please generate the number 1 if there is smoke in the image, otherwise generate the number 0. Only one number has to be generated in your response.'

            image_file=FILE
            #image_file=os.path.join(directory, FILE)
            response, time=run_llava(model, processor, user_question, user_prompt, image_file)
            print(FILE)
            #print(response)
            pattern='###Assistant:'
            print(response[-1:])
            RESULTS.append(FILE + ', ' + response[-1:] + ', ' + str(time))
            if int(response[-1])==1: # FIRE!!!
                #if not os.path.isfile(image_file):
                try:
                    #print('Trying to copy '+image_file+' into '+'/RESULTS/FIRE_IMAGES/')
                    shutil.copy(image_file, '/RESULTS/FIRE_IMAGES/')
                except IOError:
                    print("Unable to copy file: ", image_file)

                user_question="An expert inspected the image and claimed to see a wildfire in it."
                #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
                user_prompt='Please, generate a detailed explanation of why the expert thinks this way.'
                response, time=run_llava(model, processor, user_question, user_prompt, image_file)
                response=response.split(pattern, 1)[1]
                BAG_OF_WORDS.append(image_file)
                BAG_OF_WORDS.append(response)
                BAG_OF_WORDS.append(str(time))
                print('--------------->>>>>>>>>>>>>>>>>>')
                print(response)
                #break
                
                user_question="As a WildfireWatcher, your task is to scrutinize optical camera images for wildfire indicators."
                #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
                #user_prompt='Please, generate a detailed explanation of why the expert thinks this way.'
                user_prompt='Issue a one-word alert based on your findings: WILDFIRE if smoke plumes, dispersed smoke, or visible fire are detected; NA for absence of fire or smoke.'
                response, time=run_llava(model, processor, user_question, user_prompt, image_file)
                response=response.split(pattern, 1)[1]
                BAG_OF_WORDS.append(response)
                BAG_OF_WORDS.append(str(time))
                print('--------------->>>>>>>>>>>>>>>>>>')
                print(response)
                #break

                user_question="Is there fire in the image?"
                #user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                #user_prompt='Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response.'
                #user_prompt='Please, generate a detailed explanation of why the expert thinks this way.'
                user_prompt='Please, provide a detailled explanation of what you see in the image.'
                response, time=run_llava(model, processor, user_question, user_prompt, image_file)
                response=response.split(pattern, 1)[1]
                NEUTRAL_BAG_OF_WORDS.append(image_file)
                NEUTRAL_BAG_OF_WORDS.append(response)
                NEUTRAL_BAG_OF_WORDS.append(str(time))
                print('--------------->>>>>>>>>>>>>>>>>>')
                print(response)
                #break


            CHECKPOINT.append(FILE)
            counter += 1
            if counter%10 == 0:
                save_results(RESULTS=RESULTS, BAG_OF_WORDS=BAG_OF_WORDS, NEUTRAL_BAG_OF_WORDS=NEUTRAL_BAG_OF_WORDS, CHECKPOINT=CHECKPOINT)

    save_results(RESULTS=RESULTS, BAG_OF_WORDS=BAG_OF_WORDS, NEUTRAL_BAG_OF_WORDS=NEUTRAL_BAG_OF_WORDS, CHECKPOINT=CHECKPOINT)





def load_checkpoint(checkpoint_path='/RESULTS/checkpoint.txt',
                    output_path='/RESULTS/output.csv',
                    bag_of_words_path='/RESULTS/bag_of_words.csv',
                    neutral_bag_of_words_path='/RESULTS/neutral_bag_of_words.csv'):
    RESULTS=[]
    BAG_OF_WORDS=[]
    NEUTRAL_BAG_OF_WORDS=[]
    CHECKPOINT=[]
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as file:
            CHECKPOINT = [line.rstrip() for line in file]

    if os.path.isfile(output_path):
        with open(output_path) as file:
            RESULTS = [line.rstrip() for line in file]

    if os.path.isfile(bag_of_words_path):
        with open(bag_of_words_path) as file:
            BAG_OF_WORDS = [line.rstrip() for line in file]

    if os.path.isfile(neutral_bag_of_words_path):
        with open(neutral_bag_of_words_path) as file:
            NEUTRAL_BAG_OF_WORDS = [line.rstrip() for line in file]

    return CHECKPOINT, RESULTS, BAG_OF_WORDS, NEUTRAL_BAG_OF_WORDS





def save_results(RESULTS, BAG_OF_WORDS, NEUTRAL_BAG_OF_WORDS, CHECKPOINT,
                 checkpoint_path='/RESULTS/checkpoint.txt',
                 output_path='/RESULTS/output.csv',
                 bag_of_words_path='/RESULTS/bag_of_words.csv',
                 neutral_bag_of_words_path='/RESULTS/neutral_bag_of_words.csv'):
    with open(checkpoint_path, 'w') as f:
        for file_line in CHECKPOINT:
            f.write(f"{file_line}\n")

    os.chmod(checkpoint_path, 0o666)

    os.chmod("/RESULTS/FIRE_IMAGES/", 0o777)
    with open(output_path,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(RESULTS)
      
    os.chmod(output_path, 0o666)

    with open(bag_of_words_path,'w') as bow_file:
        wr = csv.writer(bow_file, dialect='excel')
        wr.writerow(BAG_OF_WORDS)

    os.chmod(bag_of_words_path, 0o666)

    with open(neutral_bag_of_words_path,'w') as bow_file:
        wr = csv.writer(bow_file, dialect='excel')
        wr.writerow(NEUTRAL_BAG_OF_WORDS)

    os.chmod(neutral_bag_of_words_path, 0o666)




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
    return processor.decode(output[0][2:], skip_special_tokens=True), end_t-start_t 



# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

