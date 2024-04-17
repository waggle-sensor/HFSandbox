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
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        output_hidden_states=True
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
    CHECKPOINT, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP = load_checkpoint()
    #checkpoint_path='/RESULTS/checkpoint.txt'
    #if os.path.isfile(checkpoint_path):
    #    with open(checkpoint_path) as file:
    #        CHECKPOINT = [line.rstrip() for line in file]

    #RESULTS=RESULTS[0]
    #RESULTS='File Name, RGB+IR, RGB, IR \n'
    #RESULTS.append('File Name, RGB+IR, RGB, IR \n')
    counter=len(CHECKPOINT)
    for FILE in files:
        if FILE not in CHECKPOINT:
            #user_question="As a WildfireWatcher, your task is to scrutinize images for wildfire indicators."
            #user_prompt='Issue a one-word alert based on your findings: WILDFIRE if smoke plumes, dispersed smoke, or visible fire are detected; NA for absence of fire or smoke.'

            user_question="Is there fire or not in the image?"
            user_prompt="Please generate the number 1 if there is fire in the image, otherwise generate the number 0. Only one number has to be generated in your response."

            #user_question="Issue a one-word alert based on your findings: WILDFIRE if smoke plumes, dispersed smoke, or visible fire are detected; NA for absence of fire or smoke."
            #user_question="Describe this image in detail."
            #user_prompt="These are optical camera image and the pseudo-color thermal IR image showing relative temperatures of the scene, side by side. As a WildfireWatcher, your task is to scrutinize images for wildfire indicators."



            image_file=FILE
            #image_file=os.path.join(directory, FILE)
            pattern='###Assistant:'
            internal_rep, time=run_llava(model, processor, user_prompt, image_file)
            #user_prompt="This is an optical camera image. As a WildfireWatcher, your task is to scrutinize images for wildfire indicators."
            rgb_internal_rep, rgb_time=rgb_run_llava(model, processor, user_prompt, image_file)
            #user_prompt="This is a pseudo-color thermal IR image showing relative temperatures of the scene. As a WildfireWatcher, your task is to scrutinize images for wildfire indicators."
            ir_internal_rep, ir_time=ir_run_llava(model, processor, user_prompt, image_file)

            #response=response.split(pattern, 1)[1]
            #rgb_response=rgb_response.split(pattern, 1)[1]
            #ir_response=ir_response.split(pattern, 1)[1]
            print(FILE)
            print(counter)
            #print(response)
            #print('--------------->>>>>>>>>>>>>>>>>>')
            #print(response)
            #RESULTS+=FILE + ',' + response + ',' + rgb_response + ',' + ir_response + '\n'
            #RESULTS.append(FILE + ',' + response + ',' + rgb_response + ',' + ir_response + '\n')
            #RESULTS.append('RGB-IR' + '\n')
            #RESULTS.append(response + ', ' + str(time) + '\n')
            #RESULTS.append('RGB' + '\n')
            #RESULTS.append(rgb_response + ', ' + str(rgb_time) + '\n')
            #RESULTS.append('IR' + '\n')
            #RESULTS.append(ir_response + ', ' + str(ir_time) + '\n')


            CHECKPOINT.append(FILE)
            INTERNAL_REP.append(internal_rep.to('cpu').detach())
            RGB_INTERNAL_REP.append(rgb_internal_rep.to('cpu').detach())
            IR_INTERNAL_REP.append(ir_internal_rep.to('cpu').detach())

            counter += 1
            if counter%10 == 0:
                save_results(CHECKPOINT=CHECKPOINT, INTERNAL_REP=INTERNAL_REP, RGB_INTERNAL_REP=RGB_INTERNAL_REP, IR_INTERNAL_REP=IR_INTERNAL_REP)
        else:
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('File is in checkpoint!!!!!!!!!!')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')
            print('-------------------->>>>>>>>>>>>>>>>>>>>')

    save_results(CHECKPOINT=CHECKPOINT, INTERNAL_REP=INTERNAL_REP, RGB_INTERNAL_REP=RGB_INTERNAL_REP, IR_INTERNAL_REP=IR_INTERNAL_REP)





def load_checkpoint(checkpoint_path='/RESULTS/checkpoint.txt',
                    internal_rep_path='/RESULTS/internal_rep.pt',
                    rgb_internal_rep_path='/RESULTS/rgb_internal_rep.pt',
                    ir_internal_rep_path='/RESULTS/ir_internal_rep.pt'):
    CHECKPOINT=[]
    INTERNAL_REP=[]
    RGB_INTERNAL_REP=[]
    IR_INTERNAL_REP=[]
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as file:
            CHECKPOINT = [line.rstrip() for line in file]

    if os.path.isfile(internal_rep_path):
        aux=torch.load(internal_rep_path)
        for element in aux:
            INTERNAL_REP.append(element.unsqueeze(0))

    if os.path.isfile(rgb_internal_rep_path):
        aux=torch.load(rgb_internal_rep_path)
        for element in aux:
            RGB_INTERNAL_REP.append(element.unsqueeze(0))

    if os.path.isfile(ir_internal_rep_path):
        aux=torch.load(ir_internal_rep_path)
        for element in aux:
            IR_INTERNAL_REP.append(element.unsqueeze(0))


    return CHECKPOINT, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP





def save_results(CHECKPOINT, INTERNAL_REP, RGB_INTERNAL_REP, IR_INTERNAL_REP,
                 checkpoint_path='/RESULTS/checkpoint.txt',
                 internal_rep_path='/RESULTS/internal_rep.pt',
                 rgb_internal_rep_path='/RESULTS/rgb_internal_rep.pt',
                 ir_internal_rep_path='/RESULTS/ir_internal_rep.pt'):

    with open(checkpoint_path, 'w') as f:
        for file_line in CHECKPOINT:
            f.write(f"{file_line}\n")

    os.chmod(checkpoint_path, 0o666)

    #os.chmod("/RESULTS/FIRE_IMAGES/", 0o777)

    with torch.no_grad():
        if len(INTERNAL_REP) > 0:
            #print('len(INTERNAL_REP): ', len(INTERNAL_REP))
            #print('INTERNAL_REP[0].shape: ', INTERNAL_REP[0].shape)
            #print('len(INTERNAL_REP[0].shape): ', len(INTERNAL_REP[0].shape))
            #if len(INTERNAL_REP[0].shape) == 1:
            aux = torch.Tensor(len(INTERNAL_REP), INTERNAL_REP[0].shape[0], INTERNAL_REP[0].shape[1]).to(INTERNAL_REP[0].device)
            torch.cat(INTERNAL_REP, out=aux)
            torch.save(aux, internal_rep_path)
            os.chmod(internal_rep_path, 0o666)

        if len(RGB_INTERNAL_REP) > 0:
            aux = torch.Tensor(len(RGB_INTERNAL_REP), RGB_INTERNAL_REP[0].shape[0], RGB_INTERNAL_REP[0].shape[1]).to(RGB_INTERNAL_REP[0].device)
            torch.cat(RGB_INTERNAL_REP, out=aux)
            torch.save(aux, rgb_internal_rep_path)
            os.chmod(rgb_internal_rep_path, 0o666)

        if len(IR_INTERNAL_REP) > 0:
            aux = torch.Tensor(len(IR_INTERNAL_REP), IR_INTERNAL_REP[0].shape[0], IR_INTERNAL_REP[0].shape[1]).to(IR_INTERNAL_REP[0].device)
            torch.cat(IR_INTERNAL_REP, out=aux)
            torch.save(aux, ir_internal_rep_path)
            os.chmod(ir_internal_rep_path, 0o666)



def run_llava(model, processor,
              user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
              image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"

    raw_image = Image.open(image_file)
    raw_image = raw_image.resize((int(raw_image.size[0]/8),int(raw_image.size[1]/8)))
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    with torch.no_grad():
        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

        return output.hidden_states[-1].mean(1), end_t-start_t 




def rgb_run_llava(model, processor,
                  user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                  image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"

    im = Image.open(image_file)
    im = im.resize((int(im.size[0]/8),int(im.size[1]/8)))
    width, height = im.size
    left = 0
    top = 0
    right = width / 2
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    with torch.no_grad():
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

    return output.hidden_states[-1].mean(1), end_t-start_t 






def ir_run_llava(model, processor,
                 user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                 image_file="http://images.cocodataset.org/val2017/000000039769.jpg"):

    prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"

    im = Image.open(image_file)
    im = im.resize((int(im.size[0]/8),int(im.size[1]/8)))
    width, height = im.size
    left = width / 2
    top = 0
    right = width
    bottom = height
    raw_image = im.crop((left, top, right, bottom))

    with torch.no_grad():
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        start_t = time.time()
        output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        end_t = time.time()

    return output.hidden_states[-1].mean(1), end_t-start_t 





# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 





















# #source
# # https://huggingface.co/llava-hf/vip-llava-13b-hf
# import os
# import csv
# import shutil
# import requests
# from glob import glob
# from PIL import Image
# import time
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration, VipLlavaForConditionalGeneration


# def main():
   # image_file="http://images.cocodataset.org/val2017/000000039769.jpg"
   # user_question='What are these?'
   # user_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives detailed, and polite answers to the human's questions."
   # prompt = "USER: <image>\n" + user_prompt + "\nASSISTANT:"
   # raw_image = Image.open(requests.get(image_file, stream=True).raw)

   # model_id = "llava-hf/llava-1.5-7b-hf"

   # model = LlavaForConditionalGeneration.from_pretrained(
       # model_id, 
       # torch_dtype=torch.float16, 
       # low_cpu_mem_usage=True,
       # load_in_4bit=True,
       # output_hidden_states=True
   # )
   # model.zero_grad()

   # processor = AutoProcessor.from_pretrained(model_id)
   # inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

# #    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# #    print(output.shape)
   # output = model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
   # print(type(output))
   # print(output.keys())
   # print(len(output.hidden_states))
   # for hidden_state in output.hidden_states:
       # #print(hidden_state.shape)
       # print(hidden_state.mean(1).shape)
   # #print(type(output['image_hidden_states']))
   # ##output = model.generate(**inputs, max_new_tokens=200, do_sample=False)





# ## Using the special variable  
# ## __name__ 
# if __name__=="__main__": 
   # main() 

