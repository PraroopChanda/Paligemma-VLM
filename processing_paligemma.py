import numpy as np
from PIL import Image
import torch 
from typing import Dict, List, Optional, Union, Tuple, Iterable

'''' This File will basically be used to pre-process the image and prompt
what will happen is the images would be normalized , resized. and an tokenized input prompt would
be returned which would contain the placeholder for image tokens (256--> tokens for image so 256 placeholders),
This would then be concatinated with the input(text) prompt and returned
'''

IMAGENET_STANDARD_MEAN=[0.5,0.5,0.5]
IMAGENET_STANDARD_STD=[0.5,0.5,0.5]

''' building how to prepend image tokens and input text prompt'''
def add_image_tokens_to_prompt(prefix_prompt,bos_token,image_seq_len,image_token):


    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n" ## /n is basically a separator token 


def resize(
        image:Image,
        size:Tuple[int,int],
        resample: Image.Resampling=None,
        reducing_gap:Optional[int]=None,
)->np.ndarray:
    height,width=size
    resized_image=image.resize(                                  ### This is basically PIL resizing
        (width,height),resample=resample,reducing_gap=reducing_gap
    )

    return resized_image

def rescale(
        image:np.ndarray,scale:float,dtype:np.ndarray=np.float32
) ->np.ndarray:
    rescaled_image=image*scale
    rescaled_image=rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image:np.ndarray,
        mean:Union[float,Iterable[float]],
        std:Union[float,Iterable[float]],
)->np.ndarray:
    mean=np.array(mean,dtype=image.dtype)
    std=np.array(std,dtype=image.dtype)
    image= (image-mean)/std
    return mean


def process_images(
    images:List[Image.Image],
    size:Dict[str,int] = None,  ## dictionary where keys are strings and values are integers
    resample: Image.Resampling = None,
    rescale_factor: float=None,
    image_mean: Optional[Union[float,List[float]]]=None, ## Optional[Union[float, List[float]]]= (float OR list of floats) OR None,, Union[float, List[float]]= either float OR list of floats
    image_std: Optional[Union[float,List[float]]]=None,
) -> List[np.ndarray]:

    height,width=size[0],size[1]
    images=[resize(image=image,size=(height,width),resample=resample) for image in images]

    ## convert each image to numpy array (from PIL Image to numpy array)
    images=[np.array[image] for image in images]

    ## rescale values to be in the rane from [0,1] (done by torchvision by doing transforms.ToTensor())
    images=[rescale(image,scale=rescale_factor) for image in images]

    ## normalize the image to have 0 mean and 1 standard deviation
    images=[normalize(image,mean=image_mean,std=image_std) for image in images]

    ## move channel to the first dimension, the visionmodel would expect images in [Channel, height, width] ( H,W,C --> C,W,H (2,0) then --> C,H,W (0,1))
    images=[image.transpose(2,0,1) for image in images]
    return images




class PaliGemmaProcessor:
    IMAGE_TOKEN="<image>" ## this would be a class attribute instead of instance attribute

    def __init__(self,tokenizer,num_image_tokens:int,image_size:int): ## here tokenizer is the tokenizer of the gemma model
        super().__init__()

        self.image_seq_length=num_image_tokens ## this is basically the number of tokens output by the vision encoder
        self.image_size=image_size ## 224*224 in this case

        ''' now add the extra special tokens'''
        tokens_to_add={"additional_special_tokens":[self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS=[
            f"<loc{i:04d}>" for i in range(1024)
        ] ## These tokens would be used for object detection (bounding boxes)

        EXTRA_TOKENS+=[
            f"<seg{i:03d}>" for i in range(128)
        ] ## these would be the tokens used for object segmentation 

        ''' adding extra tokens for object detection and segmentation'''
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id=tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) ## get the token id for the special image token, this is basically a placeholder for image tokens from the vision encoder

        ## will add BOS and EOS ourselves
        tokenizer.add_bos_token=False
        tokenizer.add_eos_token=False

        self.tokenizer=tokenizer


    def __call__(self,
                 text:List[str],
                 images:List[Image.Image],
                 padding: str="longest",
                 truncation: bool=True,  
    )-> dict:
        assert len(images)==1 and len(text)==1, f"Recevied {len(images)} images for {len(text)} promts"
        ### as of now we are just processing/inferencing one image and one prompt, although code is written taking a list of images

        '''' Basically we want to process the images given as input, resize them to 224*224, normalize them as well, finally convert to a tensor to be process by the vision model'''
        pixel_values=process_images(
            images,
            size=(self.image_size,self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        ## convert the list of numpy arrays to a single numpy array with shape [Batch size, channel, height, width], because there is just one image it would be [1,C,H,W]
        pixel_values=np.stack(pixel_values,axis=0) 

        ## converting numpy array to a tensor 
        pixel_values=torch.tensor(pixel_values)   ## THIS IS THE PIXEL VALUES WHICH WILL GO INTO THE VISION ENCODER (1, C, H, W) --> because during inference we are taking batch size as 1

        '''
        we want to now input the image tokens with the text (question) and pass it to the gemma decoder
        but before that we need to merge the vision tokens and the text question tokens(obtained by passing through the tokenizer)
        for this we prepend, 'image_tokens'*self.image_seq_length number of image tokens to the prompt,
        and during training/inference, these tokens would be filled with actual image tokens
        '''

        input_strings=[
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text    ### we just directly add the prompt without any tokenization as of now example prompt -->"<image><image><image> ... <BOS>Hello how are you doing?? "
    
        ]

        ### return the input_ids (the input id's of the correspoding text) and attention_mask as pytorch tensors
        ### example --> "Hello world " --> [5, 2, 9] --> torch.tensor[[....] [...] [....]] (converted by the embedding layer(1024 dimension))
        ### output would be token ids and attention masks
        inputs=self.tokenizer(
            input_strings,
            return_tensor="pt", ## this basically means to return as pytorch tensors
            padding=padding,
            truncation=truncation
        )
        ### generally we are not using any padding, so the mask would be all True , inputs as output would be input_ids (after tokenization) --> this would basically be a list [25600,25600, ..., 23,1,4,5]
        ##, attention_mask , would be the one used in attention mechanism

        return_data={"pixel_values":pixel_values,**inputs}  ## **inputs basically just unpacks the key value pairs, masking it form key=value instead of key:value
        
        return return_data
