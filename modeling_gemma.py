import torch 
import torch.nn as nn
from typing import Optional,Tuple,Union,List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig,SiglipVisionModel





class GemmaConfig():
    def __init__(self,
        vocab_size, ## this is the size of the total vocabulary
        hidden_size, ## dimension of the embeddings for the gemma decoder
        intermediate_size, ## intermediate size of gemma MLP projection
        num_hidden_layers, ## total number of decoder blocks
        num_attention_heads, ## total number of attention heads
        num_key_value_heads, ## in multi-query attention we use less heads than query to save computer and GPU memory 
        head_dim=256, ## dimension of each head
        max_position_embeddings=8192,
        rms_norm_eps=1e-6, ## used to prevent division by 0 during normalization
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.head_dim=head_dim
        self.num_key_value_heads=num_key_value_heads
        self.rms_norm_eps=rms_norm_eps
        self.rope_theta=rope_theta
        self.attention_bias=attention_bias
        self.attention_dropout=attention_dropout
        self.pad_token_id=pad_token_id

class PaliGemmaConfig():
    
    def __init__(self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs, 
    ):

        super().__init__()
        self.ignore_index=ignore_index
        self.image_token_idex=image_token_index ## basically the index that would be assigned to <image> token, which we will replace with the image tokens from vision encoder
        self.vocab_size=vocab_size
        self.projection_dim=projection_dim ## Dimension to which we project the image embeddings to match the input token embeddings(which will pe passed to gemma decoder)
        self.hidden_size=hidden_size ## size at which decoder embeddings will work (B, seq_len, hidden_size) --> this would be fed to the transformer/gemma decoder
        self.vision_config=SiglipVisionConfig(**vision_config) ## ** will automatically convert K:V of the form K=V and pass in the constructor
        self.is_encoder_decoder=False
        self.pad_token_id=pad_token_id ## not using the padding token as of now
        self.text_config=GemmaConfig(**text_config,pad_token_id=pad_token_id)
        self.vocab_size=self.text_config.vocab_size
        self.text_config.num_image_tokens=(self.vision_config.image_size // self.vision_config.patch_size) ** 2   ## would be 256 in this case, patch size 14, image size 224*224
        self.vision_config.projection_dim=projection_dim



'''THIS IS THE ACTUAL CLASS WHICH WILL GENERATE THE OUTPUT FROM (<IMAGE><BOS><INPUT_PROMT> TOKENS)'''
class PaliGemmaForConditionalGeneration(nn.Module): 
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config=config
        self.vision_tower=SiglipVisionModel(config.vision_config)
        self.multi_modal_projector=PaliGemmaMultiModalProjector(config) ## convert image tokens to compatible dimension to input text tokens
        self.vocab_size=config.vocab_size

        language_model=GemmaForCasualLM(config.text_config) ## Gemma model which will generate the output from Prefix input tokens
        self.language_model=language_model

        self.pad_token_id=self.config.pad_token_id if self.config.pad_token_id is not None else -1

    '''This will make the use same weights for both embeddings and vocabulary ---> will help save a lot of parameters '''
    def tie_weights(self):
        return self.language_model.tie_weights() 
    
    def _merge_input_ids_with_image_features(
            self,image_features:torch.Tensor, inputs_embeds: torch.Tensor, input_ids:torch.Tensor, attention_mask:torch.Tensor, kv_cache:Optional[KVCache]=None
    ):
        _,_,embed_dim=image_features.shape
        batch_size,sequence_len=input_ids.shape ## here seq_len would be the total number of tokens in the input sequence <image><BOS><prompt><anygeneration>
        dtype,device=inputs_embeds.dtype,inputs_embeds.device
        ### scaling the image features, because we have projected them into a new dimension now --> hidden size
        ##[Batch_size,seq_len,hidden_size]
        scaled_image_features=image_features / (self.config.hidden_size ** 0.5)

        ### now combining the image and text embeddings correctly [B,]
        final_embedding=torch.zeros(batch_size,sequence_len,embed_dim,dtype=dtype,device=device)
        text_mask=(input_ids!=self.config.image_token_idex) & (input_ids!=self.pad_token_id)  ##[B,seq_len]
        image_mask=(input_ids==self.config.image_token_idex) ##[B,seq_len]. True for image tokens
        pad_mask=(input_ids == self.pad_token_id) ## (B,sequence_len)

        ### matching the final embeddings size
        text_mask_expanded=text_mask.unsqueeze(-1).expand(-1,-1,embed_dim) ## (B,seq_len,embed_dim/hidden_size]
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        ## building the final embeddings now
        ## adding the text embeddings
        final_embedding=torch.where(text_mask_expanded,inputs_embeds,final_embedding) ## whereever text mask is true, take from input_embeds, wherever false just take from final embeddings

        ## adding the image embeddings, but the issue here is image_embeddings don't match the size of final embeddings, as length of image_embeddings == no of image tokens
        ## for this we use masked_scatter, basically where image mask is true take from image embeddings, but the total dimension of image_mask should be equal to final embeddings (True in our case)
        ## Also number of true (1) in the mask should be equal to image tokens, (basically the number of value we can fill), they should match
        final_embedding=final_embedding.masked_scatter(image_mask_expanded,scaled_image_features)

        ## fill the pad tokens
        final_embedding=torch.where(pad_mask_expanded,torch.zeros_like(final_embedding),final_embedding)

        ### NOW BUILDING THE ATTENTION MASK
        dtye,device=inputs_embeds.dtype,inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len=inputs_embeds.shape[1] ## sequence length of the input tokens, (during inference this would just 1 as )

        if kv_cache is None or kv_cache.num_items()==0:
            ## is kv_cache is 0, this means we are in the prefill phase, and in paligemma , we let it all tokens <image_token><BOS><input_prompt> because technically this is the input
            ## and hence the model should be able to see this in full
            causal_mask=torch.full((batch_size,q_len,q_len),fill_value=0,dtype=dtye,device=device)

        else: 
            ## this means we are in the generation/inference phase, only one query should come as input
            assert q_len == 1
            kv_len=kv_cache.num_items()+q_len ## increasing the length of kv cache
            causal_mask=torch.full((batch_size,q_len,kv_len),fill_value=0,dtype=dtye,device=device) ## [B,q_len,kv_len]  kv_len is basically number of tokens in the kv cache, we add +q_len as this will attent to itself as well during attention

        ## adding head dimension as well
        causal_mask=causal_mask.unsqueeze(1) ##  B,q_len,kv_len] ---> [B,1,q_len,kv_len]]


        #### in a similar way will build the position ids matrix now
        if kv_cache is not None or kv_cache.num_items()>0:
            ## This means we are in the generation/inference phase 
            position_ids=torch.cumsum(attention_mask,dim=-1)[:-1] ## i just need the last position id, which is for the current query token
            if position_ids.dim()==1:
                position_ids=position_ids.unsqueeze(0)

        else: ## means we are still in the prefilling phase --> just make pad tokens position ids as 1
            position_ids=torch.cumsum(attention_mask,dim=-1).masked_fill_((attention_mask==0),1).to(device)

        return final_embedding,causal_mask,position_ids    


    def forward(
            self,
            input_ids:torch.LongTensor=None,
            pixel_values:torch.FloatTensor=None,
            attention_mask:Optional[torch.Tensor]=None,
            kv_cache:Optional[KVCache]=None,
    )->Tuple:
        
        '''
        input_ids:[B,seq_len]
        attention_mask:[B,seq_len]
        pixel_values:[1, C, H, W] , 1 as batch size is 1 during inference
        '''
        
        ## making sure there is no padding
        assert torch.all(attention_mask==1),"input cannot be padded because we are doing inference"

        ## getting the input ids correspoding to out prompt, by passing through the decoder (gemma) input embedding layers
        ## [Batch, seq_len,hidden_size]
        inputs_embeds=self.language_model.get_input_embeddings()(input_ids) 
        
        #[B,seq_len,embed_dimension ]  ---> will have to convert to hidden size (compatible with text embeddings)
        selected_image_features=self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        
        image_features=self.multi_modal_projector(selected_image_features) ## [B,seq_len,embed_dimension] --> [B,seq_len,hidden_size]

        ### now we have to replace the "<image>" tokens in the embeddings with the image features
        ## merge the embeddigs of the text tokens and image tokens
        input_embeds,attention_mask,position_ids=self._merge_input_ids_with_image_features(image_features,inputs_embeds,input_ids,attention_mask,kv_cache)

        ### now we have the correct embeddings (Image_embeddings|<BOS>|prompt) --> pass through the gemma model
        outputs=self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs

    















