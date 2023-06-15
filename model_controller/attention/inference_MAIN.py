import os
import torch
import numpy as np

from functools import partial
from tqdm import tqdm
from itertools import chain
from datetime import datetime
from .inference_batch import get_batch_information_general
from .inference_eval_toolkit import top_k_logits
from .inference_language_toolkit import tokenize_string, tokenize_list

import pandas as pd
from .inference_utils import get_device
from .inference_language_toolkit import START_INDEX, END_INDEX, PAD_INDEX, pretty_decode
from .inference_batch import pad_recipe_info, load_recipe_tensors, get_ingr_mask, pad_ingredients, pad_name
from .inference_baseline_model import create_model
from .inference_visualization import get_batch_generated_recipes


# Filters
MAX_NAME = 15
MAX_INGR = 5
MAX_INGR_TOK = 20
MAX_STEP_TOK = 256

def eval_model(input, device, model, logit_modifier_fxn, token_sampler,
               max_len, max_name_len=15, ingr_map=None, **tensor_kwargs):

    recipe_reprs_list = []

    # Iterate through batches in the epoch
    model.eval()
    with torch.no_grad():

        '''
        Model Forward Step
        '''
        # Generates probabilities
        (log_probs, output_tokens, ingr_attns), \
        (name_log_probs, name_output_tokens) = model.forward(
            device=device, 
            inputs=(
                input.calorie.to(device),
                input.name_tk.to(device),
                input.ingredients_tk.to(device)
            ),
            ingr_masks=input.ingredient_mask.to(device),
            max_len=max_len-1,
            start_token=START_INDEX, 
            logit_modifier_fxn=logit_modifier_fxn, 
            token_sampler=token_sampler,
            visualize=True, 
            max_name_len=max_name_len-1
        )

        del log_probs, name_log_probs

        batch_generated = output_tokens.cpu().data.numpy().tolist()
        # names_gen_batch = name_output_tokens.cpu().data.numpy().tolist()
        for i, generated in enumerate(batch_generated):
            # Clean up generated text once it hits END_INDEX
            generated = generated[:generated.index(END_INDEX)] if END_INDEX in generated else generated
            
            # name_tokens = names_gen_batch[i] if names_gen_batch is not None else None
            name_tokens = None
            full_recipe_str = ''
            if name_tokens is not None:
                name_str = pretty_decode(name_tokens)
                full_recipe_str += '\nGENERATED NAME: `{}`\n'.format(name_str)
            
            output_str = None
            if generated is not None:
                output_str = pretty_decode(generated)
                full_recipe_str += output_str
                # print(full_recipe_str)
            
            # recipe_reprs_list.append(full_recipe_str)


        # print('DECODED RECIPE:\n{}\n'.format(recipe_reprs_list[0]))
        
            
    return full_recipe_str

# =======================================================================================================
# =======================================================================================================
# =======================================================================================================



def load_data(dataset_folder):
    """
    Load full data (including recipe information, user information)

    Arguments:
        dataset_folder {str} -- Location of data

    Keyword Arguments:
        base_splits {list} -- Train/Validation/Test file base strs

    Returns:
        pd.DataFrame -- Items per user DataFrame
        pd.DataFrame -- DataFrame containing recipe information
    """
    start = datetime.now()

    # Recipes
    df_r = pd.read_pickle(os.path.join(dataset_folder, 'recipes.pkl'))
    # df_r = pd.read_pickle(os.path.join(dataset_folder, 'PP_recipes.pkl'))
    
    print('{} - Loaded {:,} recipes ({:,.3f} MB total memory)'.format(
        datetime.now() - start, len(df_r), df_r.memory_usage(deep=True).sum() / 1024 / 1024
    ))

    # Ingredient map
    df_ingr = pd.read_pickle(os.path.join(dataset_folder, 'ingr_map.pkl'))
    ingr_ids, ingr_names = zip(*df_ingr.groupby(['id'], as_index=False)['replaced'].first().values)
    ingr_map = dict(zip(ingr_ids, ingr_names))
    ingr_map[max(ingr_ids) + 1] = ''
    print('{} - Loaded map for {:,} unique ingredients'.format(
        datetime.now() - start, len(ingr_map)
    ))

    return df_r, ingr_map

def pad_name_string(name_string, max_name_tokens=15):
    padded_tokens = name_string + [PAD_INDEX] * (max_name_tokens - len(name_string))
    return padded_tokens

def pad_ingredients_string(name_ingredients, max_ingredients=5, max_ingr_tokens=20):
    new_tokens = [
        i[:max_ingr_tokens] + [PAD_INDEX]*(max_ingr_tokens - len(i[:max_ingr_tokens])) for i in name_ingredients[:max_ingredients]
    ]
    # Pad with empty ingredients
    new_tokens += [[PAD_INDEX]*max_ingr_tokens] * (max_ingredients - len(name_ingredients[:max_ingredients]))
    return new_tokens

def _flatten(list_of_lists):
    flattened_list = []
    for x in list_of_lists:
        if isinstance(x, list):
            flattened_list.extend(_flatten(x))
        else:
            flattened_list.append(x)
    return flattened_list

class input_data():
    def __init__(self, name, ingredients, calorie):
        

        self.name_tk = tokenize_string(name)
        self.name_tk = pad_name_string(self.name_tk)

        self.ingredients_tk = tokenize_list(ingredients, flatten=False)
        self.ingredients_tk  = pad_ingredients_string(self.ingredients_tk)
        self.ingredients_tk = _flatten(self.ingredients_tk)
        
        self.ingredient_mask = torch.unsqueeze(torch.LongTensor(get_ingr_mask(self.ingredients_tk, max_ingredients=MAX_INGR)), 0)
        self.name_tk = torch.unsqueeze(torch.LongTensor(self.name_tk), 0)
        self.ingredients_tk = torch.unsqueeze(torch.LongTensor(self.ingredients_tk), 0)
        self.calorie = torch.unsqueeze(torch.tensor(calorie, dtype=torch.long),0)

def initiate_model():
    start = datetime.now()

    USE_CUDA, DEVICE = get_device()
    data_dir = 'saved_models/attention_model/data'         # location of the data corpus
    vocab_emb_dim = 50        # size of word embeddings
    calorie_emb_dim = 50      # size of calorie embeddings
    ingr_emb_dim = 10         # size of ingr embeddings
    hidden_size = 256         # number of hidden units per layer
    n_layers = 1              # number of layers
    batch_size = 1            # the name says it all
    model_path = 'saved_models/attention_model/model_weights/model_baseline-v1CANDIDATE.pt' # model weights
    # save_dir = '/Users/Jia Yu/Documents/GitHub/Attention-Recipe-Personalization/OUTPUT_FOLDER' # save model output
    overwrite = False         # Overwrite existing outputs
    ingr_gru = False          # Use BiGRU for ingredient encoding
    decode_name = False       # Multi-task learn to decode name along with recipe
    ingr_emb = False          # Use Ingr embedding in encoder
    shared_proj = False       # Share projection layers for name and steps
    ppx_only = False          # Only calculate perplexity (on full test set)
    n_samples = 1e9           # sample test items

    '''
    Load data
    '''
    # Get the DFs
    df_r, ingr_map = load_data(data_dir)
    n_items = len(df_r)
    print('{} - Data loaded.'.format(datetime.now() - start))

    # Pad recipe information
    N_INGREDIENTS = 0

    # masks are added here
    df_r = pad_recipe_info(
        df_r, max_name_tokens=MAX_NAME, max_ingredients=MAX_INGR,
        max_ingr_tokens=MAX_INGR_TOK, max_step_tokens=MAX_STEP_TOK
    )

    '''
    Create model
    '''

    model = create_model(
        vocab_emb_dim=vocab_emb_dim, 
        calorie_emb_dim=calorie_emb_dim,
        hidden_size=hidden_size, 
        n_layers=n_layers, 
        dropout=0.0,
        max_ingr=MAX_INGR, 
        max_ingr_tok=MAX_INGR_TOK, 
        use_cuda=USE_CUDA, 
        state_dict_path=model_path,
        ingr_gru=ingr_gru, 
        decode_name=decode_name,
        ingr_emb=ingr_emb, 
        num_ingr=N_INGREDIENTS, 
        ingr_emb_dim=ingr_emb_dim,
        shared_projection=shared_proj,
    )

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    tensors_to_load = [
        ('name_tensor', 'name_tokens'),
        ('calorie_level_tensor', 'calorie_level'),
        ('technique_tensor', 'techniques'),
        ('ingr_tensor', 'ingredient_tokens'),
        ('steps_tensor', 'steps_tokens'),
        ('ingr_mask_tensor', 'ingredient_mask'),
        ('tech_mask_tensor', 'techniques_mask'),
    ]
    tensor_names, tensor_cols = zip(*tensors_to_load)

    # Load tensors into memory
    memory_tensors = load_recipe_tensors(
        df_r, DEVICE, cols=tensor_cols, types=[torch.LongTensor] * len(tensors_to_load)
    )
    memory_tensor_map = dict(zip(tensor_names, memory_tensors))
    print('{} - Tensors loaded in memory.'.format(datetime.now() - start))

    
    
    # Sample via top-3
    logit_mod = partial(top_k_logits, k=3)
    sample_method = 'multinomial'

    return model, logit_mod, sample_method, ingr_map, memory_tensor_map

    
def attention_inference(name, ingredients, calorie, model, logit_mod, sample_method, ingr_map, memory_tensor_map):
    USE_CUDA, DEVICE = get_device()
    # INPUT DATA ============================
    name = name
    ingredients = ingredients
    calorie = calorie
    # ======================================
    
    input_obj = input_data(name=name, ingredients=ingredients, calorie=calorie)
    
    recipe_repr_list = eval_model(
        input = input_obj,
        device=DEVICE,
        model=model,
        logit_modifier_fxn=logit_mod,
        token_sampler=sample_method,
        max_len=MAX_STEP_TOK,
        max_name_len=MAX_NAME,
        ingr_map=ingr_map,
        **memory_tensor_map
    )

    return recipe_repr_list
'''
==== RUN CLUSTER ALL
python -m recipe_gen.models.baseline.test --data-dir <DATA FOLDER> --model-path <MODEL PATH> --vocab-emb-size 300 --calorie-emb-size 5 --nhid 256 --nlayers 2 --save-dir <OUTPUT FOLDER> --batch-size 48 --ingr-emb --ingr-gru
'''
if __name__ == "__main__":

    start = datetime.now()
    USE_CUDA, DEVICE = get_device()
    
    
    # TEST DATA ============================
    name = 'big mac pizza'
    ingredients = ['thousand island dressing', 'prepared pizza crust', 'lean ground beef', 'colby-monterey jack cheese', 'bacon']	
    calorie = 2  # 0-low, 1-medium, 2-high
    # ======================================
    
    input_obj = input_data(name=name, ingredients=ingredients, calorie=calorie)


    data_dir = 'data'         # location of the data corpus
    vocab_emb_dim = 50        # size of word embeddings
    calorie_emb_dim = 50      # size of calorie embeddings
    ingr_emb_dim = 10         # size of ingr embeddings
    hidden_size = 256         # number of hidden units per layer
    n_layers = 1              # number of layers
    batch_size = 1            # the name says it all
    model_path = 'model_weights/model_baseline-v1CANDIDATE.pt' # model weights
    save_dir = '/Users/gohyixian/Documents/GitHub/Attention-Recipe-Personalization/OUTPUT_FOLDER' # save model output
    overwrite = False         # Overwrite existing outputs
    ingr_gru = False          # Use BiGRU for ingredient encoding
    decode_name = False       # Multi-task learn to decode name along with recipe
    ingr_emb = False          # Use Ingr embedding in encoder
    shared_proj = False       # Share projection layers for name and steps
    ppx_only = False          # Only calculate perplexity (on full test set)
    n_samples = 1e9           # sample test items

    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    '''
    Load data
    '''
    # Get the DFs
    df_r, ingr_map = load_data(data_dir)
    n_items = len(df_r)
    print('{} - Data loaded.'.format(datetime.now() - start))

    # Pad recipe information
    N_INGREDIENTS = 0

    # masks are added here
    df_r = pad_recipe_info(
        df_r, max_name_tokens=MAX_NAME, max_ingredients=MAX_INGR,
        max_ingr_tokens=MAX_INGR_TOK, max_step_tokens=MAX_STEP_TOK
    )

    tensors_to_load = [
        ('name_tensor', 'name_tokens'),
        ('calorie_level_tensor', 'calorie_level'),
        ('technique_tensor', 'techniques'),
        ('ingr_tensor', 'ingredient_tokens'),
        ('steps_tensor', 'steps_tokens'),
        ('ingr_mask_tensor', 'ingredient_mask'),
        ('tech_mask_tensor', 'techniques_mask'),
    ]
    tensor_names, tensor_cols = zip(*tensors_to_load)

    # Load tensors into memory
    memory_tensors = load_recipe_tensors(
        df_r, DEVICE, cols=tensor_cols, types=[torch.LongTensor] * len(tensors_to_load)
    )
    memory_tensor_map = dict(zip(tensor_names, memory_tensors))
    print('{} - Tensors loaded in memory.'.format(datetime.now() - start))

    '''
    Create model
    '''
    model = create_model(
        vocab_emb_dim=vocab_emb_dim, 
        calorie_emb_dim=calorie_emb_dim,
        hidden_size=hidden_size, 
        n_layers=n_layers, 
        dropout=0.0,
        max_ingr=MAX_INGR, 
        max_ingr_tok=MAX_INGR_TOK, 
        use_cuda=USE_CUDA, 
        state_dict_path=model_path,
        ingr_gru=ingr_gru, 
        decode_name=decode_name,
        ingr_emb=ingr_emb, 
        num_ingr=N_INGREDIENTS, 
        ingr_emb_dim=ingr_emb_dim,
        shared_projection=shared_proj,
    )

    model_id = os.path.basename(model_path)[:-3]
    
    # Sample via top-3
    logit_mod = partial(top_k_logits, k=3)
    sample_method = 'multinomial'

    
    recipe_repr_list = eval_model(
        input = input_obj,
        device=DEVICE,
        model=model,
        logit_modifier_fxn=logit_mod,
        token_sampler=sample_method,
        max_len=MAX_STEP_TOK,
        max_name_len=MAX_NAME,
        ingr_map=ingr_map,
        **memory_tensor_map
    )

    print(recipe_repr_list)
    
