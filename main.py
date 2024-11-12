import argparse
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef

VAL_DIRECTIONS = ['deu_Latn_eng_Latn', 'eng_Latn_deu_Latn']

# Functions
def prepare_data(input_path, load_embed, embed_methods):
    """Load and prepare the dataset."""
    if load_embed:
        # Load embeddings with converters for embedded columns.
        embed_cols = [f'{method}_src_embeds' for method in embed_methods] + [f'{method}_mt_embeds' for method in embed_methods]
        converters = {col: ast.literal_eval for col in embed_cols}
        df = pd.read_csv(input_path, converters=converters)
    else:
        # TODO: generate embeddings
        df = pd.read_csv(input_path, sep="\t")

    df = df[df.perturbation == 'natural'].reset_index(drop=True)
    df['binary_hall'] = df['class_hall'].apply(lambda x: 0 if str(x).startswith('1') else 1)
    
    df_val = df[df.direction.isin(VAL_DIRECTIONS)].reset_index(drop=True)
    df_test = df[~df.direction.isin(VAL_DIRECTIONS)].reset_index(drop=True)

    return df_val, df_test

def calculate_similarity(df, methods):
    """Calculate cosine similarity scores between source and translated texts for each embedding space."""
    for method in methods:
        df[f'score_{method}_cos'] = df.apply(
            lambda x: -cosine_similarity([x[f'{method}_src_embeds']], [x[f'{method}_mt_embeds']])[0][0], axis=1
        )
    return df

def binarize_scores(df_val, df_test, methods):
    """Binarize similarity scores based on the optimal threshold."""
    for method in methods:
        # Identify the optimal threshold using F1 score
        y_true = df_val['binary_hall']
        y_pred = df_val[f'score_{method}_cos']
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        optimal_thresh = thresholds[np.argmax(fscore)]

        # Binarize cosine similarities 
        df_test[f'bin_{method}_cos'] = df_test[f'score_{method}_cos'].apply(lambda x: 0 if x < optimal_thresh else 1)
    
    return df_test

# Main function
def main(args):
    # Load and prepare data
    df_val, df_test = prepare_data(args.input, args.load_embed, args.embed_methods)
    
    # ----------Embeddings---------
    # Calculate similarity scores
    df_val = calculate_similarity(df_val, args.embed_methods)
    df_test = calculate_similarity(df_test, args.embed_methods)
    # Binarize scores based on optimal threshold
    df_test = binarize_scores(df_val, df_test, args.embed_methods)
    
    # -----------Evaluate------------
    df_eval = df_test.groupby("direction", group_keys=True).apply(
        lambda x: pd.Series(
            {method: matthews_corrcoef(x['binary_hall'], x[f'bin_{method}_cos']) for method in args.embed_methods}
        )
        )
    print("Evaluation results:")
    print(df_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT Hallucination Detection")
    parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('--load_embed', type=bool, required=False, help='Load pre-computed sentence embeddings if set, otherwise generate new ones')
    parser.add_argument('--embed_methods', nargs='+', required=False, choices=['sonar', 'gpt', 'cohere', 'mistral'], help='Embedding methods to evaluate')
    
    args = parser.parse_args()
    main(args)
