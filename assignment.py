import os
import argparse
import numpy as np
import pickle
import torch
import torch.optim as optim

from model import ImageCaptionModel, accuracy_function, loss_function
from decoder import TransformerDecoder, RNNDecoder


def parse_args(args=None):
    """
    Perform command-line argument parsing (or parse arguments with defaults).
    To parse in an interactive context (e.g. in a notebook), pass a list:

        parse_args(['--type', 'rnn', '--task', 'train', '--data', 'data/data.p'])
    """
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--type',        required=True,             choices=['rnn', 'transformer'],    help='Type of model to train')
    parser.add_argument('--task',        required=True,             choices=['train', 'test', 'both'], help='Task to run')
    parser.add_argument('--data',        required=True,                                               help='File path to the assignment data file.')
    parser.add_argument('--epochs',      type=int,   default=3,                                       help='Number of epochs used in training.')
    parser.add_argument('--lr',          type=float, default=1e-3,                                    help="Model's learning rate")
    parser.add_argument('--optimizer',   type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help="Model's optimizer")
    parser.add_argument('--batch_size',  type=int,   default=100,                                     help="Model's batch size.")
    parser.add_argument('--hidden_size', type=int,   default=256,                                     help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size', type=int,   default=20,                                      help='Window size of text entries.')
    parser.add_argument('--chkpt_path', default='',                                                   help='where the model checkpoint is saved/loaded')
    parser.add_argument('--check_valid', default=True, action='store_true',                           help='if training, also print validation after each epoch')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cpu / cuda)')

    if args is None:
        return parser.parse_args()       # called from command line
    return parser.parse_args(args)      # called from notebook / tests


def main(args):

    device = torch.device(args.device)
    print(f"Using device: {device}")

    ##############################################################################
    ## Data Loading
    with open(args.data, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)

    train_captions  = torch.tensor(np.array(data_dict['train_captions']),  dtype=torch.long)
    test_captions   = torch.tensor(np.array(data_dict['test_captions']),   dtype=torch.long)
    train_img_feats = torch.tensor(feat_prep(data_dict['train_image_features']), dtype=torch.float32)
    test_img_feats  = torch.tensor(feat_prep(data_dict['test_image_features']),  dtype=torch.float32)
    word2idx        = data_dict['word2idx']

    # Move data to the selected device
    train_captions  = train_captions.to(device)
    test_captions   = test_captions.to(device)
    train_img_feats = train_img_feats.to(device)
    test_img_feats  = test_img_feats.to(device)

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):

        decoder_class = {
            'rnn'         : RNNDecoder,
            'transformer' : TransformerDecoder,
        }[args.type]

        decoder = decoder_class(
            vocab_size  = len(word2idx),
            hidden_size = args.hidden_size,
            window_size = args.window_size,
        )

        model = ImageCaptionModel(decoder).to(device)
        compile_model(model, args)

        train_model(
            model, train_captions, train_img_feats, word2idx['<pad>'], args,
            valid=(test_captions, test_img_feats)
        )

        if args.chkpt_path:
            save_model(model, args)

    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        if args.task != 'both':
            model = load_model(args, device)

        if not (args.task == 'both' and args.check_valid):
            test_model(model, test_captions, test_img_feats, word2idx['<pad>'], args)

    ##############################################################################


##############################################################################
## UTILITY METHODS

def save_model(model, args):
    """Saves model checkpoint."""
    os.makedirs(args.chkpt_path, exist_ok=True)
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'decoder_type'     : args.type,
        'vocab_size'       : model.decoder.vocab_size,
        'hidden_size'      : model.decoder.hidden_size,
        'window_size'      : model.decoder.window_size,
        'args'             : vars(args),
    }
    torch.save(checkpoint, os.path.join(args.chkpt_path, 'model.pt'))
    print(f"Model saved to {args.chkpt_path}/model.pt")


def load_model(args, device=None):
    """Loads model from checkpoint."""
    if device is None:
        device = torch.device(args.device if hasattr(args, 'device') else 'cpu')
    checkpoint = torch.load(os.path.join(args.chkpt_path, 'model.pt'), map_location=device)

    decoder_class = {
        'rnn'         : RNNDecoder,
        'transformer' : TransformerDecoder,
    }[checkpoint['decoder_type']]

    decoder = decoder_class(
        vocab_size  = checkpoint['vocab_size'],
        hidden_size = checkpoint['hidden_size'],
        window_size = checkpoint['window_size'],
    )
    model = ImageCaptionModel(decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Rebuild compile so that loss/accuracy functions are attached
    from types import SimpleNamespace
    saved_args = SimpleNamespace(**checkpoint['args'])
    compile_model(model, saved_args)

    print(f"Model loaded from '{args.chkpt_path}/model.pt'")
    return model


def compile_model(model, args):
    """Attaches optimizer and loss/metric functions to the model."""
    optimizer_map = {
        'adam'    : optim.Adam,
        'rmsprop' : optim.RMSprop,
        'sgd'     : optim.SGD,
    }
    optimizer = optimizer_map[args.optimizer](model.parameters(), lr=args.lr)
    model.compile(
        optimizer = optimizer,
        loss      = loss_function,
        metrics   = [accuracy_function],
    )


def train_model(model, captions, img_feats, pad_idx, args, valid=None):
    """
    Runs the full training loop for args.epochs epochs.
    Calls model.train_epoch() for each epoch and optionally runs validation.
    """
    for epoch in range(args.epochs):
        print(f"[Epoch {epoch+1}/{args.epochs}]")
        try:
            model.train_epoch(captions, img_feats, pad_idx, batch_size=args.batch_size)
        except KeyboardInterrupt:
            if epoch > 0:
                print("\nEarly stopping via keyboard interrupt.")
                break
            raise

        if args.check_valid and valid is not None:
            model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)


def test_model(model, captions, img_feats, pad_idx, args):
    """Runs one test epoch and returns (perplexity, accuracy)."""
    return model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())
