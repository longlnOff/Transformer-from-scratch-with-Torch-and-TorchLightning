from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
from Source.ConfigureInformation import *
from Source.DataClass import *
from Source.models.Transformer import *

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', 
                          f'{config["lang_src"]}-{config["lang_tgt"]}',
                          split='train')
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # train test split
    train_ds_size = int(len(ds_raw) * config['train_ds_size'])
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = ds_raw.train_test_split(
        test_size=val_ds_size, shuffle=False)
    

    train_ds = BilingualDataset(train_ds_raw,
                                tokenizer_src,
                                tokenizer_tgt,
                                config['lang_src'],
                                config['lang_tgt'],
                                config['seq_len'],)

    val_ds = BilingualDataset(val_ds_raw,
                                tokenizer_src,
                                tokenizer_tgt,
                                config['lang_src'],
                                config['lang_tgt'],
                                config['seq_len'],)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    # print(src_ids, '---', tgt_ids)
    print('max_len_src: ', max_len_src)
    print('max_len_tgt: ', max_len_tgt)

    train_dataloader = DataLoader(train_ds,
                                    batch_size=config['batch_size'],
                                    num_workers=4,
                                    shuffle=True)
    
    val_dataloader = DataLoader(val_ds,
                                    batch_size=config['batch_size'],
                                    num_workers=4,
                                    shuffle=False)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_model(src_vocab_size=vocab_src_len,
                        tar_vocab_size=vocab_tgt_len,
                        src_seq_len=config['seq_len'],
                        tar_seq_len=config['seq_len'],
                        d_model=config['d_model'],
                        d_ff=config['d_ff'],
                        num_head_attention=config['num_head_attention'],
                        num_encoder_blocks=config['num_encoder_blocks'],
                        num_decoder_blocks=config['num_decoder_blocks'],
                        dropout=config['dropout'],)

    return model





def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print('preload model {0}'.format(model_filename))
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])


    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch{epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     # (batch_size, 1, seq_len, seq_len)

            # Run the tensors through the transformer model
            encoder_output = model.encoder(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decoder(decoder_input, decoder_mask, encoder_output, encoder_mask)   # (batch_size, seq_len, d_model)
            project_output = model.project(decoder_output)   # (batch_size, seq_len, vocab_size)

            label = batch['label'].to(device)   # (batch_size, seq_len)

            # Calculate loss
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': loss.item()})

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Save the model every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')

        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(state, model_filename)   


if __name__ == '__main__':

    config = get_config()
    train_model(config)
    print('Done!')
