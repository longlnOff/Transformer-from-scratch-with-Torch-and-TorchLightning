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
                          split='train', cache_dir=path_git + '/' + 'DataFolder')
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    torch.manual_seed(0)
    train_ds_size = int(config['train_ds_size'] * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    

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
    # print(train_ds[0])
    
    
    val_dataloader = DataLoader(val_ds,
                                batch_size=1,
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

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
    sos_ids = tokenizer_tgt.token_to_id("[SOS]")
    eos_ids = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from decoder
    encoder_output = model.encode(source, source_mask) # (batch_size, seq_len, d_model)

    # Initialise the decoder input with SOS token
    decoder_input = torch.empty(1,1).fill_(sos_ids).type_as(source).to(device) # (batch_size, 1)
    while True:
        if decoder_input.shape[1] >= max_len:
            break

        # Build mask for the target
        decoder_mask = casual_mask(decoder_input.shape[1]).type_as(source).to(device) # (batch_size, 1, seq_len, seq_len)

        # Calculate output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)

        # Get the next token
        prob = model.project(out[:, -1]) # (batch_size, vocab_size)
        # greedy search (get maximum token)
        _, next_token = torch.max(prob, dim=1) # (batch_size, 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_token.item()).to(device)], dim=1) # (batch_size, seq_len)

        if next_token == eos_ids:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    # size of the control window (just use default value)
    console_width = 80

    with torch.no_grad():
        # when inference, batch_size = 1
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)   # (batch_size, seq_len)
            encoder_mask  = batch['encoder_mask'].to(device)    # (batch_size, 1, 1, seq_len)
            
            # check the batch size
            assert encoder_input.shape[0] == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            predicted_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # print the result to the console
            print_msg('-'*console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {predicted_text}')

            if count == num_examples:
                break







def train_model(config):
    # Define the device
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
    if config['preload'] == 'latest':
        model_filename = latest_weights_file_path(config)
        print('preload model {0}'.format(model_filename))
    elif config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print('preload model {0}'.format(model_filename))
    else:
        model_filename = None

    if model_filename is not None:
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])


    # print model size
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')


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
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)   # (batch_size, seq_len, d_model)
            project_output = model.project(decoder_output)   # (batch_size, seq_len, vocab_size)

            label = batch['label'].to(device)   # (batch_size, seq_len)
            

            # Calculate loss
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': loss.item()})

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            # Back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)




            global_step += 1

        run_validation(model=model,
                        validation_ds=val_dataloader,
                        tokenizer_src=tokenizer_src,
                        tokenizer_tgt=tokenizer_tgt,
                        max_len=config['seq_len'],
                        print_msg=lambda msg: batch_iterator.write(msg),
                        global_state=global_step,
                        writer=writer,
                        num_examples=2)
        
        # Save the model every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        print(f"{model_filename}")

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

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    for i in train_dataloader:
        print(i['label'])
        break