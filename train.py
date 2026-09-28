import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from data.WHU_CDC.WHUCDC import WHUCDCDataset
from model.model_encoder import Encoder, DCPNetEncoder
from model.model_decoder import DecoderTransformer
from utils import *

import torch

def main(args):
    """
    Training and validation.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    best_bleu4 = float('-inf')
    start_epoch = 0
    if args.data_name == 'LEVIR_CC':
        data_folder = 'dataset/LEVIR_CC/images'
        list_path = 'project/DCPNet/data/LEVIR_CC/'
        token_folder = 'project/DCPNet/data/LEVIR_CC/tokens/'
        max_length = 41
    elif args.data_name == 'Dubai_CC':
        data_folder = 'dataset/datasetDubaiCCPublic/imgs_tiles/RGB'
        list_path = 'project/DCPNet/data/Dubai_CC/'
        token_folder = 'project/DCPNet/data/Dubai_CC/tokens/'
        max_length = 27
    elif args.data_name == 'WHU_CDC':
        data_folder = 'dataset/WHU_CDC/images'
        list_path = 'project/DCPNet/data/WHU_CDC/'
        token_folder = 'project/DCPNet/data/WHU_CDC/tokens/'
        max_length = 26
    with open(os.path.join(list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu') if args.checkpoint else None
    if checkpoint is None or 'encoder_dict' in checkpoint:
        encoder = Encoder(args.network)   
        encoder.fine_tune(args.fine_tune_encoder)     
        encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                            lr=args.encoder_lr) if args.fine_tune_encoder else None
        encoder_trans = DCPNetEncoder(
            n_layers=args.n_layers,
            feature_size=(args.feat_size, args.feat_size, 2048),
            heads=args.n_heads,
            hidden_dim=args.feature_dim,
            semantic_dim=args.semantic_dim,
            dropout=args.dropout,
        )
        encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                            lr=args.encoder_lr)
        decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
        if checkpoint is not None:
            encoder.load_state_dict(checkpoint['encoder_dict'])
            encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
            decoder.load_state_dict(checkpoint['decoder_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_bleu4 = checkpoint.get('best_bleu4', float('-inf'))
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint.get('best_bleu4', checkpoint.get('bleu-4', float('-inf')))
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=args.encoder_lr)
    # Move to GPU, if available
    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()
    # Loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'train', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'val', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'Dubai_CC':
        train_loader = data.DataLoader(
            DubaiCCDataset(data_folder, list_path, 'train', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            DubaiCCDataset(data_folder, list_path, 'val', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'WHU_CDC':
        train_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'train', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'val', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.5) if args.fine_tune_encoder else None
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_trans_optimizer, step_size=5, gamma=0.5)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs

    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_loader):
            #if id == 20:
            #    break
            start_time = time.time()
            decoder.train()  # train mode (dropout and batchnorm is used)
            encoder.train()
            encoder_trans.train()
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            # Forward prop.
            feat1, feat2 = encoder(imgA, imgB)         
            enc1, enc2, aux_dict = encoder_trans(feat1, feat2, return_aux=True)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(enc1, enc2, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # Calculate loss
            caption_loss = criterion(scores, targets)
            saa_loss = aux_dict['saa_loss']
            loss = caption_loss + args.saa_loss_weight * saa_loss
            # Back prop.
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)

            # Update weights                      
            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            index_i += 1   
            # Print status
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs*len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))
        # One epoch's validation
        decoder.eval()  # eval mode (no dropout or batchnorm)
        encoder_trans.eval()
        if encoder is not None:
            encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate(val_loader):
                # Move to GPU, if available
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                if args.data_name == 'Dubai_CC':
                    imgA = l_resizeA(imgA)
                    imgB = l_resizeB(imgB)
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                if encoder is not None:
                    feat1, feat2 = encoder(imgA, imgB)
                enc1, enc2 = encoder_trans(feat1, feat2)
                seq = decoder.sample(enc1, enc2, k=1)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Calculate evaluation scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            average = (Bleu_4 + Meteor + Rouge + Cider) / 4.0
            
            print('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-4: {1:.4f}\t' 'Meteor: {2:.4f}\t' 'Rouge: {3:.4f}\t' 'Cider: {4:.4f}\t' 'Average: {5:.4f}\t'
                .format(val_time, Bleu_4, Meteor, Rouge, Cider, average))
        
        #Adjust learning rate
        decoder_lr_scheduler.step()
        print(decoder_optimizer.param_groups[0]['lr'])
        encoder_trans_lr_scheduler.step()
        if encoder_lr_scheduler is not None:
            encoder_lr_scheduler.step()
            print(encoder_optimizer.param_groups[0]['lr'])
        # Save whenever validation BLEU-4 improves.
        if Bleu_4 > best_bleu4:
            best_bleu4 = Bleu_4
            print('Save Model')
            state = {'epoch': epoch,
                    'best_bleu4': best_bleu4,
                    'encoder_dict': encoder.state_dict(),
                    'encoder_trans_dict': encoder_trans.state_dict(),
                    'decoder_dict': decoder.state_dict()
                    }
            model_name = (
                str(args.data_name) + '_batchsize_' + str(args.train_batchsize)
                + '_' + str(args.network)
                + '_epo_' + str(epoch)
                + '_Bleu4_' + str(round(10000 * best_bleu4)) + '.pth'
            )
            torch.save(state, os.path.join(args.savepath, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", choices=['LEVIR_CC', 'Dubai_CC', 'WHU_CDC'], help='base name shared by data files.')

    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=8, help='batch_size for training')
    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim',default=512, help='the dimension of extracted features using different network')
    parser.add_argument('--num_epochs', type=int, default=11, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="project/DCPNet/models_checkpoint/")
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--semantic_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=8)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--saa_loss_weight', type=float, default=0.2, help='weight for symmetric-anchor triplet loss on prompt features')
    args = parser.parse_args()
    main(args)
