import argparse
import json
import os
import time

import torch
import torch.optim
from torch.utils import data

from data.Dubai_CC.DubaiCC import DubaiCCDataset
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.WHU_CDC.WHUCDC import WHUCDCDataset
from model.model_decoder import DecoderTransformer
from model.model_encoder import Encoder, DCPNetEncoder
from utils import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if os.path.exists(args.savepath) is False:
        os.makedirs(args.savepath)
    if args.data_name == 'LEVIR_CC':
        data_folder = 'dataset/LEVIR_CC/images'
        list_path = 'project/Chg2Cap/data/LEVIR_CC/'
        token_folder = 'project/Chg2Cap/data/LEVIR_CC/tokens/'
        max_length = 41
        data_name = 'LEVIR_CC'
    elif args.data_name == 'Dubai_CC':
        data_folder = 'dataset/datasetDubaiCCPublic/imgs_tiles/RGB'
        list_path = 'project/Chg2Cap/data/Dubai_CC/'
        token_folder = 'project/Chg2Cap/data/Dubai_CC/tokens/'
        max_length = 27
        data_name = 'Dubai_CC'
    elif args.data_name == 'WHU_CDC':
        data_folder = 'dataset/WHU_CDC/images'
        list_path = 'project/Chg2Cap/data/WHU_CDC/'
        token_folder = 'project/Chg2Cap/data/WHU_CDC/tokens/'
        max_length = 26
        data_name = 'WHU_CDC'
    with open(os.path.join(list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    checkpoint_name = args.checkpoint
    if not os.path.basename(checkpoint_name).startswith(data_name + '_'):
        checkpoint_name = data_name + '_' + checkpoint_name
    snapshot_full_path = checkpoint_name
    if not os.path.isabs(snapshot_full_path):
        snapshot_full_path = os.path.join(args.savepath, checkpoint_name)
    checkpoint = torch.load(snapshot_full_path, map_location='cpu')

    encoder = Encoder(args.network)
    encoder_trans = DCPNetEncoder(
        n_layers=args.n_layers,
        feature_size=(args.feat_size, args.feat_size, 2048),
        heads=args.n_heads,
        hidden_dim=args.feature_dim,
        semantic_dim=args.semantic_dim,
        dropout=args.dropout,
    )
    decoder = DecoderTransformer(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_vocab),
        max_lengths=max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout,
    )

    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])

    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    if args.data_name == 'LEVIR_CC':
        nochange_list = [
            'the scene is the same as before ',
            'there is no difference ',
            'the two scenes seem identical ',
            'no change has occurred ',
            'almost nothing has changed ',
        ]
        test_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'test', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif args.data_name == 'Dubai_CC':
        nochange_list = [
            'Nothing has changed ',
            'There is no difference ',
            'All remained the same ',
            'Everything remains the same ',
            'Nothing has changed in this area ',
            'Nothing changed in this area ',
            'No changes in this area ',
            'No change was made ',
            'There is no change to mention ',
            'No changes to mention ',
            'No changed to mention ',
            'No difference in this area ',
            'No change to mention ',
            'No change was made ',
            'No change occurred in this area ',
            'The area appears the same ',
        ]
        test_loader = data.DataLoader(
            DubaiCCDataset(data_folder, list_path, 'test', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif args.data_name == 'WHU_CDC':
        nochange_list = [
            'the scene is the same as before ',
            'there is no difference ',
            'the two scenes seem identical ',
            'no change has occurred ',
            'almost nothing has changed ',
        ]
        test_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'test', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        raise ValueError(f'Unsupported data_name: {args.data_name}')

    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    test_start_time = time.time()
    references = []
    hypotheses = []
    prediction_records = []
    change_references = []
    change_hypotheses = []
    nochange_references = []
    nochange_hypotheses = []
    change_acc = 0
    nochange_acc = 0

    with torch.no_grad():
        for _, (imgA, imgB, token_all, token_all_len, _, _, names) in enumerate(test_loader):
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resize1(imgA)
                imgB = l_resize2(imgB)
            token_all = token_all.squeeze(0).cuda()

            feat1, feat2 = encoder(imgA, imgB)
            enc1, enc2 = encoder_trans(feat1, feat2)
            seq = decoder.sample(enc1, enc2)

            img_token = token_all.tolist()
            img_tokens = list(
                map(
                    lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                    img_token,
                )
            )
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)

            pred_caption = ''
            for i in pred_seq:
                pred_caption += list(word_vocab.keys())[i] + ' '
            prediction_records.append({"name": names[0], "pred_caption": pred_caption.strip()})
            ref_caption = ''
            for i in img_tokens[0]:
                ref_caption += list(word_vocab.keys())[i] + ' '

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc += 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc += 1

        if args.output_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
            with open(args.output_jsonl, "w") as f:
                for record in prediction_records:
                    f.write(json.dumps(record) + "\n")
            print("saved predictions:", args.output_jsonl, len(prediction_records))
        if args.predictions_only:
            return
        test_time = time.time() - test_start_time
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print(
                'BLEU-1: {0:.4f}	BLEU-2: {1:.4f}	BLEU-3: {2:.4f}	BLEU-4: {3:.4f}	Meteor: {4:.4f}	Rouge: {5:.4f}	Cider: {6:.4f}	'.format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print('nochange_acc:', nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print(
                'BLEU-1: {0:.4f}	BLEU-2: {1:.4f}	BLEU-3: {2:.4f}	BLEU-4: {3:.4f}	Meteor: {4:.4f}	Rouge: {5:.4f}	Cider: {6:.4f}	'.format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print('change_acc:', change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        Average = (Bleu_4 + Meteor + Rouge + Cider) / 4.0
        print(
            'Testing:\nTime: {0:.3f}	BLEU-1: {1:.4f}	BLEU-2: {2:.4f}	BLEU-3: {3:.4f}	BLEU-4: {4:.4f} Meteor: {5:.4f}	Rouge: {6:.4f}	Cider: {7:.4f}	Average: {8:.4f} '.format(
                test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider, Average
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='path of the data lists')
    parser.add_argument('--data_name', default='LEVIR_CC', choices=['LEVIR_CC', 'Dubai_CC', 'WHU_CDC'], help='base name shared by data files.')
    parser.add_argument('--checkpoint', default='LEVIR_CC_batchsize_32_resnet101.pth', help='path to checkpoint')

    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features:resnet101,vgg16')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading')
    parser.add_argument('--encoder_dim', type=int, default=512, help='the dimension of extracted features using different network')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--semantic_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=8)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--test_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default='project/DCPNet/models_ckpt/')
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--predictions_only", action="store_true")

    args = parser.parse_args()
    main(args)
