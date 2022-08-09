#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch
import os
import copy

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils#bleu, 
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
        bert_ratio=args.bert_ratio if args.change_ratio else None,
        encoder_ratio=args.encoder_ratio if args.change_ratio else None,
        geargs=args,
    )
    
    argsCopy = copy.deepcopy(args)
    argsCopy.path = 'checkpoints/iwed_de_en_0.5/checkpoint300.pt'
    argsCopy.source_lang = 'de'
    argsCopy.data='../bert-nmt/examples/translation/destdir_de'
    argsCopy.save_dir = 'checkpoints/iwed_de_en_0.5'
    taskCopy = tasks.setup_task(argsCopy)
    print('| loading model(s) from {}'.format(argsCopy.path))
    models2, _model_args2 = checkpoint_utils.load_model_ensemble(
        'checkpoints/iwed_fr_en_0.5/checkpoint300.pt'.split(':'),
        arg_overrides=eval(argsCopy.model_overrides),
        task=taskCopy,
        bert_ratio=argsCopy.bert_ratio if argsCopy.change_ratio else None,
        encoder_ratio=argsCopy.encoder_ratio if argsCopy.change_ratio else None,
        geargs=argsCopy,
    )
    
    
    model_dict = models[0].state_dict()
    model_dict2 = models2[0].state_dict()
    
    counter = 0
    for k,v in model_dict2.items():
        counter = counter + 1
        if k.startswith('decoder'):
            #print(counter)
            #print(v[1:10])
            #print('sepeter')
            #print(model_dict.get(k)[1:10])
            model_dict.update({k : v})
            #print('after update')
            #print(model_dict.get(k)[1:10])
    
    models[0].load_state_dict(model_dict)
    model_dict = models[0].state_dict()
    model_dict2 = models2[0].state_dict()
    '''
    counter = 0
    for k,v in model_dict2.items():
        counter = counter + 1
        #if k.startswith('decoder'):
        print(counter)
        print(k)
        print(len(v))
    '''
    # Optimize ensemble for generation    
    for model in models:     
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    #if args.sacrebleu:
    #    scorer = bleu.SacrebleuScorer()
    #else:
    #    scorer = bleu. (tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    embeddings_1 = torch.empty([1,20480])
    embeddings_2 = torch.empty([1,88320])
    truths = []
    hyperthes = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            embedding_1 = task.inference_embedding(generator, models, sample, prefix_tokens)#, embedding_2
            print(embeddings_1.size())
            #print(embeddings_2.size())
            print('S-{}'.format(embedding_1[0].view(128,-1).size()))
            #print('S-{}'.format(embedding_2[0].get('bert_encoder_out').view(128,-1).size()))
            
            #embeddings_1 = torch.cat((embeddings_1, embedding_1[0].get('encoder_out').view(128,-1)), 0)
            #embeddings_2 = torch.cat((embeddings_2, embedding_2[0].get('bert_encoder_out').view(128,-1)), 0)
            print(len (sample['id'].tolist()))
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)
            
            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))
                        truths.append(target_str)

                # Process top predictions
                for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        hyperthes.append(hypo_str)
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        #if hasattr(scorer, 'add_string'):
                        #    scorer.add_string(target_str, hypo_str)
                        #else:
                        #    scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    #if has_target:
    #    print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    
    print(embeddings_1.size())
    print(embeddings_2.size())
    
    fp = open("truths_fr2fr.txt", "a") 
    fp.write('\n'.join(truths))
    fp.close()
    
    fp = open("hyperthes_fr2fr.txt", "a") 
    fp.write('\n'.join(hyperthes))
    fp.close()
    
    return 0.0#scorer
    


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
