---
layout:     post
title:      如何基于huggingface的model.generate修改自己想要的生成方法
subtitle:   
date:       2024-07-25
author:     BY
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - Blog
---

1.背景：在输入端增加新的信息，在生成的时候每个step需要修改输入，看看怎么基于transformers进行操作，主要是model.generate方法

2.梳理model.generate调用链

```
model.generate下面有很多解码策略，此处以greedy为例，看看greedy里面操作了什么

循环部分主要在transformers.utils.GenerationMixin的_sample方法里

class GenerationMixin:
    def _sample(···):
        ···
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
>>>>>>>>>>>># prepare model inputs # WE DO NOTHING HERE !!!
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
>>>>>>>>>>>># AND WE DO SOMETHING HERE
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
```

3.修改逻辑

```
1.对input进行增加信息操作 (generate输入时就要有，当然也可以放到prepare_inputs，但会有问题，这个主要涉及配置问题，与本身逻辑无关)
2.往模型输入中加入新增信息，通过kwargs字段， copy到model_inputs里面，这个是prepare_inputs的返回值
3.模型前向完成后，更新model_kwargs，其实就是更新新增信息（修改_update_model_kwargs_for_generation）
```
