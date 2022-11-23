import copy

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput
)
from typing import Union, Tuple, List, Iterable, Dict, Callable
import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F

from custom_t5 import CustomT5Stack


class HeroGAN(T5ForConditionalGeneration):
    def __init__(self,
                 config,
                 max_temperature=10,
                 sentence_embedding_dimension: int = 768,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_norm: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.BCEWithLogitsLoss()):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared)

        self.temperature = 1  # init value is 1.0
        self.max_temperature = max_temperature

        self.word_embedding_dimension = self.model_dim
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_norm = pooling_mode_norm
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.sentence_embedding_dimension = sentence_embedding_dimension

        self.project = nn.Linear(self.word_embedding_dimension, sentence_embedding_dimension)
        self.norm = nn.LayerNorm(sentence_embedding_dimension)

        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, 1)
        self.loss_fct = loss_fct
        self.mle_loss_fct = CrossEntropyLoss(ignore_index=-100)

    def get_sentence_embedding(
            self,
            features
    ):
        if 'encoder_outputs' in features:
            token_embeddings = features['encoder_outputs']['last_hidden_state']
        else:
            output_states = self.encoder(
                input_ids=features['input_ids'],
                attention_mask=features['attention_mask'],
                return_dict=False,
            )
            token_embeddings = output_states[0]

        features.update({'token_embeddings': token_embeddings, 'attention_mask': features['attention_mask']})

        attention_mask = features['attention_mask']

        ## Pooling strategy
        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            vectors = cls_token
        elif self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                vectors = sum_embeddings / sum_mask
            elif self.pooling_mode_mean_sqrt_len_tokens:
                vectors = sum_embeddings / torch.sqrt(sum_mask)

        output = self.project(vectors)
        if self.pooling_mode_norm:
            output = self.norm(output)
        features.update({'sentence_embedding': output})
        return features

    def compute_score_matching(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        reps = [self.get_sentence_embedding(sentence_feature)['sentence_embedding'] for sentence_feature in
                sentence_features]
        rep_a, rep_b = reps

        if rep_a.shape[0] != rep_b.shape[0]:
            rep_a = torch.cat((rep_a, rep_a), dim=0)

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)

        if labels is not None:
            labels = labels.unsqueeze(1).float()
            loss = self.loss_fct(output, labels)
            # return loss, torch.softmax(output, dim=-1).detach()
            return loss, torch.sigmoid(output).detach()
        else:
            # return torch.mean(torch.softmax(output, dim=-1), dim=0), output
            return torch.sigmoid(output), output

    def calculate_score(self, sentence_embeddings):
        rep_a, rep_b = sentence_embeddings

        if rep_a.shape[0] != rep_b.shape[0]:
            rep_a = torch.cat((rep_a, rep_a), dim=0)

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat)
        output = self.classifier(features)

        return F.sigmoid(output).detach()

    def forward(
            self,
            input_ids=None,  #
            attention_mask=None,  #
            decoder_input_ids=None,  #
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,  #
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = NLLLoss(ignore_index=-100)
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = self.mle_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
