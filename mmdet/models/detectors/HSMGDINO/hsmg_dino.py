# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ...layers import SinePositionalEncoding
from ...layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from ..dino import DINO
from ..glip import (create_positive_map, create_positive_map_label_to_token,
                    run_ner)


def clean_label_name(name: str) -> str:
    # re.sub()将name中符合r'\(.*\)'的替换为''
    # 这里是用于去除干扰字符
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class HSMGDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 hsm_dict=dict(
                     v_dim=256,
                     hsm_prototype_num=8,
                     hsm_queries=100,
                     hsm_lr=0.01,
                     hsm_velocity_momentum=0.9,
                     # hsm_save_pth="",
                 ),
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        # TODO: hsm
        self.hsm_queries = hsm_dict.get("hsm_queries")
        self.hsm_prototype_num = hsm_dict.get("hsm_prototype_num")
        self.hsm_lr = hsm_dict.get("hsm_lr")
        self.hsm_v_dim = hsm_dict.get("v_dim")
        self.hsm_velocity_momentum = hsm_dict.get("hsm_velocity_momentum")
        # self.hsm_save_pth = hsm_save_pth
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        # self.embed_dims = self.layers[0].embed_dims
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

        # TODO: hsm
        # import os
        # if os.path.exists(self.hsm_save_pth):
        #     self.hard_sample_memory = torch.load(self.hsm_save_pth)
        # else:
        #     self.hard_sample_memory = torch.zeros(self.hsm_prototype_num, self.hsm_v_dim, dtype=torch.float32,
        #                                           require_grad=False)
        device = torch.device('cuda')
        self.hard_sample_memory = torch.randn(self.hsm_prototype_num, self.hsm_v_dim, dtype=torch.float32,
                                              requires_grad=False).to(device)
        self.hard_sample_memory = torch.nn.functional.normalize(self.hard_sample_memory, p=2, dim=-1)
        self.hsm_prototype = 0
        self.hsm_velocity = torch.zeros(self.hsm_prototype_num, self.hsm_v_dim, dtype=torch.float32,
                                        requires_grad=False).to(device)
        # # note the class with hard sample memory
        # self.hsm_init = [False for _ in range(self.hsm_prototype_num)]

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        '''
        将enhanced_text_prompts加入original_caption，并转换为:前缀+name+后缀+self._special_tokens。
        目的是增补输入text_prompts，加入更多的“name”

        Args:
            original_caption: 主要指需要检测的class名字
            enhanced_text_prompts:

        Returns:
            caption_string：添加了enhanced_text_prompts后的original_caption，且转换为了
                前缀+name+后缀+self._special_tokens
            tokens_positive：记录了每个“name”的[start_i, end_i]，也即有效token的masks用于排除前后缀等附加字符的干扰

        '''
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                # 加入token前缀标志
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                # 加入token后缀标志
                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False,
            enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            # 如果original_caption是一个token且表示class名字(也即二分类)
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            # 去除冗余或干扰字符
            original_caption = [clean_label_name(i) for i in original_caption]

            # caption_string是转换为“前缀+name+后缀+self._special_tokens”的prompts，
            # tokens_positive记录每个"name"的[start_i, end_i]
            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        '''

        Args:
            tokenized: 经过预处理和分词的输入text_prompts
            tokens_positive: 和gt_labels对应上的text_prompts位置（包含了label-clsName对的信息）；
                这里tokens_positive.shape=(gt_bbox_labels, 2)

        Returns:
            positive_map_label_to_token：positive_map_label_to_token[第i + plus个gt_bbox]=token对应label的相对编码
            positive_map：positive_map[i,j] = True,i表示gt的第i个gt_bbox，j为该gt_bbox对应的tokens_positive位置
        '''
        # positive_map[i,j] = True, 这里True值是归一化值(1.0/gt_num)
        # i表示gt的第i个gt_bbox，j为该gt_bbox对应的tokens_positive位置
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False,
            enhanced_text_prompt: Optional[ConfigType] = None,
            tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                       positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
            caption_string, \
            positive_map, \
            entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
               positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
               caption_string_chunked, \
               positive_map_chunked, \
               entities_chunked

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            text_dict: Dict,
            batch_data_samples: OptSampleList = None,
    ) -> Dict:
        '''

        Args:
            img_feats: 5 level的特征图
            text_dict: text_dict包含'embedded'，'masks'，'hidden'
            batch_data_samples: 包含gt的annotations

        Returns:

        '''
        # encoder_inputs_dict将多尺度feature maps拼接：“feat”为(bs, num_feat_points, dim)，“feat_pos”为对应“feat”的位置编码
        # decoder_inputs_dict主要包含mask信息（encoder_inputs_dict也包含了）
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        # encoder_outputs_dict包含了fusion(cross attn)后的vision特征（memory），language特征（memory_text）和两者对应的mask
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        # tmp_dec_in主要为concat了dn_label_query的query和concat了dn_bbox_query的reference_points，memory和memory_text不变
        # head_inputs_dict主要为enc_outputs_class（也即两个模态的对比得分矩阵）的topk_score和topk_coords
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        # decoder_outputs_dict的内容 hidden_states=inter_states, references=list(references)
        # hidden_states是decoder每一层输出的query的堆叠集合，(num_decoder_layers, num_queries, bs, embed_dims)
        # references是根据每一层输出query用self.bbox_head.reg_branches做回归得到的坐标的堆叠集合，(num_decoder_layers, bs, num_queries, 4)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        '''

        Args:
            feat: (bs, num_feat_points, dim)
            feat_mask: (bs, num_feat_points)
            feat_pos: (bs, num_feat_points, dim)
            spatial_shapes: (num_level, 2)
            level_start_index: (num_level, 1)
            valid_ratios: (bs, num_level, 2)
            text_dict: text_dict包含'embedded'，'masks'，'hidden'，embedded.shape=[bs, len_input_text, language_dim]

        Returns:

        '''
        text_token_mask = text_dict['text_token_mask']
        # 对应论文的Fig.3的feature enhancer layer
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            memory_text: Tensor,
            text_token_mask: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        '''
.
        Args:
            memory: (bs, num_queries, v_dim)
            memory_mask: (bs, num_queries)
            spatial_shapes: has shape (num_levels, 2), last dimension represents (h, w)
            memory_text: (bs, len_text, text_embed_dims)
            text_token_mask: (bs,len_text)
            batch_data_samples: annotations

        Returns:

        '''
        bs, _, c = memory.shape

        # output_memory, output_proposals的shape=(bs, num_feat_points, dim)和(bs, num_feat_points, 4)
        # 这里基于anchor-based生成每个尺度特征图所有有效feat_points的anchor，output_proposals最后一维信息为(cx, cy, dh, dw)
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        # enc_outputs_class.shape=(bs, num_feat_points, max_text_len)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        # reg_branches是fc层，生成enc_outputs_coord_unact.shape=(bs, num_feat_points, 4)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                      self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        # torch.tensor.max()返回两个值，[0]为value，[1]为indices。torch.topk()相同，返回value, indices
        # 所以topk_indices为(bs, num_queries)，表示置信度得分最高的self.num_queries个proposals的indices
        # self.num_queries在Grounding DINO默认为900，同DETR
        # topk_indices = torch.topk(
        #     enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=(self.num_queries - self.hsm_queries), dim=1)[1]

        # TODO: hsm cos_similarity
        with torch.no_grad():
            output_memory_norm = torch.nn.functional.normalize(output_memory, p=2, dim=-1)
            bs_om, num_feat_points_om, c_om = output_memory_norm.shape
            # hard_sample_memory = self.hard_sample_memory.unsqueeze(0).repeat(output_memory.shape[0], 1, 1)
            # hsm_cos_sim.size = (bs * num_feat_points, self.hsm_prototype_num)
            # hsm_cos_sim = torch.nn.functional.cosine_similarity(output_memory_norm, hard_sample_memory, dim=-1)
            hsm_cos_sim = torch.mm(output_memory_norm.view(-1, c_om), self.hard_sample_memory.t())
            hsm_cos_sim = hsm_cos_sim.view(bs_om, num_feat_points_om, -1).reshape(bs_om, num_feat_points_om, -1)
            # each sample can only be similar to one hsm_prototype
            # hsm_cos_sim.size = (bs, num_feat_points)
            hsm_cos_sim = hsm_cos_sim.max(-1)[0]
            # get topk hsm indices
            hsm_cos_sim.scatter_(1, topk_indices, -100.0)
            # topk_hsm_indices.size=[bs, self.hsm_queries]
            topk_hsm_indices = torch.topk(
                hsm_cos_sim, k=self.hsm_queries, dim=-1)[1]

            # concat the topk_hsm_indices with the topk_indices and also remove the redundant indices
            topk_indices = torch.cat([topk_indices, topk_hsm_indices], 1)
            # self.hsm_prototype.size=[bs, num_queries, dim]
            self.hsm_prototype = torch.gather(output_memory_norm, 1, topk_indices.unsqueeze(-1).repeat(1, 1, c_om))
        # TODO: hsm end

        # 根据indices提取enc_outputs_class中的对应proposal置信度得分，topk_score.shape=(bs, self.num_queries, max_text_len)
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        # detach()使topk_coords_unact后续的使用不影响反向传播梯度
        topk_coords_unact = topk_coords_unact.detach()

        # 每个label的可训练标志向量 query.shape=(bs, self.num_queries, self.embed_dims)
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            # dn_label_query.shape=(bs, num_denoising_queries, dim),
            # dn_bbox_query.shape=(bs, num_denoising_queries, 4)
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            # reference_points.shape=(bs, num_denoising_queries+self.num_queries, dim)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        # 这里memory_text和text_token_mask只用于生成enc_outputs_class（计算两个模态的对比得分ContrastiveEmbed），未作改变
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    @torch.no_grad()
    def hsm_memory_update(self, ):
        '''
        update the self.hard_sample_memory based on the gt hard samples with the lowest cls_scores

        Returns:

        '''
        # 1. update the self.hard_sample_memory
        # first for loop -> bs
        with torch.no_grad():
            for bs, (hsm_indices_memory_bz, hsm_labels_bz, hsm_degree_bz) in enumerate(
                    zip(self.bbox_head.hsm_indices_memory,
                        self.bbox_head.hsm_labels,
                        self.bbox_head.hsm_degree)):
                # seen_hsm_prototype_bz.size=[num_queries, dim]
                seen_hsm_prototype_bz = self.hsm_prototype[bs]
                # second for loop -> seen_label
                # hsm_indices_memory_label.size=[num_query], hsm_labels_label.size=1
                for hsm_indices_memory_label, hsm_labels_label, hsm_degree_label in zip(hsm_indices_memory_bz,
                                                                                        hsm_labels_bz, hsm_degree_bz):
                    # seen_hsm_prototype.size=[dim], seen_hard_sample_memory.size=[self.hsm_v_dim]
                    seen_hsm_prototype = seen_hsm_prototype_bz[hsm_indices_memory_label].squeeze(0)
                    seen_hard_sample_memory = self.hard_sample_memory[hsm_labels_label]
                    cos_similarity = torch.nn.functional.cosine_similarity(seen_hard_sample_memory.unsqueeze(0),
                                                                           seen_hsm_prototype.unsqueeze(0),
                                                                           dim=-1)
                    cos_sim_grad = -seen_hsm_prototype + cos_similarity * seen_hard_sample_memory
                    self.hsm_velocity[hsm_labels_label] = self.hsm_velocity_momentum * self.hsm_velocity[
                        hsm_labels_label] + (1.0 - self.hsm_velocity_momentum) * cos_sim_grad

                    # update self.hard_sample_memory
                    self.hard_sample_memory[hsm_labels_label] += self.hsm_lr * hsm_degree_label * self.hsm_velocity[
                        hsm_labels_label]
                    self.hard_sample_memory[hsm_labels_label] = torch.nn.functional.normalize(
                        self.hard_sample_memory[hsm_labels_label], p=2, dim=-1)

            # 3. init hard sample prototypes and the bbox_head
            self.hsm_prototype = 0
            self.bbox_head.hsm_indices_memory = []
            self.bbox_head.hsm_labels = []
            self.bbox_head.hsm_degree = []

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # batch_data_samples一般是annotations
        # data_info['text'] = self.metainfo['classes']
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        # 见image_demo.py，一般属于推理阶段输入的参数，
        # 用于标志which locations in the input text是感兴趣区段(cls_name)，内容为[start,end]
        # input带有tokens_positive表示以及经过了输入规范性处理，可以直接分词
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                # 分词 （初始化self.language_model的时候已从hugging face下载了tokenizer预训练模型）
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                # tokenized是分词结果，
                # caption_string是转换为“前缀+cls_name+后缀+self._special_tokens”的prompts，
                # tokens_positive记录每个"name"的[start_i, end_i]
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                # 遍历batch size
                for gt_label in gt_labels:
                    # 遍历gt_label（也即每个gt_bbox的cls标签），以将训练标签label与text_prompts的位置对应上
                    # new_tokens_positive.shape=(gt_bbox_labels, 2)，将每个gt_bbox与token对应上，dim=1包含label-token信息
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # positive_map[i, j] = True, i表示gt的第i个gt_bbox，j为该gt_bbox对应的tokens_positive位置(start:end)
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                # 遍历batch size
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # 获取论文图Fig.3中vanilla text features
        # text_dict包含'embedded'，'masks'，'hidden'
        # embedded是所有embedding feature的stack；
        # mask是对应input token的有效范围。由于bert和transformer都是一一对应token输出的，所以是'hidden'的同shape的mask
        # 'hidden'=encoded_layers[-1]是最后一层输出embedding feature
        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        # batch size遍历gt_annotation，以准备好适合loss计算的annotations
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        # 对应论文图Fig.3中Feature Enhancer
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # TODO:hsm update
        self.hsm_memory_update()

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            # 见image_demo.py，一般属于推理阶段输入的参数，用于标志which locations in the input text是感兴趣区段，内容为[start,end]
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        # 分词预处理获取text prompts
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                                             self.get_tokens_positive_and_prompts(
                                                 text_prompts[0], custom_entities, enhanced_text_prompts[0],
                                                 tokens_positives[0])
                                         ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            # 逐个text_prompts处理，提取文本特征然后获取pred_instances
            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # 这里的label_names取决于输入的text prompt，也就是entities
        # 因此前面推理过程与prompt无关，仅仅只是在检测结果将label与label_names进行匹配
        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
