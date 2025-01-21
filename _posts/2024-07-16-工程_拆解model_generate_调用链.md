---
layout:     post
title:      Uniform Sequence Better:Time Interval Aware Data Augmentation for Sequential Recommendation
subtitle:   From data perspective
date:       2014-07-02
author:     BY
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - Blog
---

> TL;DR：keep the uniform and augment the non-uniform to uniform, while don't break the mutation at the junction of uniform and non-uniform when substituion.
>

> reason：time interval is an import feature. how to integrate it into generative model is under exploration and hope I can get some inspiration from this traditional recommend paper.
>
> basic：basic idea is that uniform sequences are more valuable for next-item prediction. This assumption was validated by an empirical study. Then, proposed five data operators to augment item sequences in the light of time intervals verified the effectiveness.
>
> explain-in-my-way：1.do experiment to show uniform time interval matters（rank the data by std and seperate into two parts by sample-dim or history_length-dim） → 2.modify traditional data augmentations by considering time interval to keep as uniform as possible.（only σ% non-uniform data to be DA）. besides, these DA will be different on samples of different length to keep the import features under DA.
>
> inspiration：take a sequence into three parts A/B/C，A is an uniform part, C is a non-uniform part and B is the span bewteen A and C or B and C is indivually uniform part in non-uniform one?

### Approach

[![](http://upload-images.jianshu.io/upload_images/2178672-51a2fe6fbe24d1cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)](http://qiubaiying.github.io/)


| operation | method | detail | effect |
| -- | -- | -- | -- |
| Nothing | x | x | x |
| Ti-Insert | x | 1.v_i is an item that has not been interacted with by the user but is correlated with contextual items in the sequence, to be inserted to the right position such that the non-uniform sub-sequence becomes more uniform. <br> 2. T_u is all time intervals like [3, 2, 1, 1, 1, 0.75, 0.5] | x |
| Ti-Crop | x | 1.crop at p from which sub-sequence of length c has minimum std. <br> 2. | x |
| Ti-Mask | x | 1.mask at the positon of the minimum interval, make the std changes little. | x |
| Ti-Substitute | x | 1.replace the original items with fake yet correlated items <br> 2.obtain new sequences by imposing the minimum changes (via substitution) to the original ones. | x |
| Ti-Reorder | x | 1.creates a new sequence by shuffling the positions of items in a sub-sequence <br> 2.select the sub-sequence with minimum standard deviation, it has a relatively high chance to preserve similar preference pattern after data reordering. | x |
| DA by length | x | x | x |
