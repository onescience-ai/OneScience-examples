---
tags:
- climate
---
# Introduction

This repository contains a customized weight file for the [NowcastNet](https://codeocean.com/capsule/3935105/tree/v1) model, originally proposed in the study titled '[Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4)' by Yuchen Zhang, Mingsheng Long et al.

The customization was performed to enable the original `state_dict` to be loaded by my personal reimplementation of NowcastNet, available [here](https://github.com/VioletsOleander/nowcastnet-rewritten). The modifications solely involved remapping certain keys to match the updated architecture, while all weight values remained unchanged.