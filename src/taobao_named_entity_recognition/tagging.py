#coding:utf-8

import sys
sys.path.append("../")
import os
import re
import chardet
import jieba
import jieba.posseg as pseg

color_dict = {}
product_dict = {}
brand_dict = {}
location_dict = {}
material_dict = {}
style_dict = {}
sex_dict = {}
effect_dict = {}

def read_color_dict():
    with open("../../data/entity/color_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            color_dict[line] = 1
    return

def read_product_dict():
    with open("../../data/entity/product_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            product_dict[line] = 1
    return

def read_brand_dict():
    with open("../../data/entity/brand_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            brand_dict[line.lower()] = 1
    return

def read_location_dict():
    with open("../../data/entity/location_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            location_dict[line] = 1
    return

def read_material_dict():
    with open("../../data/entity/material_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            material_dict[line] = 1
    return

def read_style_dict():
    with open("../../data/entity/style_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            style_dict[line] = 1
    return

def read_sex_dict():
    with open("../../data/entity/sex_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            sex_dict[line] = 1
    return

def read_effect_dict():
    with open("../../data/entity/effect_ext.dict", "r") as f:
        for line in f:
            line = line.strip()
            effect_dict[line] = 1
    return



def tagging():
    output = open("./tagged.data", "w")
    with open("./train.data", "r") as f:
        for line in f:
            line = line.strip()
            ll = line.split(",")
            for i in range(len(ll)):
                term = ll[i]
                seg = pseg.cut(term)

                nominal = next(seg).flag
                tag = ""
                if product_dict.get(term, None):
                    tag = "P-product"
                elif brand_dict.get(term, None):
                    tag = "B-brand"
                elif location_dict.get(term, None):
                    tag = "L-location"
                elif material_dict.get(term, None):
                    tag = "M-material"
                elif style_dict.get(term, None):
                    tag = "ST-style"
                elif sex_dict.get(term, None):
                    tag = "S-sex"
                elif effect_dict.get(term, None):
                    tag = "E-effect"
                term = term.replace("\t", " ")
                term = term.replace(" ","")
                ll[i] = "%s\t%s\t%s" % (term, nominal, tag)

            output.write(",".join(ll) + "\n")
    output.close()

    return


def main():
    read_color_dict()
    read_product_dict()
    read_brand_dict()
    read_location_dict()
    read_material_dict()
    read_style_dict()
    read_sex_dict()
    read_effect_dict()
    tagging()
    return

if __name__ == "__main__":
    main()
