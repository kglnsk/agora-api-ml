from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Product
from .serializers import ProductSerializer
import json
import pandas as pd
import numpy as np
import scann
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi


def merge_name_and_properties(df):
    dataframe = df.copy()
    merged = []
    for item,row in df.iterrows():
        merged.append(("Название: "+ str(row['name'] + '; Характеристики товара:' + ', '.join(row['props']).replace("\\t"," "))).lower())
    dataframe['data_string'] = merged
    return dataframe
        
        

class ProductView(APIView):
    def get(self, request):
        products = Product.objects.all()
        serializer = ProductSerializer(products, many=True)
        return Response({"products": serializer.data})


class CreateView(APIView):
    

    def post(self, request):
        bm25_thresh = 10.0
        data=request.data #type : dict полученные данные от пользователя
        df = pd.read_json('agora_hack_products.json')
        df_new = df[df['is_reference']==True].copy()
        df_new = merge_name_and_properties(df_new)
        print(df_new.columns)
        tokenized_corpus = [doc.split(" ") for doc in list(df_new['data_string'].values)]
        bm25 = BM25Okapi(tokenized_corpus)
        ids = []
        for item in data:
            id_dict = {}
            merged_str = ("Название: "+ item['name'] + '; Характеристики товара:' + ', '.join(item['props']).replace("\\t"," ")).lower().split(" ")
            id_dict['id'] = item['product_id']
            scores = bm25.get_scores(merged_str)
            if np.max(scores)>bm25_thresh:
                id_ref = np.argmax(scores)
                id_dict['reference_id'] = list(df_new['product_id'].values)[id_ref]
            else:
                id_dict['reference_id'] = 0
            ids.append(id_dict)
        print(ids)
        return Response((ids))

from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
def clean(request):
    pr = Product.objects.all()
    pr.delete()
    return HttpResponseRedirect("/list")

def main(request):
    return render(request,'main.html')
