from sklearn.feature_extraction.text import CountVectorizer
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from wordcloud import WordCloud

from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
import pandas as pd
import json
import boto3
import re
import nltk
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from collections import Counter

nltk.download('punkt_tab')
nltk.download('punkt')


def compare_for_category(category_name, df1_name, df2_name, df1, df2):
    compare_by_category = {'vendor_id':[], 'name':[], df1_name:[], df2_name:[], 'similar':[]}
    similar = 0
    for i in range(min(len(df1), len(df2))):
        compare_by_category['vendor_id'].append(df1['vendor_id'].values[i])
        compare_by_category['name'].append(df1['name'].values[i])
        compare_by_category[df1_name].append(df1[category].values[i])
        compare_by_category[df2_name].append(df2[category].values[i])
        if compare_by_category[df1_name][-1] == compare_by_category[df2_name][-1] :
            compare_by_category['similar'].append(1)
        elif set(word_tokenize(str(compare_by_category[df1_name][-1]))) & set(word_tokenize(str(compare_by_category[df2_name][-1]))):
            compare_by_category['similar'].append(0.5)
        else:
            compare_by_category['similar'].append(0)
        if category_name in ['View','Terrace'] and  compare_by_category['similar'][-1] == 0:
            if not check_no_anwswer(compare_by_category[df1_name][-1]) and not check_no_anwswer(str(compare_by_category[df2_name][-1])):
                compare_by_category['similar'][-1] = 0.5
    return compare_by_category

def create_llm(region='eu-west-3', model_id='mistral.mistral-7b-instruct-v0:2', max_tokens=None):
    bedrock_client = boto3.client(service_name="bedrock-runtime",
                                  region_name=region)
    model_kwargs={"temperature": 0}
    if max_tokens:
        if 'meta' in model_id:
            model_kwargs["max_gen_len"] = max_tokens
        else:
            model_kwargs["max_tokens"] = max_tokens
    llm = BedrockLLM(client=bedrock_client, model_id=model_id, model_kwargs=model_kwargs)
    return llm

def format_dataframe_atena(df: pd.DataFrame) -> pd.DataFrame:
    features_to_keep = ['name', 'types', 'takeout', 'delivery', 
                        'vicinity', 'serves_beer', 'serves_wine', 'serves_lunch', 'weekday_text',
                        'serves_brunch', 'serves_dinner', 'serves_breakfast', 
                        'formatted_address', 'user_ratings_total', 'formatted_phone_number',
                        'wheelchair_accessible_entrance','url' , 'website','editorial_summary', 'reviews', 'rating']
    new_df_schema = {'vendor_id':[]} | {feat : [] for feat in features_to_keep}
    for i, row in df.iterrows():
        new_df_schema['vendor_id'].append(row['vendorid'])
        raw_json = json.loads(row['json']).get('result', {})
        rating = 0
        reviews = raw_json.get('reviews')
        if reviews:
            for review in reviews:
                rating += review['rating']
            rating = rating / len(reviews)
        raw_json['rating'] = rating if rating > 0 else None
        for feat in features_to_keep:
            new_df_schema[feat].append(raw_json.get(feat))
    return pd.DataFrame(new_df_schema)


# Function to generate bigrams with stopwords removed
def get_ngrams(texts, range_ngrams=(2, 2)):
    vectorizer = CountVectorizer(ngram_range=range_ngrams, stop_words='english')  # for bigrams
    X = vectorizer.fit_transform(texts)
    bigrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    bigram_freq = dict(zip(bigrams, counts))
    return bigram_freq

def show_wordcloud_of_ngrams(ngrams_freq,fig_name):
    # Create a word cloud of ngrams
    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(ngrams_freq)

    # Display the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    
    
    
def get_summary_and_reviews_from_vendor_id(formated_df, vendor_id):
    summary = formated_df[formated_df['vendor_id'] == vendor_id]['editorial_summary'].values[0]
    reviews = formated_df[formated_df['vendor_id'] == vendor_id]['reviews'].values[0]
    formated_string = ""
    if summary and summary.get('overview'):
        formated_string += f"Summary : {summary['overview']}\n\n\n\n"
    if reviews:
        formated_string += '\n\n\n\n'.join([f"Review {i+1} : {review['text']}" for i, review in enumerate(reviews)])
    return formated_string

def get_json_from_llm_output(llm_output):
    llm_output = "{"+llm_output.split('{')[1].split('}')[0]+"}"
    return json.loads(re.sub(r'\n+|\t+','',llm_output))

def check_no_anwswer(llm_answer):
    return True if 'Unknown' in str(llm_answer) or str(llm_answer) == '[]' or str(llm_answer) == '' else False


def check_json_output(enriched_content):
    print('Number of vendors treated : ',len(enriched_content))
    print('Number of vendors without reviews (enrichment is None) : ',sum([1 for v in enriched_content.values() if v is None]))    
    columns = []
    for enrichment in enriched_content.values():
        if enrichment :
            columns += list(enrichment.keys())
    print(Counter(columns))
    
    
def add_enriched_content_to_dataframe(augmented_content, mandatory_columns):
    enriched_df = {'vendor_id': []} | {col : [] for col in mandatory_columns} | {'other_infos': []}
    for vendor_id, enrichment in augmented_content.items():
        enriched_df['vendor_id'].append(vendor_id)
        # add mandatory columns
        for col in mandatory_columns:
            enriched_df[col].append(enrichment.get(col,None))
        # concatenate additional columns in other infos
        other_infos = {}
        for additional_column in list(set(enrichment.keys()) - set(mandatory_columns)):
            other_infos[additional_column] = enrichment[additional_column]
        enriched_df['other_infos'].append(other_infos)
    return pd.DataFrame(enriched_df)


def get_website_summary(llm, website_markdown):
    template = """Give me the main infos from this website page : {website_markdown}
                Summarized Information :
                """
    prompt = PromptTemplate(
        input_variables=[ "website_markdown"],
        template=template
    )
    prompt_input = prompt.invoke({#"json_vendor" : from_vendor_id_to_json(vendor_id, categories, mistral_generation_1000),
                   "website_markdown" : website_markdown
                  })
    output = llm.invoke(prompt_input)
    return output.strip()