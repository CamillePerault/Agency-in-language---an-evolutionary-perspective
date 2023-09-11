import spacy
import json
import re
import regex as re
import unicodedata
from collections import defaultdict
import os 
import pandas as pd
import torch
import unidecode
import string
import math 
import time 
import functools
from pie_extended.cli.utils import get_tagger, get_model, download
from spacy.language import Language
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

nlp = spacy.load('fr_dep_news_trf') 

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
 
################## FORMATAGE EN VUE DE L'EXTRACTION DES ATTRIBUTS ET REPLIQUES DES PERSONNAGES DE CHAQUE PIECE ##################
@timer
def extract_character_status_from_didascaly(play) : 
    # Initialize an empty dictionary to store the characters' name and their social status
    characters_name = []
    characters_status = {}

    # Define the keywords that will unilateraly assign a low social status to the corresponding character
    low_keywords = ['domestique', 'esclave', 'chambre', 'camariste', 'servant', 'servante', 'suivante', 'valet', 'suite', 'confident', 'confidente'] #confident: equivalent to servant in tragedy (although more secundary role ->we want to capture the diff tragedy/comedy with play genre as indep variable)

    # Find the didascaly of the play by searching for the mention of "ACTEURS" or "PERSONNAGES" (and which should end with the mention of "ACTE" or "SCENE")
    didascaly_start = re.search(r"(^|\n)(\s*)(ACTEURS|PERSONNAGES)", play)
    didascaly_end = re.search(r"(^|\n)(\s*)(ACTE|SC[EÈÉÊ]NE)\s", play, flags=re.ASCII) #whitespace needed so as not to confound it with ACTE-URS

    # If a match was found, extract the didascaly text (and define "text" as what follows the initial didascaly)
    if didascaly_start and didascaly_end:
        didascaly = play[didascaly_start.start():didascaly_end.start()]
        text = play[didascaly_end.start():]
    
        # Split the didascaly into a list of lines  
        didascalylines = didascaly.split('\n')

        # Iterate over each line in the didascaly (which corresponds to a specific character)
        for i, line in enumerate(didascalylines):
            line = line.lstrip()
            doc = nlp(line)
            # Ignore the line that signals the beginning of the didascaly
            if "ACTEURS" in line or "PERSONNAGES" in line:
                continue
            name = None
            for token in doc: 
                # If the first word of the line is all in uppercase, it corresponds to the description of a character (otherwise, skip)
                if token.i == 0 : 
                    if not token.text.isupper():
                        continue
                    if "," in doc.text: 
                        # If the line includes any of the following punctuation, extract the characters' name as the text up to the punctuation mark
                        name = doc.text.split(",")[0].upper()
                    elif ";" in doc.text : 
                        name = doc.text.split(";")[0].upper()
                    elif ":" in doc.text : 
                        name = doc.text.split(":")[0].upper()
                    else:
                        # If the line does not include any of the previous punctuation, the character's name is all the words written all in uppercase
                        name_tokens = [token.text]
                        for next_token in doc[token.i+1:]:
                            if next_token.text.isupper(): 
                                name_tokens.append(next_token.text)
                        # Add the name to the list
                        name = " ".join(name_tokens).upper()
                if name:     
                    # Remove any space at the beginning or end of the name
                    name = name.strip()
                    characters_status[name] = None
                    # For the rest of the line describing a given character, attribute a low social status if it contains one of the above keywords
                    if token.text.lower() in low_keywords: 
                        characters_status[name] = 'low'
                        break 
        characters_name = list(characters_status.keys())
        # To attribute high status, search for a name complement, after a low keyword (for instance, "valet de [name of another character, to which we should hence assign a high social status]")
        for line in didascalylines:
            doc = nlp(line)
            for i, token in enumerate(doc):
                if token.text.lower() in low_keywords: 
                    name_tokens = []
                    found_de = False
                    for j in range(i + 1, len(doc)):
                        if doc[j].text.lower() in ["de", "d'", "du"]:
                            found_de = True
                        if found_de :
                            if doc[j].text[0].isupper():
                                name_tokens.append(doc[j].text)
                            elif doc[j].text in [",", ";", ":", "."]: 
                                break
                    name_mention_upper = " ".join(name_tokens).upper()
                    exact_match_found = False
                    if name_mention_upper in characters_name :
                        characters_status[name_mention_upper] = "high"
                        exact_match_found = True
                        break 
                    # If the name only partially matches one of the name stored in the initial list, search for substring match 
                    if not exact_match_found: 
                        for name in characters_name: 
                            name_words = name.split()
                            mention_words = name_mention_upper.split()
                            if len(mention_words) > 1 and (name in mention_words) :
                                characters_status[name] = "high"
                            elif len(name_words) > 1 and (name_mention_upper in name_words):
                                characters_status[name] = "high"
                            elif len(mention_words) > 1 and len(name_words) > 1 and (any(word in name_words for word in mention_words) or any(word in mention_words for word in name_words)) : 
                                characters_status[name] = "high"
                            elif unidecode.unidecode(name_mention_upper) == unidecode.unidecode(name):
                                characters_status[name] = "high"
                                break  

        # Suppress characters who still have a None status
        characters_status = {k: v for k, v in characters_status.items() if v is not None}
        print(characters_status)
    
    else: 
        text = play
        print('no didascaly found')

    return text, characters_name, characters_status

@timer
def assign_gender_to_character(file_name): 
    # Load the Excel file containing the gender mentions into a pandas DataFrame
    df = pd.read_excel("French Theater Corpus annotation - Copy.xlsb - Copy.xlsx")
    # Convert the DataFrame to a dictionary
    filtered_df = df[df[df.columns[7]] == file_name]
    result = filtered_df.set_index(filtered_df.columns[0]).to_dict()[filtered_df.columns[1]]
    # Assign the gender mentioned to the corresponding character in the dictionary
    characters_gender = {k: v for k, v in result.items() if (v == "Masculin" or v == "Féminin")}
    return characters_gender

def filtered_text_in_lines(text): 
    # Split the text into lines
    textlines = text.split('\n')
    # Initialize a list which will contain the text without mentions of scene or act changes
    filtered_textlines = []
    scene_textlines = []
    # Suppress the lines that simply indicate a change of scene or act
    for line in textlines:
        if line.startswith("SCÈNE") or line.startswith("SCENE") or line.startswith("SCÉNE") or line.startswith("SCÊNE") or line.startswith("ACTE"): 
            scene_textlines.append(line)
            continue
        filtered_textlines.append(line)
    return filtered_textlines, scene_textlines

@timer
def attribute_replica_to_character(filtered_textlines, characters_name): 
    # Initialize an empty dictionary to store the characters' lines 
    characters_lines = {}
    characters_interventions = {}
    # Initialize a variable to store the character currently speaking
    current_character = None
    for line in filtered_textlines:
        # Ignore empty lines
        if line.strip() != '':
            first_word = line.split()[0]
            if len(first_word) > 1 and first_word.isupper(): 
                # Store as the name of the character speaking the corresponding uppercase string
                name_match = re.search(r"\b[A-ZÀ-ÖØ-Ý']+(?:\s+[A-ZÀ-ÖØ-Ý']+)*\b", line)
                if name_match is not None:
                    # If there is a match, strips the string of any punctuation
                    full_name = name_match.group(0).strip('.,:;!?')
                    # Check if the name matches any of the names in the character names list                    
                    if full_name in characters_name:
                        # Update the current character
                        current_character = full_name
                        # If the character is not in the dictionary yet, add it
                        if full_name not in characters_lines:
                            characters_lines[full_name] = []
                            characters_interventions[full_name] = 0
                        characters_interventions[current_character] += 1
                    else:
                        #histoire de longueur (?)
                        # If there is no exact match, check if any part of the full name is in the character names list
                        for name in characters_name:
                            if any(part.upper() in name.upper().split() for part in full_name.split()):
                                # Update the current character
                                current_character = name
                                # If the character is not in the dictionary yet, add it
                                if name not in characters_lines:
                                    characters_lines[name] = []
                                    characters_interventions[name] = 0
                                characters_interventions[current_character] += 1
                                break
                else:
                    # If none of the names match, set the current character to None
                    current_character = None
            
            # If the current character is not None, and the directly preceding line is not the character name (otherwise this is a line defining a word)
            if current_character is not None : 
                characters_lines[current_character].append(line)
    return characters_lines  

def narrative_importance_words_spoken(characters_lines, scene_textlines, characters_name):
    replica_length = {}
    for name, lines in characters_lines.items():
        words_spoken = sum(len(line.split()) for line in lines)
        replica_length[name] = words_spoken
    
    if not replica_length:
        return None
    else: 
        most_talkative_character = max(replica_length, key=replica_length.get)
        return most_talkative_character

@timer
def narrative_importance_order_didascaly(characters_name): 
    order_weights = {}
    for i, name in enumerate(characters_name):
        weight = 1 / (i + 1)
        order_weights[name] = weight
    return order_weights

@timer
def find_genre_of_the_play(file_name): 
    # Load the Excel file containing the gender mentions into a pandas DataFrame
    df = pd.read_excel("French Theater Corpus annotation - Copy.xlsb - Copy.xlsx")
    # Convert the DataFrame to a dictionary
    filtered_df = df[df[df.columns[7]] == file_name]
    result = filtered_df.set_index(filtered_df.columns[7]).to_dict()[filtered_df.columns[8]]
    play_genre = result[file_name]
    return play_genre

@timer
def attribute_replica_to_gender(characters_lines, characters_gender): 
    # Initialize an empty dictionary to store the characters' lines 
    words_by_gender = {'Masculin': [], 'Féminin': []}
    # Iterate through the characters in the first dictionary containing characters' name and status
    for name, gender in characters_gender.items():
        # If the character doesn't have any replica, pass
        if name not in characters_lines:
            continue
        # Get the lines of dialogue spoken by the character
        replicas = characters_lines[name]
        # Iterate through the lines of dialogue
        for replica in replicas:
            # Split the line into a list of words
            words = replica.split()
            # Add the words to the list in the dictionary for the corresponding social status
            words_by_gender[gender].extend(words)
    
    #Suppress  characters' name in all uppercase (which correspond to a didascaly, not spoken text; although we will still have the rest of the didascaly, following  the character's name, when there is one, but it is rather rare)
    for key, value in words_by_gender.items():
        # Remove all words written entirely in uppercase and with more than 1 letter from the value
        value = [word for word in value if not (word.isupper() and len(word) > 1)]
        # Update the value in the dictionary
        words_by_gender[key] = value

    #Create separated lists from the 2 keys of the status dictionary (high and low)
    male_words = words_by_gender['Masculin']
    female_words = words_by_gender['Féminin']

    # Concatenate the list of strings into a single string
    male_words_concat = ' '.join(male_words)
    female_words_concat = ' '.join(female_words)

    if not male_words_concat or not female_words_concat: 
        return None
    return male_words_concat, female_words_concat

@timer
def attribute_replica_to_status(characters_lines, characters_status): 
    # Initialize a dictionary that will contain all the words spoken by high and low social status characters respectively
    words_by_status = {'high': [], 'low': []}

    # Iterate through the characters in the first dictionary containing characters' name and status
    for name, status in characters_status.items():
        # If the character doesn't have any replica, pass
        if name not in characters_lines:
            continue
        # Get the lines of dialogue spoken by the character
        replicas = characters_lines[name]
        # Iterate through the lines of dialogue
        for replica in replicas:
            # Split the line into a list of words
            words = replica.split()
            # Add the words to the list in the dictionary for the corresponding social status
            words_by_status[status].extend(words)

    #Suppress  characters' name in all uppercase (which correspond to a didascaly, not spoken text; although we will still have the rest of the didascaly, following  the character's name, when there is one, but it is rather rare)
    for key, value in words_by_status.items():
        # Remove all words written entirely in uppercase and with more than 1 letter from the value
        value = [word for word in value if not (word.isupper() and len(word) > 1)]
        # Update the value in the dictionary
        words_by_status[key] = value
    
    #Create separated lists from the 2 keys of the status dictionary (high and low)
    high_status_words = words_by_status['high']
    low_status_words = words_by_status['low']

    # Concatenate the list of strings into a single string
    high_status_words_concat = ' '.join(high_status_words)
    low_status_words_concat = ' '.join(low_status_words)

    if not high_status_words_concat or not low_status_words_concat:  
        return None
    return high_status_words_concat, low_status_words_concat

@timer
def split_sentences(text):
    sentence_endings = r"\.|\?|\!|\…"
    sentence_regex = fr"{sentence_endings}(?:[^\S\n]*[{string.punctuation}]+)?"
    sentences = re.split(sentence_regex, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

@timer
def tag_sentences(concatenated_content): 
    tagger = get_tagger("fr", batch_size=256, device="cpu", model_path=None)
    from pie_extended.models.fr.imports import get_iterator_and_processor
    iterator, processor = get_iterator_and_processor()
    concatenated_content = concatenated_content.replace(","," ,").replace("."," .").replace("(","( ").replace(")", " )")
    sentences = split_sentences(concatenated_content)
    doc = list(nlp.pipe(sentences))

    spacy_tags = []
    pie_tags = []

    for i, sent in enumerate(doc): 
        spacy_tags.append([])
        pie_tags.append([])
        for token in sent:
            children = [{"child_text": child.text, "child_pos": child.pos_, "child_dep": child.dep_} for child in token.children]
            spacy_tags[i].append({
                "text": token.text,
                "lemma": token.lemma_,
                "POS": token.pos_,
                "dep": token.dep_,
                "head_text": token.head.text,
                "head_pos": token.head.pos_,
                "head_dep": token.head.dep_,
                "children": children
            })
        tagged_sent = tagger.tag_str(sent.text, iterator=iterator, processor=processor)

        for token in tagged_sent:
            pie_tags[i].append({
                "form": token["form"],
                "lemma": token["lemma"],
                "POS": token["POS"],
                "morph": token["morph"],
                "treated": token["treated"]
            })
    return spacy_tags, pie_tags

################## DEFINITION DES DIFFERENTES METRIQUES ##################
## III. 1.2. ADVERBES INTENTIONNELS
@timer
def intentional_adverbs(concatenated_content, word_count): 
    intentional_adverbs = ["volontairement", "délibérément", "expressément", "intentionnellement", "sciemment", "à dessein", "exprès"] 
    matches = {}
    for adverb in intentional_adverbs:
        pattern = r"\b" + re.escape(adverb) + r"[.,\s]"
        matches[adverb] = [(match.start(), match.end()) for match in re.finditer(pattern, concatenated_content, re.IGNORECASE)]
    intentional_adverbs_dict = defaultdict(int)
    for adverb, positions in matches.items():
        for start, end in positions:
            matched = False
            for other_adverb, other_positions in matches.items():
                if adverb != other_adverb and any(start <= other_start < end for other_start, other_end in other_positions):
                    matched = True
                    break
            if not matched:
                intentional_adverbs_dict[adverb] += 1 
    total_intentional_count = sum(intentional_adverbs_dict.values())
    if word_count == 0: 
        return None
    else: 
        return total_intentional_count/word_count

## III. 1.3. MODALITE
@timer
def semantic_modality_vs(spacy_tag_content): 
    internal_modality_verbs = ["pouvoir", "vouloir"]
    external_modality_verbs = ["devoir", "falloir"] 
    internal_modality_count = 0
    external_modality_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in internal_modality_verbs: 
                internal_modality_count += 1
            elif token['lemma'] in external_modality_verbs: 
                external_modality_count += 1
    if external_modality_count == 0: 
        return None
    else: 
        return internal_modality_count/external_modality_count

@timer
def modal_verbs(spacy_tag_content, verb_count_pie): 
    modal_verbs = ["pouvoir", "devoir", "vouloir", "falloir"] 
    modal_verb_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in modal_verbs: 
                for child in token['children']: 
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp' :
                        modal_verb_count += 1
    if verb_count_pie == 0: 
        return None
    else: 
        return modal_verb_count/verb_count_pie

@timer
def modal_verbs_extended(spacy_tag_content, verb_count_pie): 
    extended_modal_verbs = ["pouvoir", "devoir", "vouloir", "falloir", "espérer", "savoir", "penser", "aller"] 
    modal_verb_extended_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in extended_modal_verbs: 
                for child in token['children']: 
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp': 
                        modal_verb_extended_count += 1
    if verb_count_pie == 0: 
        return None
    else: 
        return modal_verb_extended_count/verb_count_pie

@timer
def internal_vs_external_modals(spacy_tag_content): 
    internal_modals = ["pouvoir", "vouloir"] 
    external_modals = ["devoir", "falloir"] 
    internal_modal_verb_count = 0
    external_modal_verb_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in internal_modals:
                for child in token['children']: 
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp': 
                        internal_modal_verb_count += 1
            elif token['lemma'] in external_modals: 
                for child in token['children']: 
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp': 
                        external_modal_verb_count += 1
    if external_modal_verb_count == 0: 
        return 0
    else: 
        return internal_modal_verb_count/external_modal_verb_count

### AUTRES PROXY MODAUX
@timer
def affirmation_adverbs(concatenated_content, word_count): 
    affirmation_adverbs = ["à vrai dire", "assurément", 
    "à coup sûr",
    "absolument",
    "bien sûr",
    "pour sûr",
    "sans conteste",
    "sans contredit",
    "bien entendu",
    "effectivement", 
    "en effet",
    "exactement",
    "parfaitement",
    "clairement",
    "certainement",
    "évidemment",
    "fatalement",
    "forcément",
    "immanquablement",
    "indubitablement",
    "inéluctablement",
    "inévitablement",
    "infailliblement",
    "sûrement",
    "certes", 
    "en vérité", 
    "oui", 
    "précisément", 
    "sans doute", 
    "nul doute",
    "sans aucun doute",
    "sans le moindre doute",
    "si fait", 
    "tout à fait",
    "si.", "Si,", "Si :", "Si !", 
    "soit.", "soit,", "soit :", "soit !",
    "volontiers", 
    "vraiment", 
    "naturellement", 
    "vraisemblablement",
    "d'accord"
    ]

    matches = {}
    for adverb in affirmation_adverbs:
        pattern = r"\b" + re.escape(adverb) + r"[.,\s]"
        matches[adverb] = [(match.start(), match.end()) for match in re.finditer(pattern, concatenated_content, re.IGNORECASE)]
    affirmation_adverbs_dict = defaultdict(int)
    for adverb, positions in matches.items():
        for start, end in positions:
            matched = False
            for other_adverb, other_positions in matches.items():
                if adverb != other_adverb and any(start <= other_start < end for other_start, other_end in other_positions):
                    matched = True
                    break
            if not matched:
                affirmation_adverbs_dict[adverb] += 1 
    total_affirmation_count = sum(affirmation_adverbs_dict.values())
    if word_count == 0: 
        return None
    else: 
        return total_affirmation_count/word_count

@timer
def doubt_adverbs(concatenated_content, word_count):
    doubt_adverbs = ["apparemment", "peut-être", "peut être", "probablement", 
    "vraisemblablement", 
    "éventuellement",
    "supposément",
    "présumément",
    "possiblement",
    "potentiellement",
    "hypothétiquement",
    "a priori", 
    "sans doute", 
    "si jamais",
    "toutefois", 
    "certainement" 
    ] 

    matches = {}
    for adverb in doubt_adverbs:
        pattern = r"\b" + re.escape(adverb) + r"[.,\s]"
        matches[adverb] = [(match.start(), match.end()) for match in re.finditer(pattern, concatenated_content, re.IGNORECASE)]
    doubt_adverbs_dict = defaultdict(int)
    for adverb, positions in matches.items():
        for start, end in positions:
            matched = False
            for other_adverb, other_positions in matches.items():
                if adverb != other_adverb and any(start <= other_start < end for other_start, other_end in other_positions):
                    matched = True
                    break
            if not matched:
                doubt_adverbs_dict[adverb] += 1 
    total_doubt_count = sum(doubt_adverbs_dict.values())
    if word_count == 0: 
        return None
    else: 
        return total_doubt_count/word_count

@timer
def pronouns(spacy_tag_content, word_count): 
    pronoun_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "PRON": 
                pronoun_count += 1
    if word_count == 0: 
        return None
    else: 
        pronoun_normalized = pronoun_count/word_count
        return pronoun_normalized

@timer
def definite_vs_undefinite_articles(pie_tag_content): 
    definite_article_count = 0
    undefinite_article_count = 0
    for sentence in pie_tag_content:
        for token in sentence: 
            if token['POS'] == "DETdef" or token['POS'] == "PRE.DETdef": 
                definite_article_count += 1
            elif token['POS'] == "DETndf": 
                undefinite_article_count += 1
    if undefinite_article_count == 0: 
        return None 
    else: 
        return definite_article_count/undefinite_article_count

@timer
def progressive(spacy_tag_content, word_count): 
    progressive_count = 0
    for sent in spacy_tag_content: 
        for j, token in enumerate(sent):
            if j+2 < len(sent) and sent[j]['lemma'] == "en" and sent[j+1]['lemma'] == "train" and sent[j+2]['lemma'] == "de" : 
                progressive_count += 1
    if word_count == 0: 
        return None
    else: 
        return progressive_count/word_count

@timer
def ellipsis(concatenated_content, punctuation_count):
    ellipsis_count = concatenated_content.count('...')
    if punctuation_count == 0: 
        return None
    else: 
        return ellipsis_count/punctuation_count

@timer
def future(pie_tag_content, conjugated_verb_count): 
    future_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if "TEMPS=fut" in token['morph']:
                future_count += 1
    if conjugated_verb_count == 0: 
        return None
    else: 
        return future_count/conjugated_verb_count

## III. 2.1. VERBES
@timer
def verbs_pie(pie_tag_content, word_count): 
    verb_count_pie = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if token['POS'] in ["VERcjg", "VERinf", "VERppa"]: 
                verb_count_pie += 1
    if word_count == 0: 
        return verb_count_pie, None
    else: 
        verb_pie_normalized = verb_count_pie/word_count
        return verb_count_pie, verb_pie_normalized

@timer
def conjugated_verbs(pie_tag_content, verb_count_pie): 
    conjugated_verb_count = 0
    for sentence in pie_tag_content:
        for token in sentence: 
            if token['POS'] == "VERcjg":
                conjugated_verb_count += 1
    if verb_count_pie == 0: 
        return conjugated_verb_count, None
    else: 
        conjugated_verb_normalized = conjugated_verb_count/verb_count_pie 
        return conjugated_verb_count, conjugated_verb_normalized 

## III. 2.2. PRONOMS ET DEICTIQUES
@timer
def individual_vs_collective_pronouns(spacy_tag_content): 
    individual_pronoun_count = 0
    collective_pronoun_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "PRON": 
                if token['lemma'] in ['je', 'tu', 'me', 'te', 'moi', 'toi', 'mien', 'mienne', 'miens', 'miennes', 'tien', 'tienne', 'tiens', 'tiennes']: 
                    individual_pronoun_count += 1
                elif token['lemma'] in ['nous', 'vous', 'nôtre', 'vôtre', 'notre', 'votre', 'nos', 'vos']:
                    collective_pronoun_count += 1
    if collective_pronoun_count == 0:
        return 0
    else:
        return individual_pronoun_count/collective_pronoun_count

@timer
def first_person_pronouns(spacy_tag_content, word_count): 
    first_person_pronouns_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "PRON":
                if token['lemma'] in ['je', 'me', 'moi', 'soi', 'mien', 'mienne', 'miens', 'miennes']: 
                    first_person_pronouns_count += 1
    if word_count == 0: 
        return None
    else: 
        return first_person_pronouns_count/word_count

@timer
def first_person_pronouns_on_pronouns(spacy_tag_content): 
    pronoun_count = 0
    first_person_pronouns_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "PRON":
                pronoun_count += 1
                if token['lemma'] in ['je', 'me', 'moi', 'soi', 'mien', 'mienne', 'miens', 'miennes']: 
                    first_person_pronouns_count += 1
    if pronoun_count == 0: 
        return None
    else: 
        return first_person_pronouns_count/pronoun_count


@timer
def deictics_total(pronoun_normalized, possessive_normalized, temporal_adverb_normalized, locative_adverb_normalized, definite_article_normalized, demonstrative_normalized, exclamation_normalized, interjection_normalized):
    all_deictic_count = 0
    if pronoun_normalized is not None:
        all_deictic_count += pronoun_normalized
    if possessive_normalized is not None:
        all_deictic_count += possessive_normalized
    if temporal_adverb_normalized is not None:
        all_deictic_count += temporal_adverb_normalized
    if locative_adverb_normalized is not None:
        all_deictic_count += locative_adverb_normalized
    if definite_article_normalized is not None:
        all_deictic_count += definite_article_normalized
    if demonstrative_normalized is not None: 
        all_deictic_count += demonstrative_normalized
    if exclamation_normalized is not None: 
        all_deictic_count += exclamation_normalized
    if interjection_normalized is not None: 
        all_deictic_count += interjection_normalized
    if all_deictic_count == 0:
        return None
    else:
        return all_deictic_count

@timer
def deictics_conservative(pronoun_normalized, possessive_normalized, temporal_adverb_normalized, locative_adverb_normalized): 
    conservative_deictic_count = 0
    if pronoun_normalized is not None:
        conservative_deictic_count += pronoun_normalized
    if possessive_normalized is not None:
        conservative_deictic_count += possessive_normalized
    if temporal_adverb_normalized is not None:
        conservative_deictic_count += temporal_adverb_normalized
    if locative_adverb_normalized is not None:
        conservative_deictic_count += locative_adverb_normalized
    if conservative_deictic_count == 0:
        return None
    else:
        return conservative_deictic_count

@timer
def proximal_distal_deictics(spacy_tag_content): 
    proximal_deictics = ['ceci', 'voici', '-ci', 'ici', 'deçà', '-deçà', 'maintenant'] 
    distal_deictics = ['cela', 'voilà', '-là', 'là', 'delà', '-delà', 'alors'] 
    proximal_deictics_count = 0
    distal_deictics_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in proximal_deictics: 
                proximal_deictics_count += 1
            elif token['lemma'] in distal_deictics: 
                distal_deictics_count += 1
    if distal_deictics_count == 0: 
        return 0
    else: 
        return proximal_deictics_count/distal_deictics_count

## III. 2.3. DEPENDANCES SYNTAXIQUES
# AGENTIVITE GRAMMATICALE
@timer
def grammatical_agency(spacy_tag_content, sentence_count): 
    agent_count = 0
    for sent in spacy_tag_content: 
        for token in sent:
            if token['dep'] == "obl:agent" and token['text'].lower() in ['moi', 'ma', 'mon', 'mes']: 
                agent_count += 1
    if sentence_count == 0: 
        return None
    else: 
        return agent_count/sentence_count 

# VOIX GRAMMATICALES
@timer
def active_vs_passive_voice(spacy_tag_content, sentence_count): 
    passive_voice_count = 0
    active_voice_count = 0
    counted_sentences = set()
    for i in range(len(spacy_tag_content)):
        sent = spacy_tag_content[i]
        if i not in counted_sentences:
            for token in sent:
                if token['dep'] == "nsubj:pass" or token['dep'] == "aux:pass": 
                    passive_voice_count += 1
                    counted_sentences.add(i)
                    break
    active_voice_count = sentence_count - passive_voice_count
    if passive_voice_count == 0: 
        return None
    else: 
        return active_voice_count/passive_voice_count

# VERBES STATIFS
@timer
def stative_verbs(pie_tag_content, word_count): 
    stative_count = 0
    for sentence in pie_tag_content: 
        for i, token in enumerate(sentence): 
            if sentence[i]['lemma'] in ["être", "avoir"]:
                past_tense = False  
                for j in range(i+1, len(sentence)): 
                    if sentence[j]['POS'].startswith('PON'): 
                        break
                    elif sentence[j]['POS'] == 'VERppe': 
                        past_tense = True
                if not past_tense : 
                    stative_count += 1
    if word_count == 0: 
        return None
    else: 
        stative_normalized = stative_count/word_count
        return stative_normalized

@timer
def stative_on_verbs(pie_tag_content, verb_count_pie): 
    stative_count = 0
    for sentence in pie_tag_content: 
        for i, token in enumerate(sentence): 
            if sentence[i]['lemma'] in ["être", "avoir"]:
                past_tense = False  
                for j in range(i+1, len(sentence)): 
                    if sentence[j]['POS'].startswith('PON'): 
                        break
                    elif sentence[j]['POS'] == 'VERppe': 
                        past_tense = True
                if not past_tense : 
                    stative_count += 1
    if verb_count_pie == 0: 
        return None
    else: 
        return stative_count/verb_count_pie

@timer
def linking_verbs(spacy_tag_content, verb_count_pie):
    linking_verb_count = 0
    for sent in spacy_tag_content: 
        for token in sent:
            if token['dep'] == "cop":
                linking_verb_count += 1
    if verb_count_pie == 0: 
        return None
    else: 
        return linking_verb_count/verb_count_pie

# TRANSITIVITE
@timer
def transitive(spacy_tag_content, verb_count_pie): 
    transitive_count = 0
    counted_verbs = set() 
    for sent_idx, sent in enumerate(spacy_tag_content):
        for token_idx, token in enumerate(sent):
            if token['POS'] == "VERB" and (token['text'], sent_idx, token_idx) not in counted_verbs: 
                has_obj = False
                has_conj = False 
                for child in token['children']:
                    if child['child_dep'] in ["obj", "iobj"]:
                        has_obj = True
                        break
                if has_obj:
                    transitive_count += 1
                    counted_verbs.add((token['text'], sent_idx, token_idx))

    if verb_count_pie == 0: 
        return None
    else: 
        return transitive_count/verb_count_pie

## III. 3.1. TELICITE
@timer
def telicity(spacy_tag_content, pie_tag_content): 
    time_expressions = ["coucher", 
        "lever",
        "an", "ans", 
        "année",  "années",
        "soir", "soirs", 
        "soirée", "soirées", 
        "matin",  "matins", 
        "matinée", "matinées", 
        "jour",  "jours", 
        "nuit", "nuits", 
        "journée", "journées", 
        "midi",  
        "minuit", 
        "heure", "heures", 
        "minute",  "minutes", 
        "seconde", "secondes", 
        "mois",
        "janver", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "siècle", "siècles", 
        "saison",  "saisons", 
        "hiver", "printemps", "été", "automne", "hivers", "étés", "automnes", 
        "semaine", "semaines" ,
        "temps", 
        "moment", "moments", 
        "instant", "instants", 
        "fois"]
    
    temporal_adverbs = ["actuellement",
        "anciennement", 
        "simultanément",
        "fréquemment",
        "régulièrement",
        "antan", 
        "alors", 
        "auparavant", 
        "aussitôt", 
        "autrefois", 
        "avant", 
        "bientôt", 
        "déjà", 
        "demain",
        "d’main", 
        "depuis",
        "derechef", 
        "dernièrement", 
        "désormais", 
        "dorénavant", 
        "encore", 
        "enfin", 
        "ensuite", 
        "entre-temps", 
        "entretemps",
        "hier",
        "hui", 
        "illico", 
        "immédiatement", 
        "instantanément",
        "jadis", 
        "jamais", 
        "longtemps", 
        "lors", 
        "maintenant", 
        "maishui",  
        "meshui", 
        "méshui", 
        "momentanément",
        "naguère", 
        "naguères", 
        "parfois", 
        "présentement", 
        "prochainement",
        "puis", 
        "quelquefois", 
        "rarement",
        "récemment", 
        "sitôt", 
        "soudain", 
        "soudainement",
        "souvent", 
        "subito", 
        "sur-le-champ", 
        "tantôt", 
        "tard", 
        "tardivement", 
        "tôt", 
        "toujours", 
        "tandis",
        "dès", 
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", 
        "quand", 
        "lorsque", 
        "piéça"]
    
    locative_adverbs = ["aí", 
        "ailleurs", 
        "alentour", "alentours",
        "arrière", 
        "autour", 
        "çà", 
        "céans", 
        "chez", 
        "ci",  
        "deçà",  
        "delà", 
        "exa", 
        "hái", 
        "ici", 
        "là",
        "léans", 
        "loin", 
        "où", 
        "partout", 
        "près", 
        "sus", 
        "y", 
        ]
    manner_adverbs = ["comment", 
        "ainsi", 
        "aussi", 
        "bien", 
        "debout", 
        "également", 
        "ensemble", 
        "franco", 
        "gratis", 
        "incognito", 
        "mal", 
        "mieux",
        "pis", 
        "plutôt", 
        "presque",
        "quasi", 
        "recta", 
        "vite", 
        "volontiers"]
    
    total_telicity_score = 0 
    sentence_count = 0
    for i, (sent, pie_sent) in enumerate(zip(spacy_tag_content, pie_tag_content)):
        has_event_verb = False 
        for j, spacy_token in enumerate(sent):
            if spacy_token['POS'] == 'VERB': 
                has_event_verb = True 
        if not has_event_verb: 
            continue  
            ##AUTOMATIC TELICITY 
        else: 
            sentence_count += 1 
            telicity = None 
            telicity_intentional = None 
            telicity_score_sentence = 0 
            normalized_scores_sentence = []
            for j, spacy_token in enumerate(sent):
                if sent[j]['lemma'] ==  "en": 
                    for k in range(j+1, len(sent)):
                        if sent[k]['POS'] == "PUNCT": 
                            break 
                        elif sent[k]['POS'] == "NUM": 
                            telicity = True 
                            break 
                elif sent[j]['lemma'] == "pendant" and sent[j]['POS'] != "VERB": 
                    for k in range(j+1, len(sent)):
                        if sent[k]['POS'] == "PUNCT": 
                            break 
                        elif sent[k]['POS'] == "NUM": 
                            telicity = False 
                            break  
                elif spacy_token['lemma'] in ['volontairement', 'délibérément', "expressément", "exprès", "intentionnellement", "sciemment"] :  
                    telicity_intentional = True
                elif spacy_token['lemma'] == "pour" and spacy_token["POS"] == "ADP" and spacy_token['dep'] == "mark" : 
                    telicity_intentional = True
                elif spacy_token['lemma'] == "finalité": 
                    for child in spacy_token['children']: 
                        if child['child_text'] == "pour" and child['child_pos'] == "ADP" and child['child_dep'] == "case": 
                            telicity_intentional = True
                            break
                elif spacy_token['lemma'] == "afin" :  
                    telicity_intentional = True
                elif spacy_token['lemma'] == "dessein" : 
                    for child in spacy_token['children']: 
                        if child['child_text'].lower() == "à" and child['child_pos'] == "ADP" and child['child_dep'] == "case": 
                            telicity_intentional = True
                            break
                elif spacy_token['lemma'] in ["objectif", "but", "intention", "dessein"] :
                    has_dans = False
                    has_acl = False 
                    for child in spacy_token['children']: 
                        if child['child_text'].lower() == "dans" and child['child_pos'] == "ADP" and child['child_dep'] == "case": 
                            has_dans = True
                        elif child['child_dep'] in ["acl", "acl:relcl"]: 
                            has_acl = True 
                    if has_dans and has_acl: 
                        telicity_intentional = True
                elif spacy_token['text'] in ["façon", "manière"] : 
                    has_de = False 
                    has_acl = False
                    has_ce = False 
                    has_telle = False 
                    for child in spacy_token['children']: 
                        if child['child_text'].lower() == "de" and child['child_pos'] == "ADP" and child['child_dep'] == "case": 
                            has_de = True 
                        elif child['child_dep'] in ["acl", "acl:relcl", "ccomp"]: 
                            has_acl = True 
                        elif child['child_text'] == "ce" and child['child_pos'] == "PRON" or child['child_dep'] == "nmod":
                            has_ce = True 
                        elif child['child_text'] == "telle" and child['child_dep'] == "amod":
                            has_telle = True 
                    if ((has_de and has_acl) or (has_de and has_ce)) and not has_telle:  
                        telicity_intentional = True 
                elif spacy_token['lemma'] == "histoire" : 
                    for child in spacy_token['children']: 
                        if child['child_dep'] == "ccomp" : 
                            telicity_intentional = True 
                            break
                elif spacy_token['lemma'] == "fin" :
                    has_case = False
                    has_seule = False  
                    has_acl = False 
                    for child in spacy_token['children']: 
                        if child['child_text'].lower() == "à" and child['child_pos'] == "ADP" and child['child_dep'] == "case":
                            has_case = True
                        elif child['child_dep'] in ["acl", "acl:relcl"]: 
                            has_acl = True
                        elif child['child_text'] == "seule" and child['child_dep'] == "amod": 
                            has_seule = True
                    if has_case and has_seule and has_acl: 
                        telicity_intentional = True 
                
            ##COMPOSITIONAL TELICITY
            if telicity == True or telicity_intentional == True: 
                telicity_score_sentence = 1
                normalized_scores_sentence.append(telicity_score_sentence)
                break
            elif telicity == False : 
                telicity_score_sentence = 0
                normalized_scores_sentence.append(telicity_score_sentence)
                break
            elif telicity == None: 
                verbal_clause_count = 0 
                temporal_complement = False
                locative_complement = False 
                manner_complement = False 
                directionality = False 
                has_progressive = False 
                normalized_scores_clause = []
                for j, spacy_token in enumerate(sent):
                    # some temporal expressions 
                    if spacy_token['text'] in ["attendre", "tarder"]:  
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "sans": 
                                temporal_complement = True
                    elif spacy_token['text'] == "perdre": 
                        has_adp = False
                        has_instant = False
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "sans": 
                                has_adp = True
                            elif child['child_text'] == "instant": 
                                has_instant = True
                        if has_adp and has_instant: 
                            temporal_complement = True
                    elif spacy_token['text'] in ["simultané", "attendant"]: 
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "en": 
                                temporal_complement = True
                    
                    #EVENTS vs STATES
                    elif spacy_token['POS'] == "VERB" and spacy_token['lemma'] != "voilà" and spacy_token['dep'] != 'fixed' :  
                        telicity_score_clause = 0 
                        verbal_clause_count += 1
                        if spacy_token['lemma'] != "attendre" and spacy_token['text'].endswith("ant"): 
                            for child in spacy_token['children']: 
                                if child['child_text'].lower() == "en" and child['child_pos'] == "ADP":  
                                    manner_complement = True
                        
                        #TENSE (punctual vs durative) 
                        pie_token = next((token for token in pie_sent if token['form'] == spacy_token['text']), None)
                        if pie_token is not None and "TEMPS=psp" in pie_token['morph']: 
                            telicity_score_clause += 1 
                        if pie_token is not None and "TEMPS=ipf" in pie_token['morph']: 
                            telicity_score_clause -= 1
                        
                        #progressive 
                        if spacy_token['head_text'] == "train": 
                            for token in sent: 
                                if token['text'] == "train": 
                                    has_aux = False
                                    has_en = False 
                                    for prog_child in token['children']: 
                                        if prog_child['child_pos'] == "AUX": 
                                            has_aux = True
                                        elif prog_child['child_text'] == "en" and prog_child['child_pos'] == 'ADP': 
                                            has_en = True
                                    if has_aux and has_en: 
                                        has_progressive = True                                    
                                        for prog_child in token['children']: 
                                            if (prog_child['child_dep'] == "nsubj" or prog_child['child_dep'] == "nsubj:pass") and (prog_child['child_pos'] == "PRON" or prog_child['child_pos'] == "PROPN") and prog_child['child_text'].lower() != ["que", "qui", "qu'"]: 
                                                telicity_score_clause += 1
                                            if prog_child['child_dep'] == "nsubj:pass":
                                                telicity_score_clause -= 1
                        for child in spacy_token['children']: 
                            objects = []
                            #SUBJECT 
                            if (child['child_dep'] == "nsubj" or child['child_dep'] == "nsubj:pass") and (child['child_pos'] == "PRON" or child['child_pos'] == "PROPN") and child['child_text'].lower() != ["que", "qui", "qu'"]: 
                                telicity_score_clause += 1
                            #VOICE 
                            if child['child_dep'] == "nsubj:pass":
                                telicity_score_clause -= 1
                            #TENSE  
                            if child['child_pos'] == "AUX" and child['child_dep'] == "aux:tense": 
                                telicity_score_clause += 1
                            # PRESENCE OF A GRAMMATICAL OBJECT 
                            if child['child_dep'] == "obj" or child['child_dep'] == "iobj" : 
                                telicity_score_clause += 1
                                objects.append(child)
                                for item in objects:
                                    for object_token in sent: 
                                        if object_token['text'] == item['child_text'] and object_token['head_text'] == spacy_token['text']: #2nd condition to be sure we have the object in question (in case of repetition of the same word in the sentence at different places)
                                            # OBJECT INDIVIDUATION 
                                            for object_child in object_token['children']: 
                                                if (object_child['child_pos'] == "DET" and object_child['child_dep'] == "det" and object_child['child_text'].lower() in ["le", "la", "l'", "les", "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "nôtre", "notre", "nos", "vôtre", "votre", "vos", "leur", "leurs"]) or object_child['child_pos'] == "NUM": 
                                                    telicity_score_clause += 1
                                     
                                                if object_child['child_pos'] == "DET" and object_child['child_dep'] == "det" and object_child['child_text'].lower() == "des": 
                                                    telicity_score_clause -= 1
                                                
                                                if object_child['child_dep'] in ["amod", "nod", "nmod", "appos"]: #OK
                                                    telicity_score_clause += 1

                            if child['child_dep'] in ["obl:arg", 'obl:mod'] and (child['child_pos'] == "NOUN" or child['child_pos'] == 'PRON'):  
                                telicity_score_clause += 1

                                ### DIRECTIONAL PREPOSITIONS 
                                particle_token = child["child_text"]
                                for k, token in enumerate(sent): 
                                    if sent[k]['text'] == particle_token: 
                                        for child in sent[k]['children']: 
                                            if child['child_text'].lower() in ["jusqu'", "jusque", "vers"] and child['child_dep'] == 'case': #OK
                                                particle = child["child_text"]
                                                for q, token in enumerate(sent): 
                                                    if sent[q]['text'] == particle: 
                                                        found_time_expression = False
                                                        for m in range(q+1, len(sent)):
                                                            if sent[m]['lemma'] in time_expressions: 
                                                                found_time_expression = True 
                                                                break 
                                                            elif sent[m]['POS'] == "PUNCT": 
                                                                break
                                                        if found_time_expression:   
                                                            directionality = False 
                                                        else: 
                                                            directionality = True
                                                               
                        # normalize
                        normalized_score_clause = (telicity_score_clause + 3)/10
                        normalized_scores_clause.append(normalized_score_clause)
                    
                    #ADVERBIAL COMPLEMENTS (temporal, manner, locative) 
                    elif spacy_token['lemma'].endswith('ment') and spacy_token['POS'] == "ADV": 
                        manner_complement = True 
                    elif spacy_token['POS'] == "ADJ" and spacy_token['dep'] == "advmod": 
                        manner_complement = True

                    elif spacy_token['POS'] == "ADV" and spacy_token['lemma'] in temporal_adverbs : 
                        temporal_complement = True 
                    elif spacy_token['lemma'] == "jusque" and spacy_token['head_dep'] != "obl:arg": 
                        temporal_complement = True 
                    elif spacy_token['POS'] == "ADV" and spacy_token['lemma'] in locative_adverbs : 
                        locative_complement = True 
                    elif j+1 < len(sent) and (sent[j]['lemma'] not in ['par', 'de', 'du']) and (sent[j+1]['lemma'] in ["dedans", "dehors", "derrière", "dessous", "dessus", "devant"]): 
                        locative_complement = True 
                    elif spacy_token['POS'] == "ADV" and spacy_token['lemma'] in manner_adverbs : 
                        manner_complement = True 
                    elif spacy_token['lemma'] == "comme" and spacy_token['POS'] == "ADP" : 
                        manner_complement = True
                    elif spacy_token['lemma'] in ["pendant", "durant"] and spacy_token['POS'] != "VERB": 
                        temporal_complement = True 
                    elif spacy_token['POS'] == "ADV" and spacy_token['lemma'] == "abord" and spacy_token['head_text'] == "d'" and spacy_token['POS'] == "ADV" and spacy_token['dep'] == 'fixed': 
                        temporal_complement = True 
                    elif spacy_token['POS'] == "ADV" and spacy_token['lemma'] == "peu" and spacy_token['head_text'] == "sous" and spacy_token['dep'] == 'fixed': 
                        temporal_complement = True 
                    elif spacy_token['lemma'] in ["délai"]:  
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "sans": 
                                temporal_complement = True
                    
                    elif spacy_token['lemma'] in ["bras-le-corps", "tire-larigot", "califourchon", "loisir", "aveuglette", "tire-d'aile", "légère", "tort"] : 
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "à" and child['child_dep'] == "case": 
                                manner_complement = True 
                    elif spacy_token['lemma'] == "guingois": 
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "de" and child['child_dep'] == "case": 
                                manner_complement = True 
                    elif spacy_token['lemma'] == "nouveau" and spacy_token['head_text'] in ["à", "de"] and spacy_token['head_pos'] == "ADP" and spacy_token['dep'] == "fixed": 
                        manner_complement = True 
                    elif spacy_token['lemma'] == "marché" and spacy_token['head_text'] == "bon" and spacy_token['dep'] == "fixed": 
                        manner_complement = True 
                    elif (j+3 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "tue" and sent[j+2]['lemma'] == '-' and sent[j+3]['lemma'] == 'tête') or (j+1 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['lemma'] == "tue-tête"): 
                        manner_complement = True 
                    elif (j+4 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "la" and sent[j+2]['text'] == 'va' and sent[j+3]['lemma'] == '-' and sent[j+4]['lemma'] == 'vite') or (j+2 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "la" and sent[j+2]['text'] == "va-vite"): 
                        manner_complement = True 
                    elif (j+10 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "la" and sent[j+2]['text'] == 'va' and sent[j+3]['lemma'] == '-' and sent[j+4]['lemma'] == 'comme' and sent[j+5]['lemma'] == '-' and sent[j+6]['lemma'] == 'je' and sent[j+7]['lemma'] == '-' and sent[j+8]['text'] == 'te' and sent[j+9]['lemma'] == '-' and sent[j+10]['text'] == 'pousse') or (j+2 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "la" and sent[j+2]['text'] == "va-comme-je-te-pousse"): 
                        manner_complement = True 
                    elif j+2 < len(sent) and (sent[j]['lemma'] == "pour" and sent[j+1]['lemma'] == "de" and sent[j+2]['lemma'] == 'bon'): 
                        manner_complement = True 
                    elif j+2 < len(sent) and (sent[j]['lemma'] == "bel" and sent[j+1]['lemma'] == "et" and sent[j+2]['lemma'] == 'bien'): 
                        manner_complement = True 
                    elif j+3 < len(sent) and (sent[j]['lemma'] == "tant" and sent[j+1]['lemma'] == "bien" and sent[j+2]['lemma'] == 'que' and sent[j+3]['lemma'] == 'mal'): 
                        manner_complement = True 

                    elif spacy_token['lemma'] in time_expressions: 
                        if (spacy_token['dep'] == 'obl:mod' or spacy_token['dep'] == 'fixed') : 
                            temporal_complement = True 
                        elif spacy_token['dep'] == "obj" :  
                            for child in spacy_token['children']: 
                                if child['child_text'].lower() in ["passé", "passée", "passés", "passées", "prochain", "prochaine", "prochains", "prochaines", 
                                "entier", "entière", "entiers", "entières", 
                                "dernier", "dernière", "derniers", "dernières", 
                                "tout", "toute", "tous", "toutes"]: 
                                    temporal_complement = True

                    elif sent[j]['lemma'] in ["sans", "avec"] and sent[j]['POS'] == "ADP" and sent[j+1]['lemma'] != "délai": 
                        manner_complement = True
                    elif j+1 < len(sent) and sent[j]['lemma'] == "par" and sent[j]['POS'] == "ADP" and sent[j]['head_dep'] != "obl:agent" and sent[j+1]['POS'] != "NOUN":  
                        manner_complement = True 

                    elif spacy_token['lemma'] == "pas": 
                        has_de = False
                        has_pas = False
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "de": 
                                has_de = True
                            elif child['child_pos'] == 'DET' and child['child_text'] == "ce": 
                                has_pas = True
                        if has_de and has_pas: 
                            temporal_complement = True

                    elif spacy_token['lemma'] in ["proximité", "abord", "environ"]: #
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["à", "au", "aux"]: 
                                locative_complement = True

                    elif spacy_token['lemma'] in ["côté", "droite", "gauche"]: 
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["à", "au", "aux", "de", "du", "d'"]: 
                                locative_complement = True 

                    elif spacy_token['lemma'] == "milieu": 
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["à", "au", "aux", "de", "du", "en", "d'"]: 
                                locative_complement = True
                                temporal_complement = True 
 
                    elif spacy_token['lemma'] in ["présent", "approche", "début", "fin", "commencement", "mesure", "bout"]: 
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["à", "au"]: 
                                temporal_complement = True
                    
                    elif spacy_token['lemma'] in ["bas", "haut"]:  
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() == "en": 
                                locative_complement = True
                    elif spacy_token['lemma'] in ["intérieur", "extérieur"]:  
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["en", "à"]: 
                                locative_complement = True

                    elif spacy_token['POS'] == "PROPN" and spacy_token['dep'] != 'nmod': 
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["dans", "sous", "sur", "vers", "à", "au", "aux", "en", "de"]: 
                                locative_complement = True

                    elif spacy_token['text'] == "part": 
                        if spacy_token['head_text'].lower() == "quelque" and spacy_token['head_pos'] == "DET":
                            locative_complement = True
                        else: 
                            for child in spacy_token['children']: 
                                if child['child_pos'] == 'DET' and child['child_text'].lower() in ["quelque", "nulle", "nul"]: 
                                    locative_complement = True

                    elif spacy_token['lemma'] == "face": 
                        if spacy_token['head_text'].lower() == "à" and spacy_token['head_pos'] == "ADP":
                            locative_complement = True
                        else: 
                            for child in spacy_token['children']: 
                                if child['child_pos'] == 'ADP' and child['child_text'].lower() in ["à", "au", "aux"]: 
                                    locative_complement = True

                    elif spacy_token['text'].lower() == "tout":
                        has_adp_child = False
                        has_fixed_coup_child = False
                        for child in spacy_token['children']:
                            if child['child_pos'] == 'ADP' and child['child_text'] == "à":
                                has_adp_child = True
                            elif child['child_dep'] == 'fixed' and child['child_text'] == "coup":
                                has_fixed_coup_child = True
                        if has_adp_child and has_fixed_coup_child:
                            temporal_complement = True

                    elif spacy_token['lemma'] == "cependant": 
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'SCONJ' and child['child_text'] in ["que", "qu'"] and child['child_dep'] == 'fixed': 
                                temporal_complement = True

                    elif spacy_token['lemma'] == "tant":
                        for child in spacy_token['children']: 
                            if child['child_pos'] == 'SCONJ' and child['child_text'] in ["que", "qu'"]: 
                                temporal_complement = True
                    
                    elif spacy_token['lemma'] == "suite" and spacy_token['head_text'] == "de" and spacy_token['head_pos'] == "ADP" and spacy_token['dep'] == 'fixed':   
                        temporal_complement = True
                    elif j+3 < len(sent) and sent[j]['lemma'] == "à" and sent[j+1]['text'] == "la" and sent[j+2]['lemma'] == "suite" and sent[j+3]['text'] in ["d'", 'de']: 
                        temporal_complement = True
                    elif j+2 < len(sent) and sent[j]['lemma'] == "partir" and sent[j]['dep'] == "fixed" and sent[j]['head_text'] == "à" and sent[j]['head_pos'] == "ADP" and (sent[j+2]['lemma'] not in temporal_adverbs): 
                        locative_complement = True
                    elif spacy_token['lemma'] == "fois" and spacy_token['dep'] == "fixed" and spacy_token['head_text'] == "une" : 
                        temporal_complement = True
                    elif spacy_token['lemma'] == "temps" and spacy_token['dep'] == "fixed" :  
                        temporal_complement = True
    
                if temporal_complement : 
                    telicity_score_sentence += 1
                if locative_complement: 
                    telicity_score_sentence += 1
                if directionality: 
                    telicity_score_sentence += 1
                if manner_complement: 
                    telicity_score_sentence -= 1
                if has_progressive: 
                    telicity_score_sentence -= 1
             
                # normalize
                if verbal_clause_count > 0: 
                    normalized_average_clauses = sum(normalized_scores_clause) / verbal_clause_count
                else: 
                    normalized_average_clauses = 0

                max_sentence_score = (verbal_clause_count * 7) + 3
                min_sentence_score = (verbal_clause_count * (-3)) -2

                unnormalized_score_sentence = normalized_average_clauses + telicity_score_sentence
                normalized_score_sentence = (unnormalized_score_sentence - min_sentence_score)/(max_sentence_score - min_sentence_score)
                normalized_scores_sentence.append(normalized_score_sentence)
        
    if sentence_count > 0:
        total_telicity_score = sum(normalized_score_sentence for normalized_score_sentence in normalized_scores_sentence) / sentence_count
        print(total_telicity_score)
    else: 
        total_telicity_score = 0
        print(total_telicity_score)
    return total_telicity_score

### AUTRES PROXY TELIQUES
# temps 
@timer
def gerundive(pie_tag_content, verb_count_pie): 
    gerundive_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
           if token['POS'] == "VERppa": 
                gerundive_count += 1
    if verb_count_pie == 0: 
        return None
    else: 
        return gerundive_count/verb_count_pie 
    
@timer
def perfect_vs_imperfect_tense(simple_past_normalized, imperfect_normalized):
    perfect_imperfect_numerator = 0
    perfect_imperfect_denominator = 0
    if simple_past_normalized is not None: 
        perfect_imperfect_numerator += simple_past_normalized
    if imperfect_normalized is not None: 
        perfect_imperfect_denominator += imperfect_normalized
    if perfect_imperfect_denominator == 0: 
        return None
    else: 
        return perfect_imperfect_numerator/perfect_imperfect_denominator

# individuation de l'objet grammatical
@timer
def definite_articles(pie_tag_content, word_count): 
    definite_article_count = 0 
    for sentence in pie_tag_content:
        for token in sentence: 
            if token['POS'] == "DETdef" or token['POS'] == "PRE.DETdef": 
                definite_article_count += 1
    if word_count == 0: 
        return None 
    else: 
        definite_article_normalized = definite_article_count/word_count
        return definite_article_normalized

@timer
def determiners(spacy_tag_content, word_count): 
    determiner_count = 0
    for sent in spacy_tag_content:
        for token in sent: 
            if token['POS'] == "DET": 
                determiner_count += 1
    if word_count == 0: 
        return None
    else: 
        return determiner_count/word_count

@timer
def possessives(pie_tag_content, word_count) : 
    possessive_count = 0
    for sentence in pie_tag_content:
        for token in sentence: 
            if token['POS'] == "DETpos" : 
                possessive_count += 1
    if word_count == 0: 
        return None
    else: 
        possessive_normalized = possessive_count/word_count
        return possessive_normalized

@timer
def demonstratives(pie_tag_content, word_count): 
    demonstrative_count = 0
    for sentence in pie_tag_content:
        for token in sentence: 
            if token['POS'] == "DETdem" : 
                demonstrative_count += 1
    if word_count == 0: 
        return None
    else: 
        demonstrative_normalized = demonstrative_count/word_count
        return demonstrative_normalized

@timer
def numbers(spacy_tag_content, word_count): 
    number_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "NUM": 
                number_count += 1
    if word_count == 0: 
        return None
    else: 
        return number_count/word_count

@timer
def individuated_vs_unindividuated_object(pie_tag_content): 
    individuated_count = 0
    non_individuated_count = 0
    definite_determiners = ["DETdef", "DETdem", "DETpos", "DETcar", "DETcom", "DETrel"]
    indefinite_determiners = ["DETndf", "DETind"]
    for sentence in pie_tag_content: 
        for token in sentence: 
            if token['POS'] in definite_determiners:
                individuated_count += 1
            elif token['POS'] in indefinite_determiners: 
                non_individuated_count += 1
    if non_individuated_count == 0:
        return 0
    else: 
        return individuated_count/non_individuated_count

# locutions adverbiales et prépositionnelles (direction vs manière)
@timer
def temporal_adverbs(concatenated_content, word_count): 
    temporal_adverbs = ["à présent", "a présent", "à l'approche d", "a l'approche d", "à l'approche de", "a l'approche de", "à l'approche du", "a l'approche du", "à l'approche des", "a l'approche des",
        "sur le moment", "sur l'heure",
        "un temps", "un moment", "un instant", "un jour", "un matin", "une nuit", "un soir", "une soirée", "un beau jour", "un beau matin", "un beau soir", "un hiver", "un été", "un printemps", "un automne", "une année", 
        "d'un moment à l'autre", "d'un instant à l'autre",
        "un de ces jours", 
        "au début", 
        "au commencement",
        "à la fin", "a la fin",
        "en fin",
        "au milieu", 
        "aux environs de", "aux environs d", 
        "actuellement",
        "anciennement", 
        "simultanément",
        "de jour", 
        "de nuit", 
        "en simultané",
        "fréquemment",
        "régulièrement",
        "antan", 
        "après", 
        "alors", 
        "auparavant", 
        "aussitôt", 
        "autrefois", 
        "avant", 
        "bientôt", 
        "dans peu", 
        "de suite", 
        "de ce pas", 
        "de temps en temps", 
        "déjà", 
        "demain",
        "d’main", 
        "depuis",
        "derechef", 
        "dernièrement", 
        "désormais", 
        "de tout temps", "de tous temps", 
        "d'abord", 
        "dorénavant", 
        "encore", 
        "enfin", 
        "ensuite", 
        "entre-temps", 
        "entretemps",
        "quelque temps", 
        "hier",
        "hui", 
        "illico", 
        "immédiatement", 
        "instantanément",
        "sous peu",
        "jadis", 
        "jamais", 
        "jusqu'à", "jusqu'au", "jusqu'aux", "jusque", 
        "longtemps", 
        "lors", 
        "maintenant", 
        "maishui",  
        "meshui", 
        "méshui", 
        "momentanément",
        "naguère", 
        "naguères", 
        "parfois", 
        "présentement", 
        "prochainement",
        "puis", 
        "quelquefois", 
        "rarement",
        "récemment", 
        "sans délai", 
        "sans perdre un instant",
        "sans attendre", 
        "sans tarder", "sans trop tarder", 
        "sitôt", 
        "soudain", 
        "soudainement",
        "souvent", 
        "subito", 
        "sur-le-champ", 
        "sur le champ",
        "sur le coup",
        "sur le coup de", "sur les coups de",
        "tantôt", 
        "tard", 
        "tardivement", 
        "tôt", 
        "toujours", 
        "tout à coup" ,
        "tout d'un coup", 
        "tout le long de", "tout le long du", "tout le long d", 
        "tu suite",  
        "tandis",
        "quand", 
        "lorsque", "lorsqu", 
        "une fois que", "une fois qu", 
        "pendant que", "pendant qu", 
        "cependant que", "cependant qu",
        "chaque fois que", "chaque fois qu", 
        "toutes les fois que", "toutes les fois qu", 
        "tant que", "tant qu", 
        "à mesure", "a mesure",
        "le temps que", "le temps qu", 
        "dès", 
        "en attendant", 
        "le temps de", "le temps d", "le temps du", 
        "durant", 
        "pendant", 
        "au cours de", "au cours d",
        "l'espace d'un instant",
        "dans l'instant", 
        "au bout d", "au bout de", 
        "à partir de", "à partir d", 
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", 
        "piéça"] 
    
    ambiguous_prep = ["tout le", "tous les", "toute la", "toutes les", "chaque", "quelques", "à", "au", "pour", "en", "par", "dans", "ce", "cet", "cette", "ces", "l'autre"] 
    ambiguous_post = ["dernier", "dernière", "derniers", "dernières", "passé", "passée", "passés", "passées", "prochain", "prochaine", "prochains", "prochaines", "entier", "entière", "entiers", "entières"] 
    time_expressions = ["coucher", 
        "lever",
        "an", "ans", 
        "année", "années", 
        "soir", "soirs", 
        "soirée", "soirées", 
        "matin", "matins", 
        "matinée", "matinées", 
        "jour", "jours", 
        "nuit", "nuits",
        "journée", "journées", 
        "midi", "midis", 
        "minuit", 
        "heure", "heures", 
        "minute", "minutes", 
        "seconde", "secondes", 
        "mois", 
        "janver", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "siècle", "siècles", 
        "saison", "saisons", 
        "hiver", "printemps", "été", "automne", "étés", "hivers", "printemps", "automnes", #PB "pour avoir été" == ppe être ->EXCLUDE VERBS as inter
        "semaine", "semaines",
        "temps", 
        "moment", "moments", 
        "instant", "instants", 
        "fois"]

    punctuation = ['.', ',', ';', ':', '?', '!']
    words = concatenated_content.split()
    negative_lookbehind = r"(?<!pendant |durant |depuis |lors |le temps d |au cours d |au bout d |à partir d |après |avant |puis |tout le long d )" 

    adverbial_pattern = re.compile(r"\b" + negative_lookbehind + "(" + "|".join(map(re.escape, temporal_adverbs)) + r")['.,\s]", re.IGNORECASE)
    prep_durative_pattern = re.compile(r"\b(" + "|".join(map(re.escape, ambiguous_prep)) + r")\s+(?:(?![,!?;:(.]).)*?(" + "|".join(map(re.escape, time_expressions)) + r")[.,\s]", re.IGNORECASE)
    durative_post_pattern = re.compile(r"\b(" + "|".join(map(re.escape, time_expressions)) + r")\W+(" + "|".join(map(re.escape, ambiguous_post)) + r")[.,\s]", re.IGNORECASE)

    temporal_counts = defaultdict(int)
    used_positions = set() 

    adverbial_matches = adverbial_pattern.finditer(concatenated_content)
    for match in adverbial_matches:
        match_start, match_end = match.span()
        match_text = match.group().lower()
        if not any(match_start <= used_end and used_start <= match_end for used_start, used_end in used_positions):
            temporal_counts[match_text] += 1
            used_positions.add((match_start, match_end))

    prep_durative_matches = prep_durative_pattern.finditer(concatenated_content)
    for match in prep_durative_matches:
        match_start, match_end = match.span()
        match_text = match.group().lower()
        if not any(match_start <= used_end and used_start <= match_end for used_start, used_end in used_positions):
            temporal_counts[match_text] += 1
            used_positions.add((match_start, match_end))

    durative_post_matches = durative_post_pattern.finditer(concatenated_content)
    for match in durative_post_matches:
        match_start, match_end = match.span()
        match_text = match.group().lower()
        if not any(match_start <= used_end and used_start <= match_end for used_start, used_end in used_positions):
            temporal_counts[match_text] += 1
            used_positions.add((match_start, match_end))

    total_temporal_count = sum(temporal_counts.values())
    if word_count == 0: 
        return None
    else: 
        temporal_adverb_normalized = total_temporal_count/word_count
        return temporal_adverb_normalized
   
@timer
def locative_adverbs(concatenated_content, word_count):
    locative_adverbs = ["à proximité", "aí", "aux abords de", "aux abords du", "aux abords d",
        "ailleurs", 
        "alentour", "alentours",
        "arrière", 
        "attenant", 
        "au diable Vauvert", 
        "autour", 
        "en bas", 
        "vers", l
        "çà", 
        "céans", 
        "chez", 
        "ci",  
        "à côté", "sur le côté", "au côté", 
        "deçà",  
        "delà", 
        "dedans",
        "dehors", 
        "derrière", 
        "dessous", 
        "dessus", 
        "devant", 
        "à droite", "à gauche", 
        "exa", 
        "à l'extérieur", 
        "face à", "face au", 
        "hái", 
        "en haut", 
        "ici", 
        "là",
        "à l'intérieur", 
        "léans", 
        "loin", 
        "nulle part", 
        "où", 
        "par monts et par vaux",
        "partout", 
        "près de", "près du", "près d", 
        "proche de", "proche du", "proche d", 
        "quelque part", 
        "sus", 
        "y"] 

    matches = {}
    for adverb in locative_adverbs:
        pattern = r"\b" + re.escape(adverb) + r"['.,\s]"
        matches[adverb] = [(match.start(), match.end()) for match in re.finditer(pattern, concatenated_content, re.IGNORECASE)]
    locative_adverbs_dict = defaultdict(int)
    for adverb, positions in matches.items():
        for start, end in positions:
            matched = False
            for other_adverb, other_positions in matches.items():
                if adverb != other_adverb and any(start <= other_start < end for other_start, other_end in other_positions):
                    matched = True
                    break
            if not matched:
                locative_adverbs_dict[adverb] += 1 

    total_locative_count = sum(locative_adverbs_dict.values())
    if word_count == 0: 
        return None
    else: 
        locative_adverb_normalized = total_locative_count/word_count
        return locative_adverb_normalized

@timer
def manner_adverbs(concatenated_content, word_count): 
    manner_adverbs = ["à bras-le-corps", 
        "à califourchon", 
        "à la légère",
        "à la va-comme-je-te-pousse", 
        "à la va-vite", 
        "à l'aveuglette", 
        "à loisir", 
        "à nouveau", "de nouveau", 
        "à tire-d'aile", 
        "à tire-larigot", 
        "à tort", 
        "à tue-tête", 
        "bel et bien", 
        "bon marché", 
        "d'arrache-pied", 
        "de guingois", 
        "par hasard", 
        "pour de bon", 
        "tant bien que mal",
        "comme", 
        "comment", 
        "ainsi", 
        "aussi", 
        "bien", 
        "debout", 
        "également", 
        "ensemble", 
        "franco", 
        "gratis", 
        "incognito", 
        "mal", 
        "mieux",
        "pis", 
        "plutôt", 
        "presque",
        "quasi", 
        "recta", 
        "vite", 
        "volontiers"]
    
    matches = {}
    for adverb in manner_adverbs:
        pattern = r"\b" + re.escape(adverb) + r"[.,\s]"
        matches[adverb] = [(match.start(), match.end()) for match in re.finditer(pattern, concatenated_content, re.IGNORECASE)]
    manner_adverbs_dict = defaultdict(int)
    for adverb, positions in matches.items():
        for start, end in positions:
            matched = False
            for other_adverb, other_positions in matches.items():
                if adverb != other_adverb and any(start <= other_start < end for other_start, other_end in other_positions):
                    matched = True
                    break
            if not matched:
                manner_adverbs_dict[adverb] += 1 
    total_manner_count = sum(manner_adverbs_dict.values())
    if word_count == 0: 
        return None
    else: 
        return total_manner_count/word_count

@timer
def directional_prep(spacy_tag_content, word_count): 
    time_expressions = ["coucher", 
        "lever",
        "an", "ans", 
        "année",  "années",
        "soir", "soirs", 
        "soirée", "soirées", 
        "matin",  "matins", 
        "matinée", "matinées", 
        "jour",  "jours", 
        "nuit", "nuits", 
        "journée", "journées", 
        "midi",  
        "minuit", 
        "heure", "heures", 
        "minute",  "minutes", 
        "seconde", "secondes", 
        "mois",
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "siècle", "siècles", 
        "saison",  "saisons", 
        "hiver", "printemps", "été", "automne", "hivers", "étés", "automnes", 
        "semaine", "semaines" ,
        "temps", 
        "moment", "moments", 
        "instant", "instants", 
        "fois"]
    
    directional_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "VERB" and token['lemma'] != "voilà" and token['dep'] != 'fixed' : 
                for child in token['children']: 
                    if child['child_dep'] in ["obl:arg", 'obl:mod'] and (child['child_pos'] == "NOUN" or child['child_pos'] == 'PRON'):  
                        particle_token = child["child_text"]
                        for k, token in enumerate(sent): 
                            if sent[k]['text'] == particle_token: 
                                for child in sent[k]['children']: 
                                    if child['child_text'].lower() in ["jusqu'", "jusque", "vers"] and child['child_dep'] == 'case': 
                                        particle = child["child_text"]
                                        for q, token in enumerate(sent): 
                                            if sent[q]['text'] == particle: 
                                                found_time_expression = False
                                                for m in range(q+1, len(sent)):
                                                    if sent[m]['lemma'] in time_expressions: 
                                                        found_time_expression = True 
                                                        break 
                                                    elif sent[m]['POS'] == "PUNCT": 
                                                        break
                                                if not found_time_expression: 
                                                    directional_count += 1
    if word_count == 0: 
        return None
    else:
        return directional_count/word_count 

@timer
def directional_prep_on_adp(spacy_tag_content): 
    time_expressions = ["coucher", 
        "lever",
        "an", "ans", 
        "année",  "années",
        "soir", "soirs", 
        "soirée", "soirées", 
        "matin",  "matins", 
        "matinée", "matinées", 
        "jour",  "jours", 
        "nuit", "nuits", 
        "journée", "journées", 
        "midi",  
        "minuit", 
        "heure", "heures", 
        "minute",  "minutes", 
        "seconde", "secondes", 
        "mois",
        "janver", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "siècle", "siècles", 
        "saison",  "saisons", 
        "hiver", "printemps", "été", "automne", "hivers", "étés", "automnes", 
        "semaine", "semaines" ,
        "temps", 
        "moment", "moments", 
        "instant", "instants", 
        "fois"]
    
    directional_count = 0
    adp_count = 0 
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "ADP": 
                adp_count += 1
            elif token['POS'] == "VERB" and token['lemma'] != "voilà" and token['dep'] != 'fixed' : 
                for child in token['children']: 
                    if child['child_dep'] in ["obl:arg", 'obl:mod'] and (child['child_pos'] == "NOUN" or child['child_pos'] == 'PRON'):  
                        particle_token = child["child_text"]
                        for k, token in enumerate(sent): 
                            if sent[k]['text'] == particle_token: 
                                for child in sent[k]['children']: 
                                    if child['child_text'].lower() in ["jusqu'", "jusque", "vers"] and child['child_dep'] == 'case': 
                                        particle = child["child_text"]
                                        for q, token in enumerate(sent): 
                                            if sent[q]['text'] == particle: 
                                                found_time_expression = False
                                                for m in range(q+1, len(sent)):
                                                    if sent[m]['lemma'] in time_expressions: 
                                                        found_time_expression = True 
                                                        break 
                                                    elif sent[m]['POS'] == "PUNCT": 
                                                        break
                                                if not found_time_expression: 
                                                    directional_count += 1
    if adp_count == 0: 
        return None
    else:
        return directional_count/adp_count 

@timer
def manner_prep(spacy_tag_content, word_count): 
    manner_count = 0
    for sent in spacy_tag_content: 
        for j, token in enumerate(sent): 
            if j+1 < len(sent) and sent[j]['lemma'] in ["sans", "avec"] and sent[j]['POS'] == "ADP" and sent[j+1]['lemma'] != "délai": 
                manner_count += 1
            elif j+1 < len(sent) and sent[j]['lemma'] == "par" and sent[j]['POS'] == "ADP" and sent[j]['head_dep'] != "obl:agent" and sent[j+1]['POS'] != "NOUN":  
                manner_count += 1          
    if word_count == 0: 
        return None
    else:
        return manner_count/word_count

@timer
def manner_prep_on_adp(spacy_tag_content): 
    manner_count = 0
    adp_count = 0 
    for sent in spacy_tag_content: 
        for j, token in enumerate(sent): 
            if sent[j]['POS'] == "ADP": 
                adp_count += 1
                if j+1 < len(sent) and sent[j]['lemma'] in ["sans", "avec"] and sent[j+1]['lemma'] != "délai": 
                    manner_count += 1
                elif j+1 < len(sent) and sent[j]['lemma'] == "par" and sent[j]['head_dep'] != "obl:agent" and sent[j+1]['POS'] != "NOUN":  
                    manner_count += 1          
    if adp_count == 0: 
        return None
    else:
        return manner_count/adp_count

## III. 3.2. MODES IRREELS
@timer
def subjunctive(pie_tag_content, conjugated_verb_count): 
    subjunctive_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
            if token['morph'].startswith("MODE=sub"): 
                subjunctive_count += 1 
    if conjugated_verb_count == 0: 
        return None
    else: 
        return subjunctive_count/conjugated_verb_count

@timer
def conditional(pie_tag_content, conjugated_verb_count): 
    conditional_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if token['morph'].startswith("MODE=con"): 
                conditional_count += 1 
    if conjugated_verb_count == 0: 
        return None
    else: 
        return conditional_count/conjugated_verb_count

@timer
def imperative(pie_tag_content, conjugated_verb_count): 
    imperative_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
            if token['morph'].startswith("MODE=imp"): 
                imperative_count += 1 
    if conjugated_verb_count == 0: 
        return None
    else: 
        return imperative_count/conjugated_verb_count

@timer
def indicative(pie_tag_content, conjugated_verb_count): 
    indicative_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
            if token['morph'].startswith("MODE=ind"): 
                indicative_count += 1 
    if conjugated_verb_count == 0: 
        return None
    else: 
        return indicative_count/conjugated_verb_count

@timer
def present(pie_tag_content, conjugated_verb_count): 
    present_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if "TEMPS=pst" in token['morph']:
                present_count += 1
    if conjugated_verb_count == 0: 
        return None
    else: 
        return present_count/conjugated_verb_count

@timer
def tense_alternance(pie_tag_content, conjugated_verb_count):  
    previous_tense = None 
    alternance_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
            current_tense = None
            if "TEMPS=" in token['morph'] : 
                current_tense = token['morph'].split("TEMPS=")[1].split("|")[0]
            if current_tense and current_tense != previous_tense: 
                alternance_count += 1
                previous_tense = current_tense
    if conjugated_verb_count == 0: 
        return None
    elif (alternance_count == 0 or alternance_count == 1): 
        return 0
    else: 
        return (alternance_count-1)/conjugated_verb_count 

## III. 3.3. MARQUEURS CORRELES d'ABSTRACTION
# composantes pas encore mesurées et calcul du ratio
@timer
def conjunctions(spacy_tag_content, word_count):
    conjunction_count = 0
    for sent in spacy_tag_content:
        for token in sent: 
            if token['POS'] in ["SCONJ", "CCONJ"]: 
                conjunction_count += 1
    if word_count == 0: 
        return None
    else: 
        conjunction_normalized = conjunction_count/word_count
        return conjunction_normalized

@timer
def abstractness_POS(spacy_tag_content, word_count, adverb_normalized, adjective_normalized, verb_count_pie, noun_normalized): 
    abstractness_ratio_numerator = 0
    function_word_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['lemma'] in nlp.Defaults.stop_words: 
                function_word_count += 1
    abstractness_ratio_numerator += function_word_count
    if adverb_normalized is not None:
        abstractness_ratio_numerator += adverb_normalized*word_count
    if adjective_normalized is not None:
        abstractness_ratio_numerator += adjective_normalized*word_count
    abstractness_ratio_denominator = 0
    abstractness_ratio_denominator += verb_count_pie
    if noun_normalized is not None: 
        abstractness_ratio_denominator += noun_normalized*word_count
        
    if abstractness_ratio_denominator == 0:
        return None
    else:
        return abstractness_ratio_numerator/abstractness_ratio_denominator
   
## III. 4. INDICES D'EXTRAVERSION ET D'AGREABILITE 
#index de formalité (composantes pas encore mesurées et calcul du ratio)
@timer
def nouns(spacy_tag_content, word_count): 
    noun_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "NOUN": 
                noun_count += 1
    if word_count == 0: 
        return None
    else: 
        noun_normalized = noun_count/word_count
        return noun_normalized

@timer
def adjectives(spacy_tag_content, word_count):  
    adjective_count = 0
    for sent in spacy_tag_content:
        for token in sent: 
            if token['POS'] == "ADJ": 
                adjective_count += 1
    if word_count == 0: 
        return None
    else: 
        adjective_normalized = adjective_count/word_count
        return adjective_normalized

@timer
def prepositions(pie_tag_content, word_count):
    preposition_count = 0
    for sentence in pie_tag_content: 
        for token in sentence: 
            if token['POS'] == "PRE":  
                preposition_count += 1
    if word_count == 0: 
        return None
    else: 
        preposition_normalized = preposition_count/word_count
        return preposition_normalized

@timer
def articles(spacy_tag_content, word_count): 
    article_count = 0
    for sent in spacy_tag_content:
        for token in sent: 
            if token['POS'] in ['DET', 'ADP'] and token['text'].lower() in ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'au', 'aux']: 
                article_count += 1
    if word_count == 0: 
        return None
    else: 
        article_normalized = article_count/word_count
        return article_normalized

@timer
def adverbs(spacy_tag_content, word_count): 
    adverb_count = 0
    for sent in spacy_tag_content: 
        for token in sent:
            if token['POS'] == "ADV": 
                adverb_count += 1
    if word_count == 0: 
        return None
    else: 
        adverb_normalized = adverb_count/word_count
        return adverb_normalized

@timer
def interjections(spacy_tag_content, word_count): 
    interjection_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if token['POS'] == "INTJ": 
                interjection_count += 1
    if word_count == 0: 
        return None
    else: 
        interjection_normalized = interjection_count/word_count
        return interjection_normalized

@timer
def formality(noun_normalized, adjective_normalized, preposition_normalized, article_normalized, pronoun_normalized, verb_pie_normalized, adverb_normalized, interjection_normalized): 
    formality_ratio = 0
    if noun_normalized is not None:
        formality_ratio += noun_normalized
    if adjective_normalized is not None:
        formality_ratio += adjective_normalized
    if preposition_normalized is not None:
        formality_ratio += preposition_normalized
    if article_normalized is not None:
        formality_ratio += article_normalized
    if pronoun_normalized is not None:
        formality_ratio -= pronoun_normalized
    if verb_pie_normalized is not None: 
        formality_ratio -= verb_pie_normalized
    if adverb_normalized is not None: 
        formality_ratio -= adverb_normalized
    if interjection_normalized is not None: 
        formality_ratio -= interjection_normalized
    
    if formality_ratio == 0:
        return None
    else:
        return (formality_ratio+100)/2

# 
@timer
def punctuation(concatenated_content, word_count):
    dot_count = len(re.findall(r'[\w]\.(?!\.)', concatenated_content))
    comma_count = concatenated_content.count(',')
    semi_colon_count = concatenated_content.count(';')
    colon_count = concatenated_content.count(':')
    exclamation_count = concatenated_content.count('!')
    question_count = concatenated_content.count('?')
    ellipsis_count = concatenated_content.count('...')
    parenthesis_count = concatenated_content.count('(')
    quote_count = concatenated_content.count('«')
    apostrophe_count = concatenated_content.count("'")
    hyphen_count = concatenated_content.count('-')

    punctuation_count = dot_count + comma_count + semi_colon_count + colon_count + exclamation_count + question_count + ellipsis_count + parenthesis_count + quote_count + apostrophe_count + hyphen_count
    if word_count == 0: 
        return punctuation_count, None
    else: 
        punctuation_normalized = punctuation_count/word_count
        return punctuation_count, punctuation_normalized

@timer
def dots(concatenated_content, punctuation_count): 
    dot_count = len(re.findall(r'[\w]\.(?!\.)', concatenated_content)) 
    if punctuation_count == 0: 
        return None
    else: 
        return dot_count/punctuation_count

@timer
def commas(concatenated_content, punctuation_count):
    comma_count = concatenated_content.count(',')
    if punctuation_count == 0: 
        return None
    else: 
        return comma_count/punctuation_count

@timer
def semi_colons(concatenated_content, punctuation_count):
    semi_colon_count = concatenated_content.count(';')
    if punctuation_count == 0: 
        return None
    else: 
        return semi_colon_count/punctuation_count

@timer
def colons(concatenated_content, punctuation_count): 
    colon_count = concatenated_content.count(':')
    if punctuation_count == 0: 
        return None
    else: 
        return colon_count/punctuation_count

@timer
def exclamations(concatenated_content, punctuation_count):
    exclamation_count = concatenated_content.count('!')
    if punctuation_count == 0: 
        return None
    else: 
        exclamation_normalized = exclamation_count/punctuation_count
        return exclamation_normalized

@timer
def questions(concatenated_content, punctuation_count):
    question_count = concatenated_content.count('?')
    if punctuation_count == 0: 
        return None
    else: 
        return question_count/punctuation_count

@timer
def parenthesis(concatenated_content, punctuation_count):
    parenthesis_count = concatenated_content.count('(')
    if punctuation_count == 0: 
        return None
    else: 
        return parenthesis_count/punctuation_count

@timer
def quotes(concatenated_content, punctuation_count):
    quote_count = concatenated_content.count('«')
    if punctuation_count == 0: 
        return None
    else: 
        return quote_count/punctuation_count

@timer
def apostrophes(concatenated_content, punctuation_count): 
    apostrophe_count = concatenated_content.count("'")
    if punctuation_count == 0: 
        return None
    else: 
        return apostrophe_count/punctuation_count

@timer
def hyphens(concatenated_content, punctuation_count):
    hyphen_count = concatenated_content.count('-')
    if punctuation_count == 0: 
        return None
    else: 
        return hyphen_count/punctuation_count

@timer
def negations(pie_tag_content, word_count): 
    negation_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if token['POS'] == "ADVneg": 
                negation_count += 1
    if word_count == 0: 
        return None
    else:
        negation_normalized = negation_count/word_count
        return negation_normalized 

# Uber index
@timer
def lexical_richness(spacy_tag_content, word_count): 
    unique_tokens = set(token["text"].lower() for sent in spacy_tag_content for token in sent if token["POS"] != "PUNCT" and token["text"].lower() not in nlp.Defaults.stop_words)
    total_types = len(unique_tokens)
    if (word_count == 0 or total_types == 0):
        return None
    else:
        log_tokens = math.log(word_count)
        log_tokens = math.log(word_count) 
        log_types = math.log(total_types)
        if (log_tokens - log_types) == 0:
            return None
        else: 
            uber_index_count = (log_tokens**2) / (log_tokens - log_types)
            return uber_index_count

# Style implicite, dynamique et catégorique 
@timer
def implicit_style(verb_pie_normalized, pronoun_normalized, adverb_normalized): 
    implicit_ratio = 0
    if verb_pie_normalized is not None: 
        implicit_ratio += verb_pie_normalized
    if pronoun_normalized is not None: 
        implicit_ratio += pronoun_normalized
    if adverb_normalized is not None: 
        implicit_ratio += adverb_normalized
    if implicit_ratio == 0:
        return None
    else:
        return implicit_ratio

@timer
def dynamic_style(stative_normalized, pronoun_normalized, adverb_normalized): 
    dynamic_ratio = 0
    if stative_normalized is not None: 
        dynamic_ratio += stative_normalized
    if pronoun_normalized is not None: 
        dynamic_ratio += pronoun_normalized
    if adverb_normalized is not None: 
        dynamic_ratio += adverb_normalized
    if dynamic_ratio == 0:
        return None
    else:
        return dynamic_ratio

@timer
def categorical_style_total(article_normalized, preposition_normalized, conjunction_normalized, negation_normalized): 
    total_categorical_ratio = 0
    if article_normalized is not None: 
        total_categorical_ratio += article_normalized
    if preposition_normalized is not None: 
        total_categorical_ratio += preposition_normalized
    if conjunction_normalized is not None: 
        total_categorical_ratio += conjunction_normalized
    if negation_normalized is not None: 
        total_categorical_ratio += negation_normalized
    if total_categorical_ratio == 0:
        return None
    else:
        return total_categorical_ratio

@timer
def categorical_style_conservative(article_normalized, preposition_normalized):
    conservative_categorical_ratio = 0
    if article_normalized is not None: 
        conservative_categorical_ratio += article_normalized
    if preposition_normalized is not None: 
        conservative_categorical_ratio += preposition_normalized
    if conservative_categorical_ratio == 0:
        return None
    else:
        return conservative_categorical_ratio

# 
@timer
def total_words(spacy_tag_content):
    word_count = 0
    for sent in spacy_tag_content: 
        for token in sent: 
            if not (token['POS'] == "PUNCT"): 
                word_count += 1
    return word_count

@timer
def word_length(spacy_tag_content, word_count): 
    char_count = 0
    for sent in spacy_tag_content: 
        for token in sent:
            if not (token['POS'] == "PUNCT"):
                char_count += len(token['text'])
    if word_count == 0: 
        return None
    else: 
        return char_count/word_count

@timer
def words_longer_than_6_letters(spacy_tag_content, word_count): 
    word_longer_than_6_count = 0
    for sent in spacy_tag_content: 
        for token in sent:
            char_count = 0 
            if not (token['POS'] == "PUNCT"): 
                char_count += len(token['text'])
                if char_count > 6: 
                    word_longer_than_6_count += 1
    if word_count == 0: 
        return None
    else: 
        return word_longer_than_6_count/word_count

@timer
def total_sentences(spacy_tag_content): 
    sentence_count = 0
    for sent in spacy_tag_content:
        sentence_count += 1
    return sentence_count

@timer
def words_per_sentence(sentence_count, word_count): 
    if sentence_count == 0: 
        return None
    else: 
        return word_count/sentence_count

@timer
def unique_ratio(spacy_tag_content, word_count): 
    unique_tokens = set(token["text"].lower() for sent in spacy_tag_content for token in sent if token["POS"] != "PUNCT" and token["text"].lower() not in nlp.Defaults.stop_words) 
    if word_count == 0: 
        return None
    else: 
        unique_ratio_count = len(unique_tokens)/word_count
        return unique_ratio_count

@timer
def unique_ratio_lemma(spacy_tag_content): 
    total_tokens_lemma = set()
    for sent in spacy_tag_content:
        for token in sent: 
            total_tokens_lemma.add(token["lemma"])
    
    unique_tokens_lemma = set(token["lemma"] for sent in spacy_tag_content for token in sent if token["POS"] != "PUNCT" and token["lemma"] not in nlp.Defaults.stop_words)

    if len(total_tokens_lemma) == 0: 
        return None
    else: 
        return len(unique_tokens_lemma)/len(total_tokens_lemma)

@timer
def negations_on_verbs(pie_tag_content, verb_count_pie): 
    negation_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if token['POS'] == "ADVneg": 
                negation_count += 1
    if verb_count_pie == 0: 
        return None
    else:
        return negation_count/verb_count_pie 

# passé
@timer
def simple_past(pie_tag_content, conjugated_verb_count): 
    simple_past_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if "TEMPS=psp" in token['morph']:
                simple_past_count += 1
    if conjugated_verb_count == 0: 
        return None
    else: 
        simple_past_normalized = simple_past_count/conjugated_verb_count
        return simple_past_normalized

@timer
def imperfect(pie_tag_content, conjugated_verb_count): 
    imperfect_count = 0
    for sentence in pie_tag_content: 
        for token in sentence:
            if "TEMPS=ipf" in token['morph']:
                imperfect_count += 1
    if conjugated_verb_count == 0: 
        return None
    else: 
        imperfect_normalized = imperfect_count/conjugated_verb_count
        return imperfect_normalized

### METRIQUES DUPLIQUEES AFIN QU'IL Y AIT OU NON UNE CONTRAINTE SUR LE SUJET DU VERBE 
@timer
def verbs_subject(spacy_tag_content, verb_count_pie): 
    je_subject_verb_count = 0
    for sent in spacy_tag_content:
        for token in sent:
            if token['POS'] == "VERB" : 
                for child in token['children']:  
                    if (child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je") : 
                        je_subject_verb_count += 1
                        for child in token['children'] : 
                            if child['child_pos'] == "VERB" and child['child_dep'] == "conj" : 
                                je_subject_verb_count += 1 
            elif token['POS'] == 'AUX' and token['dep'] == 'cop': 
                head_idx = token['head_text']
                for token in sent: 
                    if token['text'] == head_idx : 
                        for child in token['children']: 
                            if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                je_subject_verb_count += 1
    if verb_count_pie == 0: 
        return je_subject_verb_count, None
    else: 
        je_verb_subject_normalized = je_subject_verb_count/verb_count_pie
        return je_subject_verb_count, je_verb_subject_normalized

@timer
def linking_verbs_subject(spacy_tag_content, je_subject_verb_count): 
    linking_verb_subject_count = 0
    for sent in spacy_tag_content:
        for token in sent: 
            has_je = False
            has_cop = False
            for child in token['children']: 
                if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                    has_je = True
                elif child['child_dep'] == 'cop': 
                    has_cop = True
            if has_je and has_cop:
                linking_verb_subject_count += 1
    if je_subject_verb_count == 0: 
        return None
    else: 
        return linking_verb_subject_count/je_subject_verb_count

@timer
def semantic_modality_vs_subject(spacy_tag_content): 
    internal_modality_subject_count = 0
    external_modality_subject_count = 0
    for sent in spacy_tag_content:
        for token in sent:
            if token['lemma'] in ["pouvoir", "vouloir"] : 
                for child in token['children']:
                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je" : 
                        internal_modality_subject_count += 1 
                        break
                else: 
                    if token['head_pos'] == 'VERB':  
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        internal_modality_subject_count += 1  
            elif token['lemma'] == "devoir" : 
                for child in token['children']:
                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je" : 
                        external_modality_subject_count += 1 
                        break
                else: 
                    if token['head_pos'] == 'VERB':  
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        external_modality_subject_count += 1 
            elif token['lemma'] == "falloir": 
                for child in token['children']: 
                    if child['child_pos'] == 'VERB':
                        child_idx = child['child_text']
                        for token in sent: 
                            if token['text'] == child_idx: 
                                for child in token['children']:
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        external_modality_subject_count += 1 
                    elif child['child_dep'] == "ccomp": 
                        child_dep_idx = child['child_text']
                        for token in sent: 
                            if token['text'] == child_dep_idx: 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        external_modality_subject_count += 1 
    if external_modality_subject_count == 0: 
        return None
    else: 
        return internal_modality_subject_count/external_modality_subject_count

@timer
def modal_verbs_subject(spacy_tag_content, je_subject_verb_count): 
    modal_verbs = ["pouvoir", "devoir", "vouloir"] 
    modal_verb_subject_count = 0
    for sent in spacy_tag_content:
        for token in sent:
            if token['lemma'] in modal_verbs : 
                has_je = False
                has_xcomp = False 
                for child in token['children']:
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp' : 
                        has_xcomp = True
                    elif child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                        has_je = True
                        break
                if has_xcomp and has_je: 
                    modal_verb_subject_count += 1
                elif has_xcomp:  
                    if token['head_pos'] == 'VERB':  
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        modal_verb_subject_count += 1  
    if je_subject_verb_count == 0: 
        return None
    else: 
        return modal_verb_subject_count/je_subject_verb_count
    
@timer
def modal_verbs_extended_subject(spacy_tag_content, je_subject_verb_count): 
    extended_modal_verbs = ["pouvoir", "devoir", "vouloir", "falloir", "espérer", "savoir", "penser", "aller"]
    extended_modal_verb_subject_count = 0
    for sent in spacy_tag_content:
        for token in sent:
            if token['lemma'] in extended_modal_verbs : 
                has_je = False
                has_xcomp = False 
                for child in token['children']:
                    if child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp' : 
                        has_xcomp = True
                    elif child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                        has_je = True
                        break
                if has_xcomp and has_je: 
                    extended_modal_verb_subject_count += 1
                elif has_xcomp:  
                    if token['head_pos'] == 'VERB':  
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        extended_modal_verb_subject_count += 1  
    if je_subject_verb_count == 0: 
        return None
    else: 
        return extended_modal_verb_subject_count/je_subject_verb_count

@timer
def internal_vs_external_modals_subject(spacy_tag_content): 
    internal_modal_verb_subject_count = 0
    external_modal_verb_subject_count = 0
    for sent in spacy_tag_content:
        for token in sent:
            if token['lemma'] in ["pouvoir", "vouloir"]:
                has_je_internal = False
                has_xcomp_internal = False
                for child in token['children'] :
                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                        has_je_internal = True
                    elif child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp': 
                        has_xcomp_internal = True
                if has_je_internal and has_xcomp_internal: 
                    internal_modal_verb_subject_count += 1
                elif has_xcomp_internal:  
                    if token['head_pos'] == 'VERB': 
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        internal_modal_verb_subject_count += 1 
            elif token['lemma'] == "devoir" :
                has_je_external = False
                has_xcomp_external = False
                for child in token['children']:
                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je":
                        has_je_external = True
                    elif child['child_pos'] == 'VERB' and child['child_dep'] == 'xcomp': 
                        has_xcomp_external = True
                if has_je_external and has_xcomp_external:   
                    external_modal_verb_subject_count += 1
                elif has_xcomp_external: 
                    if token['head_pos'] == 'VERB': 
                        head_idx = token['head_text']
                        for token in sent: 
                            if token['text'] == head_idx : 
                                for child in token['children']: 
                                    if child['child_text'].lower() == "je" or child['child_text'].lower() == "j'" or child['child_text'] == "-je": 
                                        external_modal_verb_subject_count += 1 
    if external_modal_verb_subject_count == 0: 
        return 0
    else: 
        return internal_modal_verb_subject_count/external_modal_verb_subject_count

@timer
def active_vs_passive_voice_subject(spacy_tag_content): 
    passive_voice_subject_count = 0
    active_voice_subject_count = 0
    counted_sentences = set()
    for i in range(len(spacy_tag_content)):
        sent = spacy_tag_content[i]
        if i not in counted_sentences: 
            for token in sent:
                if (token['text'].lower() == "je" or token['text'].lower() == "j'" or token['text'].lower() == "-je") : 
                    if token['dep'] == "nsubj:pass" : 
                        passive_voice_subject_count += 1
                    else: 
                        active_voice_subject_count += 1
                    counted_sentences.add(i)
                    break
    if passive_voice_subject_count == 0: 
        return None
    else: 
        return active_voice_subject_count/passive_voice_subject_count
    
####ANALYSIS 
# Create empty dataframes to store the results 
df_status = pd.DataFrame(columns=['Oeuvre','Date','Langue', 'Play Genre', 'Most talktative character', 'Order in didascaly', 'Status', 'WC', 'Sentence_count', 'Verbs', 'Conjugated verbs', 'Punctuation', 'Dots', 'Commas', 'Semi-colons', 'Colons', 'Exclamations', 'Questions', 'Ellipsis', 'Parenthesis', 'Quotes', 'Apostrophes', 'Hyphens', 'Word length', 'Words longer than 6 letters', 'WPS', 'Unique ratio', 'Unique ratio lemma', 'Lexical richness', 'Pronouns', 'Individual vs collective pronouns', 'First-person pronouns', 'First-person pronouns / Pronouns', 'Nouns', 'Articles', 'Definite articles', 'Definite vs undefinite articles', 'Determiners', 'Possessives', 'Demonstratives', 'Adverbs', 'Temporal adverbs', 'Locative adverbs', 'Manner adverbs', 'Affirmation adverbs', 'Doubt adverbs', 'Intentional adverbs', 'Prepositions', 'Conjunctions', 'Adjectives', 'Negations', 'Negations / verbs', 'Numbers', 'Interjections', 'Future', 'Present', 'Simple past', 'Imperfect', 'Gerundive', 'Progressive', 'Tense alternance', 'Proximal vs distal deictics' , 'Subjunctive', 'Conditional', 'Imperative', 'Indicative', 'Stative verbs', 'Stative / verbs', 'Linking verbs', 'Semantic modality', 'Modal verbs', 'Modal verbs extended', 'Internal vs external modals', 'Transitive', 'Grammatical agency', 'Individuated vs unindividuated objects', 'Active vs Passive Voice', 'Directional prep', 'Directional prep on adp', 'Manner prep', 'Manner prep on adp', 'Semantic modality with locutor as subject', 'Internal vs external modals with locutor as subject', 'Active vs Passive voice with locutor as subject', 'Telicity', 'Verb with locutor as subject', 'Linking verbs with locutor as subject', 'Modal verbs with locutor as subject', 'Modal verbs extended with locutor as subject', 'Total deictics', 'Conservative deictics', 'Abstractness ratio based on POS', 'Formality index', 'Implicit style', 'Dynamic style', 'Categorical style', 'Categorical style conservative', 'Perfect vs imperfect tense'])
df_gender = pd.DataFrame(columns=['Oeuvre','Date','Langue', 'Play Genre', 'Most taltkative character', 'Order in didascaly', 'Gender', 'WC', 'Sentence_count', 'Verbs', 'Conjugated verbs', 'Punctuation', 'Dots', 'Commas', 'Semi-colons', 'Colons', 'Exclamations', 'Questions', 'Ellipsis', 'Parenthesis', 'Quotes', 'Apostrophes', 'Hyphens', 'Word length', 'Words longer than 6 letters', 'WPS', 'Unique ratio', 'Unique ratio lemma', 'Lexical richness', 'Pronouns', 'Individual vs collective pronouns', 'First-person pronouns', 'First-person pronouns / Pronouns', 'Nouns', 'Articles', 'Definite articles', 'Definite vs undefinite articles', 'Determiners', 'Possessives', 'Demonstratives', 'Adverbs', 'Temporal adverbs', 'Locative adverbs', 'Manner adverbs', 'Affirmation adverbs', 'Doubt adverbs', 'Intentional adverbs', 'Prepositions', 'Conjunctions', 'Adjectives', 'Negations', 'Negations / verbs', 'Numbers', 'Interjections', 'Future', 'Present', 'Simple past', 'Imperfect', 'Gerundive', 'Progressive', 'Tense alternance', 'Proximal vs distal deictics' , 'Subjunctive', 'Conditional', 'Imperative', 'Indicative', 'Stative verbs', 'Stative / verbs', 'Linking verbs', 'Semantic modality', 'Modal verbs', 'Modal verbs extended', 'Internal vs external modals', 'Transitive', 'Grammatical agency', 'Individuated vs unindividuated objects', 'Active vs Passive Voice', 'Directional prep', 'Directional prep on adp', 'Manner prep', 'Manner prep on adp', 'Semantic modality with locutor as subject', 'Internal vs external modals with locutor as subject', 'Active vs Passive voice with locutor as subject', 'Telicity', 'Verb with locutor as subject', 'Linking verbs with locutor as subject', 'Modal verbs with locutor as subject', 'Modal verbs extended with locutor as subject', 'Total deictics', 'Conservative deictics', 'Abstractness ratio based on POS', 'Formality index', 'Implicit style', 'Dynamic style', 'Categorical style', 'Categorical style conservative', 'Perfect vs imperfect tense'])

folder_path = 'https://github.com/CamillePerault/Agency_in_language__an_evolutionary_perspective/tree/main/French_Raw'

# Initialize a list to store the file names of the plays
file_list = []

# Iterate over the plays in the folder
for file_name in os.listdir(folder_path):
    #to only take the .txt file, not the .xml
    if file_name.endswith('.txt'):
        # add the corresponding file name to the list
        file_list.append(file_name)

# Iterate over each play
for file_name in file_list:
    # Construct the full path to the play file
    file_path = os.path.join(folder_path, file_name)
    #Get the base name of the file
    base_name = os.path.basename(file_path)
    # Split the base name by '-'
    parts = base_name.split("-")
    # Get the year from the third part (that will be the value of the column 'Date')
    year = parts[1]
    play_genre = find_genre_of_the_play(file_name)

    # Open the play, read its contents and apply the functions
    play = open(file_path, encoding = "utf-8").read()
    text, characters_name, characters_status = extract_character_status_from_didascaly(play)
    characters_gender = assign_gender_to_character(file_name)
    filtered_textlines, scene_textlines = filtered_text_in_lines(text)
    characters_lines = attribute_replica_to_character(filtered_textlines, characters_name)
    most_talktative_character = narrative_importance_words_spoken(characters_lines, scene_textlines, characters_name)
    order_in_didascaly = narrative_importance_order_didascaly(characters_name)

    # play selection : we only retain plays that have a least 1 character for each status and gender 
    presence_status = attribute_replica_to_status(characters_lines, characters_status) 
    if presence_status is None: 
        continue
    high_status_words_concat, low_status_words_concat = presence_status
    
    presence_gender = attribute_replica_to_gender(characters_lines, characters_gender) 
    if presence_gender is None: 
        continue
    male_words_concat, female_words_concat = presence_gender
    
    #Create separate files for LIWC processing
    high_status_file_name = f"{base_name}_high.txt"
    low_status_file_name = f"{base_name}_low.txt"
    female_file_name = f"{base_name}_female.txt"
    male_file_name = f"{base_name}_male.txt"
    with open(high_status_file_name, 'w', encoding="utf-8") as f:
        f.write(high_status_words_concat)
    with open(low_status_file_name, 'w', encoding="utf-8") as f:
        f.write(low_status_words_concat)
    with open(female_file_name, 'w', encoding="utf-8") as f:
        f.write(female_words_concat)
    with open(male_file_name, 'w', encoding="utf-8") as f:
        f.write(male_words_concat)

    spacy_tags_high, pie_tags_high = tag_sentences(high_status_words_concat)
    spacy_tags_low, pie_tags_low = tag_sentences(low_status_words_concat)
    spacy_tags_female, pie_tags_female = tag_sentences(female_words_concat)
    spacy_tags_male, pie_tags_male = tag_sentences(male_words_concat)

    # Create intermediary json files to have access to the parsed texts
    high_status_file_name_spacy = f"{base_name}_high_spacy.json"
    low_status_file_name_spacy = f"{base_name}_low_spacy.json"
    female_file_name_spacy = f"{base_name}_female_spacy.json"
    male_file_name_spacy = f"{base_name}_male_spacy.json"
    with open(high_status_file_name_spacy, 'w', encoding="utf-8") as f:
        json.dump(spacy_tags_high, f)
    with open(low_status_file_name_spacy, 'w', encoding="utf-8") as f:
        json.dump(spacy_tags_low, f)
    with open(female_file_name_spacy, 'w', encoding="utf-8") as f:
        json.dump(spacy_tags_female, f)
    with open(male_file_name_spacy, 'w', encoding="utf-8") as f:
        json.dump(spacy_tags_male, f)
    
    high_status_file_name_pie = f"{base_name}_high_pie.json"
    low_status_file_name_pie = f"{base_name}_low_pie.json"
    female_file_name_pie = f"{base_name}_female_pie.json"
    male_file_name_pie = f"{base_name}_male_pie.json"
    with open(high_status_file_name_pie, 'w', encoding="utf-8") as f:
        json.dump(pie_tags_high, f)
    with open(low_status_file_name_pie, 'w', encoding="utf-8") as f:
        json.dump(pie_tags_low, f)
    with open(female_file_name_pie, 'w', encoding="utf-8") as f:
        json.dump(pie_tags_female, f)
    with open(male_file_name_pie, 'w', encoding="utf-8") as f:
        json.dump(pie_tags_male ,f)

    #Create a list with all the functions to be applied to these two documents separately (to calculate our metrics)
    functions = [dots, commas, semi_colons, colons, exclamations, questions, ellipsis, parenthesis, quotes, apostrophes, hyphens, word_length, words_longer_than_6_letters, words_per_sentence, unique_ratio, unique_ratio_lemma, lexical_richness, pronouns, individual_vs_collective_pronouns, first_person_pronouns, first_person_pronouns_on_pronouns, nouns, articles, definite_articles, definite_vs_undefinite_articles, determiners, possessives, demonstratives, adverbs, temporal_adverbs, locative_adverbs, manner_adverbs, affirmation_adverbs, doubt_adverbs, intentional_adverbs, prepositions, conjunctions, adjectives, negations, negations_on_verbs, numbers, interjections, future, present, simple_past, imperfect, gerundive, progressive, tense_alternance, proximal_distal_deictics, subjunctive, conditional, imperative, indicative, stative_verbs, stative_on_verbs, linking_verbs, semantic_modality_vs, modal_verbs, modal_verbs_extended, internal_vs_external_modals, transitive, grammatical_agency, individuated_vs_unindividuated_object, active_vs_passive_voice, directional_prep, directional_prep_on_adp, manner_prep, manner_prep_on_adp, semantic_modality_vs_subject, internal_vs_external_modals_subject, active_vs_passive_voice_subject, telicity] 

    concat_status =  {'high': high_status_words_concat, 'low': low_status_words_concat}
    concat_gender = {'male': male_words_concat, 'female': female_words_concat}
    tags_status = {'high': [spacy_tags_high, pie_tags_high], 'low': [spacy_tags_low, pie_tags_low]}
    tags_gender = {'male': [spacy_tags_male, pie_tags_male],'female': [spacy_tags_female, pie_tags_female]}

    #Initialize a dictionary that will contain the results of the functions for each document
    results_status = {'high':{},'low':{}}
    results_gender = {'male':{},'female':{}}

    from inspect import signature

    # Define a function to allow multiple (different) parameters when running the different functions in one loop
    def sub_dict(d, keys):
        return dict((k, d[k]) for k in keys)

    # Calculate all the metrics for each status' concatenated replicas
    for status in tags_status: 
        word_count = total_words(tags_status[status][0])
        sentence_count = total_sentences(tags_status[status][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_status[status][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_status[status][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_status[status], word_count)
        params = dict(spacy_tag_content=tags_status[status][0], pie_tag_content=tags_status[status][1], concatenated_content=concat_status[status], word_count=word_count, punctuation_count=punctuation_count, conjugated_verb_count=conjugated_verb_count, verb_count_pie=verb_count_pie, sentence_count=sentence_count)
        for func in functions: 
            function_name = func.__name__
            result = func(**sub_dict(params, signature(func).parameters))
            results_status[status][func.__name__] = result
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_status[status][0], verb_count_pie)
        results_status[status]["linking_verbs_subject"] = linking_verbs_subject(spacy_tag_content=tags_status[status][0], je_subject_verb_count=je_subject_verb_count)
        results_status[status]["modal_verbs_subject"] = modal_verbs_subject(spacy_tag_content=tags_status[status][0], je_subject_verb_count=je_subject_verb_count)
        results_status[status]["modal_verbs_extended_subject"] = modal_verbs_extended_subject(spacy_tag_content=tags_status[status][0], je_subject_verb_count=je_subject_verb_count)

        results_status[status]["deictics_total"] = deictics_total(pronoun_normalized=results_status[status]["pronouns"],
                                               possessive_normalized=results_status[status]["possessives"],
                                               temporal_adverb_normalized=results_status[status]["temporal_adverbs"],
                                               locative_adverb_normalized=results_status[status]["locative_adverbs"],
                                               definite_article_normalized=results_status[status]["definite_articles"],
                                               demonstrative_normalized=results_status[status]["demonstratives"],
                                               exclamation_normalized=results_status[status]["exclamations"],
                                               interjection_normalized=results_status[status]["interjections"])
        results_status[status]["deictics_conservative"] = deictics_conservative(pronoun_normalized=results_status[status]["pronouns"],
                                               possessive_normalized=results_status[status]["possessives"],
                                               temporal_adverb_normalized=results_status[status]["temporal_adverbs"],
                                               locative_adverb_normalized=results_status[status]["locative_adverbs"])
        results_status[status]["abstractness_POS"] = abstractness_POS(spacy_tag_content=tags_status[status][0], 
                                               word_count=word_count, 
                                               adverb_normalized=results_status[status]["adverbs"],
                                               adjective_normalized=results_status[status]["adjectives"],
                                               verb_count_pie=verb_count_pie,
                                               noun_normalized=results_status[status]["nouns"])
        results_status[status]["formality"] = formality(noun_normalized=results_status[status]["nouns"],
                                               adjective_normalized=results_status[status]["adjectives"],
                                               preposition_normalized=results_status[status]["prepositions"],
                                               article_normalized=results_status[status]["articles"],
                                               pronoun_normalized=results_status[status]["pronouns"],
                                               verb_pie_normalized=verb_pie_normalized,
                                               adverb_normalized=results_status[status]["adverbs"],
                                               interjection_normalized=results_status[status]["interjections"])
        results_status[status]["implicit_style"] = implicit_style(verb_pie_normalized=verb_pie_normalized,
                                               pronoun_normalized=results_status[status]["pronouns"],
                                               adverb_normalized=results_status[status]["adverbs"])
        results_status[status]["dynamic_style"] = dynamic_style(stative_normalized=results_status[status]["stative_verbs"],
                                               pronoun_normalized=results_status[status]["pronouns"],
                                               adverb_normalized=results_status[status]["adverbs"])                                       
        results_status[status]["categorical_style_total"] = categorical_style_total(article_normalized=results_status[status]["articles"],
                                               preposition_normalized=results_status[status]["prepositions"],
                                               conjunction_normalized=results_status[status]["conjunctions"],
                                               negation_normalized=results_status[status]["negations"])                
        results_status[status]["categorical_style_conservative"] = categorical_style_conservative(article_normalized=results_status[status]["articles"],
                                               preposition_normalized=results_status[status]["prepositions"])                                                              
        results_status[status]["perfect_vs_imperfect_tense"] = perfect_vs_imperfect_tense(simple_past_normalized=results_status[status]["simple_past"],
                                               imperfect_normalized=results_status[status]["imperfect"])  
    # idem for each gender
    for gender in tags_gender: 
        word_count = total_words(tags_gender[gender][0])
        sentence_count = total_sentences(tags_gender[gender][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_gender[gender][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_gender[gender][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_gender[gender], word_count)
        params = dict(spacy_tag_content=tags_gender[gender][0], pie_tag_content=tags_gender[gender][1], concatenated_content=concat_gender[gender], word_count=word_count, punctuation_count=punctuation_count, conjugated_verb_count=conjugated_verb_count, verb_count_pie=verb_count_pie, sentence_count=sentence_count)
        for func in functions: 
            function_name = func.__name__
            result = func(**sub_dict(params, signature(func).parameters))
            results_gender[gender][func.__name__] = result
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_gender[gender][0], verb_count_pie)
        results_gender[gender]["linking_verbs_subject"] = linking_verbs_subject(spacy_tag_content=tags_gender[gender][0], je_subject_verb_count=je_subject_verb_count)
        results_gender[gender]["modal_verbs_subject"] = modal_verbs_subject(spacy_tag_content=tags_gender[gender][0], je_subject_verb_count=je_subject_verb_count)
        results_gender[gender]["modal_verbs_extended_subject"] = modal_verbs_extended_subject(spacy_tag_content=tags_gender[gender][0], je_subject_verb_count=je_subject_verb_count)

        results_gender[gender]["deictics_total"] = deictics_total(pronoun_normalized=results_gender[gender]["pronouns"],
                                               possessive_normalized=results_gender[gender]["possessives"],
                                               temporal_adverb_normalized=results_gender[gender]["temporal_adverbs"],
                                               locative_adverb_normalized=results_gender[gender]["locative_adverbs"],
                                               definite_article_normalized=results_gender[gender]["definite_articles"],
                                               demonstrative_normalized=results_gender[gender]["demonstratives"],
                                               exclamation_normalized=results_gender[gender]["exclamations"],
                                               interjection_normalized=results_gender[gender]["interjections"])
        results_gender[gender]["deictics_conservative"] = deictics_conservative(pronoun_normalized=results_gender[gender]["pronouns"],
                                               possessive_normalized=results_gender[gender]["possessives"],
                                               temporal_adverb_normalized=results_gender[gender]["temporal_adverbs"],
                                               locative_adverb_normalized=results_gender[gender]["locative_adverbs"])
        results_gender[gender]["abstractness_POS"] = abstractness_POS(spacy_tag_content=tags_gender[gender][0], 
                                               word_count=word_count, 
                                               adverb_normalized=results_gender[gender]["adverbs"],
                                               adjective_normalized=results_gender[gender]["adjectives"],
                                               verb_count_pie=verb_count_pie,
                                               noun_normalized=results_gender[gender]["nouns"])
        results_gender[gender]["formality"] = formality(noun_normalized=results_gender[gender]["nouns"],
                                               adjective_normalized=results_gender[gender]["adjectives"],
                                               preposition_normalized=results_gender[gender]["prepositions"],
                                               article_normalized=results_gender[gender]["articles"],
                                               pronoun_normalized=results_gender[gender]["pronouns"],
                                               verb_pie_normalized=verb_pie_normalized,
                                               adverb_normalized=results_gender[gender]["adverbs"],
                                               interjection_normalized=results_gender[gender]["interjections"])
        results_gender[gender]["implicit_style"] = implicit_style(verb_pie_normalized=verb_pie_normalized,
                                               pronoun_normalized=results_gender[gender]["pronouns"],
                                               adverb_normalized=results_gender[gender]["adverbs"])
        results_gender[gender]["dynamic_style"] = dynamic_style(stative_normalized=results_gender[gender]["stative_verbs"],
                                               pronoun_normalized=results_gender[gender]["pronouns"],
                                               adverb_normalized=results_gender[gender]["adverbs"])                                       
        results_gender[gender]["categorical_style_total"] = categorical_style_total(article_normalized=results_gender[gender]["articles"],
                                               preposition_normalized=results_gender[gender]["prepositions"],
                                               conjunction_normalized=results_gender[gender]["conjunctions"],
                                               negation_normalized=results_gender[gender]["negations"])                
        results_gender[gender]["categorical_style_conservative"] = categorical_style_conservative(article_normalized=results_gender[gender]["articles"],
                                               preposition_normalized=results_gender[gender]["prepositions"])                                                              
        results_gender[gender]["perfect_vs_imperfect_tense"] = perfect_vs_imperfect_tense(simple_past_normalized=results_gender[gender]["simple_past"],
                                               imperfect_normalized=results_gender[gender]["imperfect"])  

    if "female" in tags_gender:
        word_count = total_words(tags_gender["female"][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_gender["female"][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_gender["female"][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_gender["female"], word_count)
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_gender["female"][0], verb_count_pie)
        df_gender.loc[len(df_gender)] = [os.path.basename(file_path), year, "français", play_genre, most_talktative_character, order_in_didascaly, "female", word_count, total_sentences(tags_gender["female"][0]), verb_pie_normalized, conjugated_verb_normalized, punctuation_normalized] + [results_gender["female"][function.__name__] for function in functions] + [je_verb_subject_normalized, results_gender['female']["linking_verbs_subject"], results_gender['female']["modal_verbs_subject"], results_gender['female']["modal_verbs_extended_subject"]] + [results_gender["female"]["deictics_total"], results_gender["female"]["deictics_conservative"], results_gender["female"]["abstractness_POS"], results_gender["female"]["formality"], results_gender[gender]["implicit_style"], results_gender["female"]["dynamic_style"], results_gender["female"]["categorical_style_total"], results_gender["female"]["categorical_style_conservative"], results_gender["female"]["perfect_vs_imperfect_tense"]]

    if "male" in tags_gender:
        word_count = total_words(tags_gender["male"][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_gender["male"][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_gender["male"][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_gender["male"], word_count)
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_gender["male"][0], verb_count_pie)
        df_gender.loc[len(df_gender)] = [os.path.basename(file_path), year, "français", play_genre, most_talktative_character, order_in_didascaly, "male", word_count, total_sentences(tags_gender["male"][0]), verb_pie_normalized, conjugated_verb_normalized, punctuation_normalized] + [results_gender["male"][function.__name__] for function in functions] + [je_verb_subject_normalized, results_gender['male']["linking_verbs_subject"], results_gender['male']["modal_verbs_subject"], results_gender['male']["modal_verbs_extended_subject"]] + [results_gender["male"]["deictics_total"], results_gender["male"]["deictics_conservative"], results_gender["male"]["abstractness_POS"], results_gender[gender]["formality"], results_gender["male"]["implicit_style"], results_gender["male"]["dynamic_style"], results_gender["male"]["categorical_style_total"], results_gender["male"]["categorical_style_conservative"], results_gender["male"]["perfect_vs_imperfect_tense"]]

    # Write the output file
    df_gender.to_csv('gender_results.csv', index=False)

    if "high" in tags_status:
        word_count = total_words(tags_status["high"][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_status["high"][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_status["high"][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_status["high"], word_count)
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_status["high"][0], verb_count_pie)
        df_status.loc[len(df_status)] = [os.path.basename(file_path), year, "français", play_genre, most_talktative_character, order_in_didascaly, "high", word_count, total_sentences(tags_status["high"][0]), verb_pie_normalized, conjugated_verb_normalized, punctuation_normalized] + [results_status["high"][function.__name__] for function in functions] + [je_verb_subject_normalized, results_status['high']["linking_verbs_subject"], results_status['high']["modal_verbs_subject"], results_status['high']["modal_verbs_extended_subject"]] + [results_status["high"]["deictics_total"], results_status["high"]["deictics_conservative"], results_status["high"]["abstractness_POS"], results_status["high"]["formality"], results_status["high"]["implicit_style"], results_status["high"]["dynamic_style"], results_status["high"]["categorical_style_total"], results_status["high"]["categorical_style_conservative"], results_status["high"]["perfect_vs_imperfect_tense"]]

    if "low" in tags_status:
        word_count = total_words(tags_status["low"][0])
        verb_count_pie, verb_pie_normalized = verbs_pie(tags_status["low"][1], word_count)
        conjugated_verb_count, conjugated_verb_normalized = conjugated_verbs(tags_status["low"][1], verb_count_pie)
        punctuation_count, punctuation_normalized = punctuation(concat_status["low"], word_count)
        je_subject_verb_count, je_verb_subject_normalized = verbs_subject(tags_status["low"][0], verb_count_pie)
        df_status.loc[len(df_status)] = [os.path.basename(file_path), year, "français", play_genre, most_talktative_character, order_in_didascaly, "low", word_count, total_sentences(tags_status["low"][0]), verb_pie_normalized, conjugated_verb_normalized, punctuation_normalized] + [results_status["low"][function.__name__] for function in functions] + [je_verb_subject_normalized, results_status['low']["linking_verbs_subject"], results_status['low']["modal_verbs_subject"], results_status['low']["modal_verbs_extended_subject"]] + [results_status["low"]["deictics_total"], results_status["low"]["deictics_conservative"], results_status["low"]["abstractness_POS"], results_status["low"]["formality"], results_status["low"]["implicit_style"], results_status["low"]["dynamic_style"], results_status["low"]["categorical_style_total"], results_status["low"]["categorical_style_conservative"], results_status["low"]["perfect_vs_imperfect_tense"]]

    # Write the output file
    df_status.to_csv('status_results.csv', index=False)

