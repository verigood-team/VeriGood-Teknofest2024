import mini_pyabsa.AspectSentimentTripletExtraction as ASTE
import ast

triplet_extractor = ASTE.AspectSentimentTripletExtractor(r"checkpoints\dataset_45.31")

def get_lines():
    file_path = r"1.custom\custom.test.dat.aste"
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

def find_long_indices(sentence, analysis):

    expanded_indices = []
    for each in analysis:
        parsed_list = list(each)
        new_indices = []

        if len(parsed_list[0]) == 1:
            new_indices.append(int(parsed_list[0][0]))
            new_indices.append(parsed_list[-1])
        elif len(parsed_list[0]) == 2:
            if (int(parsed_list[0][0]) + 1 == int(parsed_list[0][1])):
                new_indices.append(int(parsed_list[0][0]))
                new_indices.append(int(parsed_list[0][1]))
                new_indices.append(parsed_list[-1])
            else:
                value = int(parsed_list[0][0])
                while (value <= int(parsed_list[0][1])):
                    new_indices.append(value)
                    value += 1
                new_indices.append(parsed_list[-1])
        else:
            for index in parsed_list[0]:
                new_indices.append(int(index))
            new_indices.append(parsed_list[-1])
        expanded_indices.append(new_indices)
    return expanded_indices

def en_to_tr(expanded_indices):
    for index in expanded_indices:
        if index[-1] == "POS":
            index[-1] = "olumlu"
        elif index[-1] == "NEG":
            index[-1] = "olumsuz"
        elif index[-1] == "NEU":
            index[-1] = "nötr"
    return expanded_indices     

def find_index_response(sentence, indices):

    changed_entities = []
    sentence = sentence.split()
    for each in indices:
        target_entity = ""
        finded_entities = []
        for index in range(len(each)-1):
            target_entity += sentence[each[index]]
            target_entity += " "
        target_entity = target_entity[:-1]

        if target_entity not in finded_entities:
            finded_entities.append(target_entity)
            finded_entities.append(each[-1])
        changed_entities.append(finded_entities)
    return changed_entities

def find_entities(str_results):
    
    just_entities = []
    for result in str_results:
        just_entities.append(result[0])
    return just_entities

def entity_analysis(process_list):

    suffixes = ["gen","sal","sel","gıl","gil", "mız", "miz", "muz", "müz", "nuz", "nüz", "nız", "niz", "ca", "ce", "ça","çe","da","de", "dı", "di", "du", "dü", "ta","te", "tı", "ti", "tu", "tü", "lu","lü", "li","lı","la","le","ki", "n", "m","ş","s", "y", "k", "a","e","r","ı","i","u","ü", "z"]
    control = ["@",",",".","!","?","#","*"]

    
    for i in range(len(process_list)):
        
        processing_word = process_list[i]
        while len(processing_word) > 0 and processing_word[0] in control:
            processing_word = processing_word[1:].strip()
        while len(processing_word) > 0 and processing_word[-1] in control:
            processing_word = processing_word[:-1].strip()
        
        process_list[i] = processing_word

        len_processing_word = len(processing_word.split())
        if len_processing_word > 1:
            splited_words = processing_word.split()
            processing_word = splited_words[-1]
        
        if "'" in processing_word or "’" in processing_word:
            cut_word = processing_word
            j=0
            while j < len(suffixes):
                if processing_word.endswith(suffixes[j]):
                    cut_word = processing_word[:-len(suffixes[j])]
                    processing_word = cut_word
                    j=0
                else:
                    j+=1
            cut_word = cut_word[:-1]
            if len_processing_word == 1:
                process_list[i] = cut_word
            else:
                temp_word = process_list[i]
                splited_temp = temp_word.split()
                splited_temp[len_processing_word-1] = cut_word
                splited_temp = splited_temp[:len_processing_word]
                process_list[i] = ' '.join(splited_temp)

    return process_list     

def make_pair_list(test_entity_list, str_results):

    all_test_entity_list = []
    for i in range(len(test_entity_list)):
        new_test_entity_list = []
        new_test_entity_list.append(test_entity_list[i])
        new_test_entity_list.append(str_results[i][1])
        all_test_entity_list.append(new_test_entity_list)
    return all_test_entity_list          

def fill_results(entities):

    test_entity_list = []
    test_results = []

    for entity in entities:
        new_dict = {}
        new_dict[entity[0]] = entity[1]
        
        new_key = entity[0].lower()
        new_value = entity[1].lower()

        exists = False
        for res in test_results:
            res_key = list(res.keys())[0].lower()
            res_value = list(res.values())[0].lower()
            if res_value == new_value and res_key == new_key:
                exists = True
                break
        
        if not exists:
            test_results.append(new_dict)


    key_polarities = {}
    original_keys = {}

    for d in test_results:
        original_key = list(d.keys())[0]
        key_lower = original_key.lower()
        value = d[original_key]
        
        if key_lower not in original_keys:
            original_keys[key_lower] = original_key
        
        if key_lower in key_polarities:
            key_polarities[key_lower].append(value)
        else:
            key_polarities[key_lower] = [value]

    filtered_results = []
    for key, values in key_polarities.items():
        if len(values) > 1 and "nötr" in values:
            values.remove("nötr")
        for value in values:
            filtered_results.append({original_keys[key]: value})

    test_results = filtered_results

    for res in test_results:
        res_key = list(res.keys())[0]
        if res_key not in test_entity_list:
            test_entity_list.append(res_key)

    return test_entity_list, test_results

def calculate_score(test_entity_list,test_results, train_entity_list,train_results):
    
    true_entity_sentiment = 0
    true_entity = 0

    for element in test_entity_list:
        if element in train_entity_list:
            true_entity += 1
    for dict in test_results:
        if dict in train_results:
            true_entity_sentiment += 1

    score = (0.65 * true_entity_sentiment / max(len(train_results),len(test_results))) + (0.35 * true_entity / max(len(train_entity_list), len(test_entity_list)))
    return score



def extract_entities_with_sentiment(entities_list):
    
    entities_set = set()
    
    for entity in entities_list:
        for key, value in entity.items():
            entities_set.add((key, value))
    
    return entities_set



def calculate_confusions_matrix(true_entities, predicted_entities):
    
    true_set = extract_entities_with_sentiment(true_entities)
    predicted_set = extract_entities_with_sentiment(predicted_entities)
    
    true_positive = len(true_set & predicted_set)
    false_positive = len(predicted_set - true_set)
    false_negative = len(true_set - predicted_set)

    return true_positive, false_positive, false_negative 



def calculate_f1_score(true_positive, false_positive, false_negative):

    # Precision ve Recall hesaplama
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # F1 skoru hesaplama
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score



grand_score = 0
count = 0

true_positive = 0
false_positive = 0
false_negative = 0

def grand():
    return grand_score / count

for line in get_lines():

    
    divided = line.split(" ####")

    sentence = divided[0].strip()
    analysis = ast.literal_eval(divided[1].strip())

    expandend_indices = find_long_indices(sentence,analysis)
    expandend_indices = en_to_tr(expandend_indices)
    str_results = find_index_response(sentence, expandend_indices)
    entitiy_names = find_entities(str_results)
    test_entity_results = entity_analysis(entitiy_names)
    test_entity_results = make_pair_list(test_entity_results, str_results)
    test_entity_list, test_results = fill_results(test_entity_results)
    train_entity_list, train_results = triplet_extractor.predict(sentence)
    score = calculate_score(test_entity_list,test_results, train_entity_list,train_results)
    confusion_matrix = calculate_confusions_matrix(test_results, train_results)
    true_positive += confusion_matrix[0]
    false_positive += confusion_matrix[1]
    false_negative += confusion_matrix[2]

    # print("\nTEST\n")
    # print(test_entity_list)
    # print(test_results)
    # print("\nTRAIN\n")
    # print(train_entity_list)
    # print(train_results)

    grand_score += score
    count+=1

    print(count)

print("f1_binary", calculate_f1_score(true_positive, false_positive, false_negative))

print("grand score", grand())