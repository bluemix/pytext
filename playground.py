import sys
import pytext

import json
import os


# with open('postgresql-data.json') as jsonfile:
#     poets_json = json.load(jsonfile)['snippets']
#
# with open('poets.json', 'w') as file:
#     poets = []
#     for poet in poets_json:
#         poets.append({'id': poet['id'], 'text': poet['text'].replace('\n', ' ')})
#
#     file.write(json.dumps(poets, ensure_ascii=False, indent=4))
#     file.close()


config_file = 'docnn.json'
model_file = 'models/model.caffe2.predictor'
text = 'مرحبا'

config = pytext.load_config(config_file)
predictor = pytext.create_predictor(config, model_file)

# Pass the inputs to PyText's prediction API
result = predictor({"text": text})

# Results is a list of output blob names and their scores.
# The blob names are different for joint models vs doc models
# Since this tutorial is for both, let's check which one we should look at.
doc_label_scores_prefix = (
    'scores:' if any(r.startswith('scores:') for r in result)
    else 'doc_scores:'
)

# For now let's just output the top document label!
best_doc_label = max(
    (label for label in result if label.startswith(doc_label_scores_prefix)),
    key=lambda label: result[label][0],
    # Strip the doc label prefix here
)[len(doc_label_scores_prefix):]

print('best_doc_label', best_doc_label)


