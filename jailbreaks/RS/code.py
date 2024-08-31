import json

jb = json.load(open('raw.json', 'r'))
jailbreaks = jb['jailbreaks']
output_file = 'rs_llama2_7b_chat.json'

output_content = []
for jb in jailbreaks:
    jb['content'] = jb['prompt']
    output_content.append(jb)
    
with open(output_file, 'w') as f:
    json.dump(output_content, f)
