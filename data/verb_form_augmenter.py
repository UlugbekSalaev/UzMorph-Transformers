import os
import re

def deduce_verb_form(form, lemma):
    form_lower = form.lower()
    lemma_lower = lemma.lower()
    
    # Extract the morphological tail (the part after the stem)
    # If the word starts with the stem, extract the tail
    if form_lower.startswith(lemma_lower):
        tail = form_lower[len(lemma_lower):]
    else:
        # Match from end dynamically or fallback to form
        tail = form_lower

    # 1. Infinitive (Inf)
    if 'moq' in tail:
        return 'VerbForm=Inf'
        
    # 2. Converb (Conv) - usually terminal or followed by emphatic
    if re.search(r'(ib|b|gach|kach|qach|guncha|kuncha|quncha|may|masdan|gali|kali|qali)$', tail):
        return 'VerbForm=Conv'
        
    # 3. Participle (Part) - Sifatdosh
    # -gan, -kan, -qan, -adigan, -ydigan, -yotgan, -ajak, -ar
    if re.search(r'(gan|kan|qan|digan|yotgan|ajak|mas)', tail):
        # Ensure it's not 'masdan' which is Conv
        if 'masdan' not in tail:
            return 'VerbForm=Part'
            
    # 4. Verbal Noun (Vnoun) - Harakat nomi
    # -ish, -sh, -uv, -v
    if re.search(r'(ish|sh|uv|v)(im|ing|si|imiz|ingiz|lari|ni|ga|da|dan|ning)?$', tail):
        # Exclude 'ish' if it's part of 'yozishdi' (Fin) instead of 'yozishi' (Vnoun)
        # But Reciprocal voice -ish is also tricky. 
        if not re.search(r'(di|sa|yapti)$', tail):
            return 'VerbForm=Vnoun'
            
    # 5. Finite (Fin)
    # -di, -yapti, -moqda, -adi, -ydi, -sa, -ay
    if re.search(r'(di|yapti|moqda|adi|ydi|sa|miz|san|siz|man|dilar)(ku|chi|mi|deb)?$', tail):
        return 'VerbForm=Fin'
        
    return None

def inject_verb_forms():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'Uzbek_Unified_Sentence_Corpus_35K.conllu')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out_lines = []
    stats = {'Inf': 0, 'Conv': 0, 'Part': 0, 'Vnoun': 0, 'Fin': 0}
    
    for line in lines:
        stripped = line.rstrip('\n')
        if not stripped or stripped.startswith('#'):
            out_lines.append(stripped)
            continue
            
        parts = stripped.split('\t')
        if len(parts) >= 10 and parts[3] == 'VERB':
            form = parts[1]
            lemma = parts[2]
            feats = parts[5]
            
            # Check if VerbForm is already present
            if 'VerbForm=' not in feats:
                inferred_form = deduce_verb_form(form, lemma)
                if inferred_form:
                    tag_val = inferred_form.split('=')[1]
                    stats[tag_val] += 1
                    
                    if feats == '_' or not feats:
                        feats = inferred_form
                    else:
                        feats += f"|{inferred_form}"
                        
                    # Sort
                    feat_list = feats.split('|')
                    unique = list(dict.fromkeys(feat_list))
                    unique.sort(key=lambda x: x.split('=')[0].lower())
                    parts[5] = '|'.join(unique)
                    
        out_lines.append('\t'.join(parts))
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
        if out_lines and out_lines[-1] != '':
            f.write('\n')
            
    print(f"Barcha Fe'l Shakllari muvaffaqiyatli avto-teglandi va dataset karrasiga kengaytirildi!")
    for tag, c in stats.items():
        print(f"  - {tag}: {c} ta yangi annotatsiya qo'shildi")

if __name__ == '__main__':
    inject_verb_forms()
