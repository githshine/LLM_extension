# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import re
import copy
import nltk
import torch
import random
import numpy as np
from tqdm import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import DiversityValueError
from evaluation.tools.oracle import QualityOracle
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM
import re

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text

class RandomWalkAttack(TextEditor):
    """
        Remove the watermark using the random walk attack (https://arxiv.org/abs/2311.04378) via black-box access to a quality oracle and a perturbaiton oracle.
        (1) Quality oracle can evaluate whether a candidate output is a high-quality response to a prompt.
        (2) Perturbation oracle can modify an output with a nontrivial probability of maintaining quality, 
            and which induces an efficiently mixing random walk on high-quality outputs.
        
        Examplar Usage: 
        '''
        model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto') 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xl", device_map='auto')
        perturbation_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
        quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
        span_length = 6
        attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                  quality_oracle=quality_oracle,
                                  max_new_tokens=int(2*span_length), min_length=int(1.5*span_length), 
                                  do_sample=True, top_p=0.95, top_k=None, repetition_penalty=1.5)
        '''
    """

    def __init__(self, perturbation_tokenizer: T5Tokenizer, perturbation_oracle: T5ForConditionalGeneration, quality_oracle: QualityOracle,
                       device='cuda', total_steps=200, span_len=6, target_valid_steps=100, **kwargs):
        """
            Parameters:
            perturbation_tokenizer (T5Tokenizer): The tokenizer for the perturbation oracle.
            perturbation_oracle (T5ForConditionalGeneration): The perturbation oracle.
            quality_oracle (QualityOracle): The quality oracle.
            device (str): The device to use for inference.
            span_len (int): The length of the span to mask in each random walk step.
            total_steps (int): The total number of random walk steps.
            target_valid_steps (int): The target number of valid steps.
        """
        self.perturbation_tokenizer = perturbation_tokenizer
        self.perturbation_oracle = perturbation_oracle.eval()
        self.quality_oracle = quality_oracle
        self.device = device
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)
        
        self.span_len = span_len
        self.total_steps = total_steps
        self.target_valid_steps = target_valid_steps
        if self.quality_oracle.check_quality == 'checker':
            from gramformer import Gramformer
            self.gf = Gramformer(models = 1, use_gpu=True)

    def perturb(self, text: str):
        final_input_text = self.mask_text(text)

        # Tokenize the input
        final_input = self.perturbation_tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the edited text
        with torch.inference_mode():
            outputs = self.perturbation_oracle.generate(**final_input, **self.gen_kwargs)
        outputs = self.perturbation_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        infilled_text = outputs[0]
        final_output_text = final_input_text.replace('<extra_id_0>', infilled_text)

        return final_output_text

    def edit(self, text: str, prompt: str, backtrack_patience: int = 100, max_attempts: int = 1000):
        """Edit the text using the T5 model."""

        original_response, n_response = text, text
        n_iter, valid_steps = 0, 0
        patience = 0
        cached_response = copy.deepcopy(n_response)
        # Process the input text in sentence windows
        pbar = tqdm(total=None)
        while n_iter < self.total_steps or valid_steps < self.target_valid_steps:
            candidate_response = self.perturb(n_response)

            candidate_response = self.grammatical_error_correction(candidate_response)
            candidate_response = self.remove_incomplete_sentences(candidate_response)
            
            if self.quality_oracle.maintain_quality(prompt, original_response, candidate_response):
                cached_response = n_response
                n_response = candidate_response
                valid_steps += 1
                if valid_steps % 10 == 0:
                    print(f"Original response: {original_response}")
                print(f"Get a better {valid_steps}-th response at step {n_iter}/{self.total_steps}: {n_response}")
                patience = 0
            else:
                patience += 1
            
            if patience > max_attempts:
                break
            elif patience > backtrack_patience:
                n_response = cached_response
                patience = 0
            
            pbar.update(1)
            n_iter += 1
        pbar.close()

        return n_response

    def grammatical_error_correction(self, text):
        sentences = sent_tokenize(text)
        corrected_sents = []
        for sent in sentences:
            corrected_sent = self.gf.correct(sent, max_candidates=1).pop()
            corrected_sents.append(corrected_sent)
        corrected_text = ' '.join(corrected_sents)
        return corrected_text

    def mask_text(self, text):
        words = text.replace('\n', ' \n').split(' ')
        if len(words) == 1:
            return text + ' <extra_id_0> '
        start = np.random.randint(0, len(words) - self.span_len)
        end = start + self.span_len
        masked_text = ' '.join(words[:start]) + ' <extra_id_0> ' + ' '.join(words[end:])
        return masked_text
    
    def contains_verb(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        return any(tag.startswith('VB') for word, tag in tagged_words)

    def remove_incomplete_sentences(self, text):
        sentences = sent_tokenize(text)
        complete_sentences = []
        for sent in sentences:
            if sent.endswith('.') and not self.contains_verb(sent) and not bool(re.match(r'^\d+\.$', sent)):
                continue
            else:
                complete_sentences.append(sent)
        return ' '.join(complete_sentences)

    def correct_text(self, text):
        """Basic punctuation correction"""
        # Replace multiple spaces with a single space
        corrected_text = re.sub(r'\s+', ' ', text)

        # Correct spaces before commas, periods, colons, semicolons, exclamation marks, and question marks
        corrected_text = re.sub(r'\s+([,.;!?])', r'\1', corrected_text)  # Remove space before punctuation
        corrected_text = re.sub(r'([,.;!?])(?!\s)', r'\1 ', corrected_text)  # Ensure space after punctuation if missing

        # Replace multiple occurrences of punctuation marks with a single instance
        # This part targets specific punctuation marks (you can add more as needed)
        corrected_text = re.sub(r'(\.){2,}', '.', corrected_text)
        corrected_text = re.sub(r'(,){2,}', ',', corrected_text)
        corrected_text = re.sub(r'(!){2,}', '!', corrected_text)
        corrected_text = re.sub(r'(\?){2,}', '?', corrected_text)
        corrected_text = re.sub(r'(:){2,}', ':', corrected_text)
        corrected_text = re.sub(r'(;){2,}', ';', corrected_text)

        return corrected_text


class TranslateAttack(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, 
                 src_lang="eng_Latn", tgt_lang="zho_Hans", batch_size=2, device='cuda', **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (AutoTokenizer): The tokenizer for the DIPPER model.
                model (AutoModelForSeq2SeqLM): The DIPPER model.
                device (str): The device to use for inference.
                src_lang: Source language.
                tgt_lang: Target language.
                batch_size: default is 2, means every time 2 sentences will be sent to translate.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

    def translate_batch(self, text: str, src_lang, tgt_lang):
        """Translate the text using the facebook/nllb-200 model. Default from English to Chinese."""
        self.tokenizer.src_lang = src_lang
        results = []

        if src_lang == 'eng_Latn':  # 英文
          sentences = sent_tokenize(text)  # 按句子分割
        elif src_lang == 'zho_Hans' :  # 中文简体
          sentences = re.split(r'(?<=[。！？])', text)
          sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            batch_text = " ".join(batch)  # 合并句子
            inputs = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            # inputs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang]
            # 设置目标语言的语言 ID
            if hasattr(self.tokenizer, "lang_code_to_id"):
                inputs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang]
            else:
                # 手动设置语言 ID
                lang_code_to_id = {
                    "eng_Latn": 128001,  # 英语
                    "zho_Hans": 128002,  # 简体中文
                }
                inputs["forced_bos_token_id"] = lang_code_to_id[tgt_lang]

            outputs = self.model.generate(**inputs, max_length=512)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
        return " ".join(results) # 输出 字符串
    
    def edit(self, text: str, reference=None):
        """Translate the text using the facebook/nllb-200 model. Default from English to Chinese."""
        translated_to_chinese = self.translate_batch(text, src_lang="eng_Latn", tgt_lang="zho_Hans")
        translated_to_english = self.translate_batch(translated_to_chinese, src_lang="zho_Hans", tgt_lang="eng_Latn")
        return translated_to_english

    
class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content="Your are a helpful assistant to rewrite the text.")
        paraphrased_text = openai_util.get_result(self.prompt + text)
        return paraphrased_text


class DipperParaphraser(TextEditor):
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")
    
    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0,5,10, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)
        
        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        
        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())
        
        output_text = ""
        
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])
            
            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            
            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            
            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text

class CopyPasteAttack(TextEditor):
    '''Copy watermarked text into unwatermarked text, and then detect the combined text'''
    def __init__(self, ratio: float) -> None:
        """
            Initialize the copy-paste attack editor.

            Parameters:
                ratio (float): The ratio of words to paste.
        """
        self.ratio = ratio

        

    def edit(self, text: str, reference=None):
      """Copy original text into extra text. The length of the extra text is determined by the ratio."""

      # Handle empty string input
      if not text:  
        return text

      # Split the text into words
      word_list = text.split()

      before_paragraph = """
                    Economy and finance – Economic downturn

                    In the mid-1430s, a sixty-year cold period began in the Northern Hemisphere, accompanied by sporadic floods and droughts that resulted in crop failures, leading to famines and epidemics. China was also struck by a series of natural disasters in the late 1430s and 1440s, with floods, droughts, epidemics, and famines occurring in succession. In 1448, the Yellow River breached its dams, causing the waters to flow into northern Jiangsu. The following year, another dam broke, diverting part of the Yellow River's flow into the Guo River and then the Huai River, eventually reaching the sea in southern Jiangsu. Flooding persisted into the 1450s, and the changing course of the Yellow River posed a threat to the water supply of Beijing.

                    The government attempted to assist the victims by remitting taxes in large quantities, particularly during the regency of Grand Empress Dowager Zhang, who consistently showed concern for the impoverished, but despite these efforts, dissatisfaction among the population continued to grow. This was largely due to the compulsory work system, which placed an unbearable burden on the people in some regions. As a result, artisans evaded state demands and peasants abandoned their land, leading to a significant decrease in population in certain areas. On the other hand, bandits and vagabonds multiplied.

                    The economic decline in China from the early 1440s to the mid-1460s resulted in a decrease in porcelain production, particularly for export. The emperor's prohibition on the private sale of blue-and-white porcelain in 1439, intended to safeguard the governmental monopoly, set further limits on production. In fact, in January 1448, the ban was reinforced and extended to forbid the production of porcelain in any colors other than blue and white for private sale in Raozhou Prefecture, where Jingdezhen, known for its porcelain, is located. These prohibitions may have been one of the reasons for the scarcity of porcelain from the Zhengtong, Jingtai and Tianshun eras (1436–1464).

                    The "Three Yangs" responded to the economic problems by cutting state spending. This was made worse by the struggles in the southwest during the 1430s and 1440s, which led to a decline in mining in the region. As a result, they cancelled overseas expeditions and restricted official foreign trade. These austerity measures were easier for them to accept because they directly affected the economic power of the eunuchs in the imperial palace, who were competing with other groups for power. The eunuchs were the ones involved in maritime expeditions and had a vested interest in silver mining, which was also limited in the mid-1430s.

                    Currency – The recognition of silver

                    By the end of the Xuande era, the government had recognized the failure to enforce baochao banknotes as the main currency and began tolerating silver. In 1433, Governor of South Zhili, Zhou Chen (周忱), introduced the payment of land tax in silver instead of rice in the most tax-burdened prefectures. From 1436, the officers of the Beijing garrison were paid in silver. In the same year, the land tax in South Zhili, Zhejiang, Jiangxi, and Huguang was also converted to silver; this transition was accompanied by a tax cut. According to historian Ray Huang, this was a concession to southern landowners and a reversal from the Hongwu Emperor's policy of suppressing the influence of wealthy landowners. Another historian, Richard Von Glahn, believes that it was an attempt to get the rich people's silver out of their coffers. Additionally, the government reduced silver mining to a minimum.

                    The surviving land sales contracts concluded in Huizhou from 1368 to 1500 demonstrate the complex search for the most suitable currency during the early Ming period. Initially, prices were set in silver until 1375, after which baochao banknotes were predominantly used until 1425, but there were instances where the price was set in grain from 1396 to 1436, and during the Xuande era (1426–1435), cloth was the preferred currency for price determination. Eventually, silver emerged as the clear winner, as all land contracts from 1456 to 1644 were priced in it.

                    After Wang Zhen gained influence in the government, the eunuchs pushed for the reopening of the silver mines under their supervision, but due to the low productivity of mining and the high demands of the eunuchs, there were a series of mining uprisings in Fujian, Zhejiang, and Jiangxi. After Emperor Yingzong was captured in a war with the Mongols in 1449, the new government restricted mining once again, but when Emperor Yingzong returned to power in 1457, the restrictions were lifted. Despite this, mining yields remained low.

                    The government's decision to allow payment in silver resulted in the rapid decline of banknotes, much to the dismay of the statesmen. By the 1430s, banknotes had practically disappeared from use, with the state only using them to pay employees to a limited extent and withdrawing them as a compulsory payment for trade fees. However, these small transactions were relatively isolated from the country's economy. While silver was used for large payments and taxes, copper coins remained the dominant currency for small transactions in cities.

                    Closure of state mints

                    In 1436, the Minister of Revenue proposed to buy out old banknotes and replace them with new ones covered in silver, but this proposal was ultimately unsuccessful. Around the same time, the government began to tolerate the use of coins in commerce, although their prohibition was not consistently enforced even before this time. While the use of coins was officially not allowed until 1436, in response to a petition from a prefect of Guangxi, the government had actually stopped the production of coins in either 1433 or 1436.

                    The lifting of the ban on the use of silver and copper coins in trade is a good example of the functioning of Ming legislation. Changes to laws were typically made based on petitions from mid-level officials (such as prefects) who requested exceptions for their areas of jurisdiction. After the emperor's approval, which was published in the official gazette, such exceptions could be seen by other authorities as a precedent for establishing a new procedure and could be further expanded based on analogy. In this case, a request for permission to use copper coins as currency in one prefecture led to the legalization of not only copper coins, but also silver, throughout the country.

                    With the closure of the mints, the shortage of coins worsened over time. Entrepreneurs responded to the demand for coins by producing them privately, which was illegal. Despite the efforts of disgruntled officials in Beijing, they were unable to suppress this private production, but they also failed to take measures to restore the state's coin production.

                    In the northern cities, particularly Beijing, coins were the primary form of currency during the 15th century. This led officials to criticize them as the reason for the failure of banknotes. In 1447, the Governor of North Zhili called for a renewed campaign against coinage, citing its exclusive use in trade in Beijing and the Grand Canal cities as the cause of the banknotes' failure. Despite efforts by his successor to lift the ban, the Ministry of Revenue continued to prohibit coinage until 1453. By the mid-1450s, private coins from Jiangnan had become more prevalent in the markets of Beijing, replacing Yongle's coins. Suggestions to combat private coinage by opening state mints were rejected, leading to the proliferation of illegal mints. These private coins were of lower quality, often containing tin or iron, but due to the scarcity of old coins, merchants had no choice but to use them, even at a nominal value. Some merchants refused to accept Ming coins altogether, while others only accepted silver. The shortage of currency resulted in a return to barter in certain regions, including Yunnan, Guizhou, Sichuan, Jiangxi, Huguang, Shaanxi, and Shanxi.

                    Private mints in Ming China also had an impact on foreign trade, as their coins were accepted as currency in other countries, despite the Chinese government's refusal to recognize them. The closure of these mints had far-reaching consequences, causing problems in places like Japan and Java. Japan, which had not minted coins since the 10th century, relied heavily on imports from China. The disruption of this supply in the early 15th century had a significant impact on the Japanese economy and even led to political turmoil, resulting in the division of Japan into competing domains during the Sengoku period.
                    """



      after_paragraph = """
                    Fighting in the South
                    War in the Southwest
                    Further information: Luchuan–Pingmian campaigns
                    In the first quarter of the 15th century, on the southwestern borders of the Ming dynasty, one of the Shan states, Möng Mao, called Luchuan by the Ming, grew in strength under the rule of the ambitious Si Renfa, who ruled from 1413. By 1436, Si Renfa had begun to pose a threat to Ming positions in the area. In 1438, Mu Sheng, the military commander of Yunnan, was ordered to attack Möng Mao with an army of conscripts from Guizhou and Huguang. Initially, the Ming army was successful in defeating the enemy, but they soon encountered supply problems and struggled to adapt to the subtropical climate. As a result, the weakened Ming army suffered a heavy defeat in 1440.

                    Wang Zhen believed that Grand Empress Dowager Zhang's tax policy was too lenient and saw the war as an opportunity to increase state revenue. As a result, he pushed for a new campaign to be launched. Reinforcements were sent from Sichuan, Guizhou, and Huguang to Yunnan, and in early 1441, Minister of War Wang Ji was placed in overall command. Wang Ji was an experienced civil official who had held the position of minister of war since 1435. He had also commanded the second to fourth campaigns in the Luchuan–Pingmian campaigns from 1441 to 1449. This was the first time in the history of the Ming dynasty that a civil official was given supreme command of the troops. Under Wang Ji's leadership, a Ming army of 50,000 soldiers successfully defeated the Shans. Si Renfa fled to Ava, a Burmese kingdom, and the territory of Luchuan was divided among other Shan states. As a reward for his success, Wang Ji was promoted to the rank of Count of Jingyuan, and his deputy Xu Xi took over as minister of war. Any criticism that resources were being drained from the North to fund the war in the South was suppressed.

                    In 1443–1444, the war continued with Ming troops unsuccessfully fighting against Ava, but in 1445, Ava surrendered, and Si Renfa committed suicide. Another campaign took place in 1448–1449, during which the Chinese and Ava successfully defeated Si Renfa's son, Si Jifa, who resided in Mong Yang west of the Irrawaddy River. In March 1449, Emperor Yingzong celebrated the victory.

                    These wars ultimately solidified Ming power in Yunnan, but at a high cost. Local rulers acknowledged Ming sovereignty and paid tribute to Beijing until the 16th century. Domestically, these wars were a success for Wang Zhen, increasing his prestige and reputation as a statesman, but they also revealed a lack of financial reserves and experienced generals on the northern frontier.

                    Rebellion in the Southeast
                    Further information: Deng Maoqi rebellion
                    In response to the increased demand for silver due to the implementation of silver-based taxes in 1436, the government took action by shutting down silver mines and prohibiting small-scale silver mining along the border of Zhejiang and Fujian two years later. However, in an area with a high population and limited job opportunities, illegal silver mining persisted. In 1447, the leader of a group of silver miners in the mountains between Zhejiang and Fujian openly rebelled, gathering followers and forming an army.

                    In the interior of Fujian, two brothers, Deng Maoqi and Deng Maoba, opposed the exploitation of tenants. The tenants themselves demanded that landlords cancel payments beyond the scope of their leases. In March 1448, the Deng brothers rebelled and began to conquer one county after another. The government attempted to calm the situation by forgiving unpaid taxes and granting a three-year exemption from compulsory labor for the population in the region, but the more radical faction of the rebels, numbering several hundred thousand men, refused to back down. The local militia was unable to handle the situation, prompting the government to send an army of 50,000 to the southeast in September 1448.

                    In late 1448, the rebel miners were defeated by troops on the border between Fujian and Jiangsu. The Deng brothers were captured in February 1449, and their successors were defeated in May of the same year. According to Japanese historian Tanaka Masayoshi, the Deng brothers' revolt was the first Chinese peasant uprising aimed at challenging class relations within the village. The miners' revolt was ultimately suppressed by August 1449, and the remaining Fujian rebels were dispersed by 1452.

                    According to Fang Chao-ying, the emperor's successes in the southeast and southwest may have led him to overestimate the strength of the Ming troops and his own willingness to personally lead the army.

                    Trouble in the North
                    Relations with the Mongols
                    The Mongols were divided into three main groups: the Uriankhai in the southeast, the Eastern Mongols (also known as Tatars) in the east, and the Oirats in the west. In 1434, the leader of the Eastern Mongols, Arughtai, was defeated in battle by the Oirats. This gave the Oirats control over Mongolia, and their chief Toghon solidified their power by arranging for his daughter to marry the young Khan of the Eastern Mongols, Toghtoa Bukha. According to historian Philip de Heer, Toghon emerged as the "de facto ruler of all Mongols", and after his death in 1440, his son Esen took over his position. Esen was more ambitious than his father, and in 1443 and 1445, he launched attacks on Hami, an important city on the route from China to Central Asia near the Chinese border. In 1448, he successfully conquered it. He also attempted to gain the support of the Mongol divisions in the Ming army in western Gansu. In the east, his authority extended to the borders of Korea. In Beijing, the unification of Mongolia was perceived as a threat by Wang Zhen's opponents.

                    The Mongols were primarily interested in free trade with China, specifically exchanging horses for luxury goods such as tea and silk. Some Mongols who resided along the border relied on agriculture for their livelihood and sought support from the Ming authorities, but the Ming government focused on trading tea for horses in Gansu, with tribes in present-day Qinghai, rather than on the border with the Mongols. This trade involved exchanging a million jin (258 tons) of tea for 14 thousand horses every three years. The Ming authorities tightly regulated and restricted trade with the Mongols, with Wang Zhen overseeing the profitable trade through a network of eunuch-trustees in border towns.

                    As Esen's power grew, so did his need for goods in order to maintain the loyalty of the Mongol tribes. This resulted in protests from the Chinese, who were concerned about the increasing influx of Mongols. By the late 1440s, up to two thousand Mongols were arriving in Datong, the main trading center, every year. The presence of such large groups of armed horsemen posed a security threat. This caused the Ming authorities to become increasingly hostile and fearful. In 1449, the Mongols were only given a fifth of the required goods, which led them to resort to force. The immediate cause of the war was Ming's refusal to grant Esen's request to marry an imperial princess for his son.

                    Defense of the Northeast
                    After the death of the Yongle Emperor, the state of defense along the northern borders began to deteriorate gradually. The quality of training, as well as the weapons and equipment, were declining. In fact, soldiers from the Beijing garrison were even being used for the construction of government and private buildings. At this time, the Great Wall had not yet been built, and the border was only guarded by patrol battalions. These battalions were expected to hold off the enemy until the main forces arrived. The main forces were located in three fortified cities—Xuanfu, Datong, and Beijing—each housing several tens of thousands of soldiers. The largest force, consisting of 160,000 men, was stationed in Beijing. The reserves were scattered throughout northeastern China, in North Zhili, Shandong, and Henan. Since Xuanfu was less than 200 km from Beijing, the defense system lacked depth and relied on a quick and decisive response to any potential attack.
                    """


      # Split the before and after paragraphs into words
      before_words = before_paragraph.split()
      after_words = after_paragraph.split()

      # Calculate the number of words to take based on the ratio
      num_extra_words = int(len(word_list) * self.ratio / 2)

      # Handle cases where num_extra_words is greater than the length of before_words or after_words
      if num_extra_words > len(before_words):
          before_excerpt = (before_words * ((num_extra_words // len(before_words)) + 1))[-num_extra_words:]
      else:
          before_excerpt = before_words[-num_extra_words:]

      if num_extra_words > len(after_words):
          after_excerpt = (after_words * ((num_extra_words // len(after_words)) + 1))[-num_extra_words:]
      else:
          after_excerpt = after_words[-num_extra_words:]

      # Add the excerpts to the text
      edited_text = ' '.join(before_excerpt) + ' ' + ' '.join(word_list) + ' ' + ' '.join(after_excerpt)

      return edited_text
       

class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        replace_details = []  # Array to store replace indices and predicted tokens

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1

            # Record the replace index and predicted token
            replace_details.append((i, predicted_tokens[0]))

        replaced_text = ' '.join(words)

        # Print the replace details
        print("ContextAwareSynonymSubstitution --- Replace Details:\n", replace_details)

        print('Text before substitution: \n')
        print(text)
        print('Text after substitution: \n')
        print(replaced_text)

        return replaced_text


class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the prompt from the text."""
        if reference is not None:
            truncated_text = ' '.join(text.split()[len(reference.split()):])
            return truncated_text
        else:
            return text


class TruncateTaskTextEditor(TextEditor):
    """Truncate the task description from the text, used in code generation."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the task description from the text."""
        if reference is not None:
            truncated_text = text[len(reference):]
            return truncated_text
        else:
            return text
        

class CodeGenerationTextEditor(TextEditor):
    """Process the code generation output, removing the extra parts."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Process the code generation output, removing the extra parts."""
        text = text.lstrip("\n")
        text = text.split("\n\n")[0]
        return text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                 translate_to_source = Translator(from_lang="zh", to_lang="en").translate) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result
