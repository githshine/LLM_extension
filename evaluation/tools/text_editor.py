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

      before_paragraph = (
          "A recent development has drawn widespread attention from both the public and experts, as new information emerges that could fundamentally reshape our understanding of the issue. "
          "Although the full picture is still unfolding, early reports suggest that the implications may be far-reaching, potentially affecting multiple sectors. "
          "The news has already sparked a wave of responses from researchers, policymakers, and concerned citizens, highlighting a wide range of perspectives and growing public interest. "
          "Some analysts believe this moment could mark a turning point—not only in how the matter is addressed going forward but also in how it is perceived by society as a whole. "
          "With uncertainty still surrounding many of the details, there are increasing calls for greater transparency, accountability, and thorough investigation into what led to this situation. "
          "As events continue to evolve, staying informed will be crucial. Here’s what we know so far—and why it matters in the days and weeks ahead."
          + "Smartphones can play a very positive role as a new tool with ophthalmologists, shows a new study conducted by the Emory University."
          + " \nSmartphones can be extensively used by ophthalmologists for viewing complex inner eye photos for diagnostic purpose. "
          "They can also be used for taking, sending and viewing pictures of the damage to the front of the eye or eyelids.\n"
          "The study was conducted among some 350 patients who reported issues like headaches, eyesight changes and similar vision problems in emergency rooms.\n"
          "The study also included the inner-eye photos taken by the ER staff using an ocular camera.\n"
          "Then the study team assessed the response of two ophthalmologists who viewed and rated the pictures on a desktop PC and did the same on an iPhone.\n"
          "The results of the study, published in the Archives of Ophthalmology, show that the doctors found the iPhone images good or better than desktop images and rated them high.\n"
          "While one ophthalmologist felt 53 pictures were of the same quality, he found 46 better on iPhone and just one was better on PC.\n"
          "For the other ophthalmologist who"
          + " I love how old movies portray New York City at Christmastime. Snow drifts gently to the ground. Carols are sung on street corners"
          + " . Decorations light up the streets. And the tree at Rockefeller Center seems to spread it's magic throughout the city, creating a scene straight out of the carol Silver Bells.\n"
          "“Children laughing, people passing, meeting smile after smile.” I love how old movies portray New York City at Christmastime.\n"
          "But I grew up in New York. And the city is nothing like that at Christmastime. The tree does light up Rockefeller Center and decorations line the streets. If you're lucky you'll get a nice snowfall. "
          "But what gives the city in those old films the spirit of Christmas is the people. The “children laughing” and the “smile after smile.” Ralph Kramden described it best in the Christmas episode of The Honeymooners, "
          "“everyone's hustling someplace. But they don't hustle around Christmastime like they usually do. You know they're a little more friendlier …"
          + " Whether it's sharing your Christmas wish list with Santa, enjoying the sounds of holiday music, or taking a trip back in time to celebrate the traditions"
          + " of the past, there will be plenty to do this holiday season. To help you celebrate we've rounded up a list of this year's happenings.\n"
          "Logan County Courthouse Lighting - The RE-1 Valley Children's Chorale will give a performance starting at 5:15 p.m., followed by the lighting of the courthouse at 5:30 p.m.\n"
          "Cocoa with Santa - 6 to 8 p.m., Christ United Methodist Church, 104 S. Fourth St. Parents make sure to take your own camera to snap of picture of your children with Santa. Reggie the Reindeer will be there too.\n"
          "Holiday Marketplace - Located at 313 Poplar St., the shop is open Nov. 24-Dec. 22, 4 to 7 p.m. Thursday/Fridays and 9 a.m. to 6 p.m. Saturdays.\n"
          "Parade of Lights - 6 p.m., downtown Sterling. This year's theme is \""
          + " Many people think the secret to great cooking is mastery of technique.\nIt helps, but it's hardly crucial. The key to cooking that tempt"
          + " s and satisfies, that brings people to the table, then brings them back for more, is understanding flavors and how they work together.\n"
          "And while a culinary degree certainly helps one understand this, more important is a willingness to try new foods, as well as old foods in new combinations. Now there is a book to help you take that flavorful trip.\n"
          "Flavor masters Karen Page and Andrew Dornenburg have compiled an encyclopedic primer to flavor. Their just-released \"The Flavor Bible\" not only explains what foods taste like, it also offers exhaustive lists of flavor pairings for each.\n"
          "They suggest mascarpone, for example, which goes nicely with almonds, ladyfingers and peaches, among many other options. They also suggest pairings to avoid, such as maple syrup and brown sugar (too intense).\n"
          "The first two sections of the book explain how flavor works and offer advice from chefs and others about how they pair various flavors to create great recipes.\n"
          + " The trial of Washington Post correspondent Jason Rezaian, on vague charges including espionage, began in Iran on Tuesday.\nOn Tuesday, the trial"
          + " of Washington Post reporter Jason Rezaian, who has been imprisoned for nearly 10 months in Iran on vaguely defined charges, started in Tehran.\n"
          "According to Iran’s official news agency IRNA, Rezaian is accused of committing \"espionage for the hostile government of the"
      )
      after_paragraph = (
        "As more information becomes available, updates will be provided to keep the public fully informed of any major developments. Authorities and subject-matter experts are continuing their investigations to better understand the full scope and underlying causes of the situation. Early findings have already raised important questions that could influence future discussions, both within policy circles and among the general public. Many analysts believe the outcome of this case could have ripple effects across related sectors, prompting a reassessment of existing practices or frameworks. Because of the issue’s complexity, perspectives may shift as new evidence emerges and public understanding deepens. Throughout this process, transparency and clear communication will be critical in maintaining trust and encouraging constructive dialogue. For now, the situation remains dynamic, and readers are encouraged to stay informed as the story continues to evolve."
        + "California is lifting its drought emergency for most of the state after a winter of record rain and snowfall that followed a five-year dry spell."
        + "\nGov. Jerry Brown's office announced Friday that his executive order will lift the drought emergency in California, except for Fresno, Kings, Tulare and Tuolumne counties. Those counties still face groundwater supply shortages.\n\"This drought emergency is over, but the next drought could be around the corner,\" Gov. Brown said. \"Conservation must remain a way of life.\"\nBrown's office also said new legislation will create long-term conservation measures as the state with a history of dry spells anticipates future droughts.\nAbout 8 percent of California is still under some type of drought, according to the most recent U.S. Drought Monitor report. At this time last year, more than 90 percent of California fell into at least one of the weekly report's four drought categories.\nMore than 31 percent of California was in the most severe category -- exceptional drought -- in April 2016. That figure dwindled to 18 percent during the height of winter's storms before falling away"
        + "Okay, okay, so maybe Delly wouldn\u2019t fall that far down the list. But still, a Matthew Dellavedova\u200b bi"
        + "opic? That seems\u2026random. And yet, it\u2019s apparently going to be real thing!\nAccording to Fox Sports Australia, Dellavedova and his Australian manager Bruce Kaider have teamed up with Los Angeles producers Zachary Green and Jason Shuman to create a film about Delly\u2019s life. It will focus on his upbringing in Australia and lead right up into the present day, which has taken Dellavedova to the magical city of...Milwaukee. No, but seriously, Kaider is promising that there will be plenty of intriguing stops along the way.\n\"Delly\u2019s inspirational story about overcoming the odds is one everyone can relate to,\" Kaider said in a press release, according to Fox Sports Australia. \"In real life, it played out just like a movie.\"\nDellavedova added that he\u2019s looking forward to getting started on the film. \"I am honored that Bruce, Zachary, and Jason think enough about my"
        + "CHICAGO -- A three-year-old boy died after a fire swept through his South Side home Sunday, police said.\nFire"
        + "fighters arrived at the three-story building on the 6700 block of South Dorchester Avenue two minutes after receiving calls of a fire, officials said. Three-year-old Maqkwone Jones was pulled from a second floor apartment and rushed to Comer Children's Hospital after suffering from cardiac arrest. He later died in the hospital.\nThe three-story building was mostly unoccupied at the time of the fire, according to the Chicago Fire Department, with people living in two of the 16 units. Thick, dark smoke could be seen from a distance as firefighters worked to stop the flames from jumping to neighboring buildings. In total, more than 100 firefighters joined in battling the blaze.\n\u201cThe flames were just everywhere. It looked like it was spreading from the building to the next building over,\" Ryan Booker, a neighbor, said.\nA man who lived in the building was treated for smoke inhalation, while a Chicago firefighter was taken to Northwestern Memorial Hospital as a precaution after showing"
        + "History is filled with stories of bold pioneers who changed the world with their visionary ideas. There exists also, in the annals of innovation, a"
        + " rich tradition of mad scientists with crazy inventions. These two notions are not mutually exclusive. In fact, many celebrated breakthroughs were initially met with scorn and ridicule. We've compiled 11 such world-famous ideas from the history of science and technology. The temptation, of course, is to include those ideas that were scorned and should have stayed scorned. The Star Wars prequels, say. But we're trying to run a classy operation here.\nHeliocentrism Galileo Galilei, sometimes called the Father of Modern Science, was among the first and most famous to pay a price for his crazy scientific ideas. Galileo publicly promoted the Copernican concept of heliocentrism -- that the Earth revolves around the Sun -- back when the Church and even most fellow mathematicians held to geocentric model. Unfortunately, in 17th century Rome, such radical ideas were quite literally heresy. Galileo was forced to recant and spend the rest of his life under house arrest.\n"
        + "As I alluded to previously, immigration reform seems to me like an issue where radical tactics, if people could be organized to engage in them, would"
        + " have high prospects for success. Among other things, Jose Antonio Vargas\u2019 story today makes that clear on a number of levels.\nOne is that people were willing to help him in a civilly disobedient way, starting with the woman at the DMV who whispered to him that his papers were fake rather than reporting him. Another is that whatever ultimately happens to Vargas, INS agents haven\u2019t rushed to his house to deport him ASAP. He\u2019s not currently sitting in a detention cell being questioned about what knowledge editors at the Washington Post and Huffington Post had of his immigration status. When Vargas pitched the story to the Post, they apparently rejected it, but they didn\u2019t turn around and immediately drop a dime on the guy.\nWhich is all just to say that the treatment of undocumented workers in this country is one of those things where we\u2019re only kinda sorta willing to enforce the law. Faced with a known quantity\u200a\u2014\u200a"
        + "The Democratic Alliance has been in the spotlight after the party's apparent mistaken vote for the Employment Equity Act Amendment Bill, with analysts saying the official opposition"
        + " is at an ideological crossroads.\nWe're talking to DA parliamentary leader and head of the opposition in Parliament Lindiwe Mazibuko about the issues facing the party, along with political analyst and PowerFM talk show host Eusebius McKaiser.\nSend us your questions here for the DA and join us on Friday.\nThe Bill, with its race-based elements, flies in the face of the DA's traditional opposition to racialised policy or legislation and angered its liberal members. On Friday, Helen Zille said the DA's vote was a mistake, due to a series of administrative errors, time constraints and misinformation, but the MP responsible for the vote said he stood by his decision.\nRead more about the saga here.\nWill the DA recover from this incident in time for next year's elections, and what does it reveal about the internal state of the party?\nJoin us live at 12.30am on Friday October 15, where the Mail & Guardian's"
      )

      # Split the before and after paragraphs into words
      before_words = before_paragraph.split()
      after_words = after_paragraph.split()

      # Calculate the number of words to take based on the ratio
      num_extra_words = int(len(word_list) * self.ratio * 0.5)

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
