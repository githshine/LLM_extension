from watermark.sir.sir import SIR, SIRLogitsProcessor, SIRConfig, SIRUtils
from watermark.synthid.synthid import SynthID, SynthIDLogitsProcessor, SynthIDConfig, SynthIDUtils

from watermark.base import BaseWatermark, BaseConfig
from functools import partial
import jieba
from transformers import LogitsProcessorList
from utils.transformers_config import TransformersConfig

from visualize.data_for_visualization import DataForVisualization


# yese fhsfifeogkgnjgfgjk/glrgl
class test:
    def func1():
        print("test!")

class SIR_SynthID(BaseWatermark):
    """Top-level class for SIR_SynthID algorithm."""

    def __init__(self, algorithm_config_sir: str | SIRConfig, algorithm_config_synthid: str | SynthIDConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the SIR algorithm.  

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config_sir, str):
            self.config_sir = SIRConfig(algorithm_config_sir, transformers_config)
        elif isinstance(algorithm_config_sir, SIRConfig):
            self.config_sir = algorithm_config_sir
        else:
            raise TypeError("algorithm_config_sir must be either a path string or a SIRConfig instance")
        if isinstance(algorithm_config_synthid, str):
            self.config_synthid = SynthIDConfig(algorithm_config_synthid, transformers_config)
        elif isinstance(algorithm_config_synthid, SynthIDConfig):
            self.config_synthid = algorithm_config_synthid
        else:
            raise TypeError("algorithm_config_synthid must be either a path string or a SynthIDConfig instance")
        
        self.utils_sir = SIRUtils(self.config_sir)
        self.utils_synthid = SynthIDUtils(self.config_synthid)
        self.logits_processor_sir = SIRLogitsProcessor(self.config_sir, self.utils_sir)
        self.logits_processor_synthid = SynthIDLogitsProcessor(self.config_synthid, self.utils_synthid)

        self.detector_sir = SIR(algorithm_config_sir, transformers_config, *args, **kwargs)
        self.detector_synthid = SynthID(algorithm_config_synthid, transformers_config, *args, **kwargs)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor_sir, self.logits_processor_synthid]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    '''改变 delta 的值 (0.2 -> 0.8), 画出不同值下的 AOC 曲线 TODO''' 
    def detect_watermark(self, text: str, delta: float = 0.5, zscore_threshold : float = 0.5, return_dict: bool = True, *args, **kwargs):
      """Detect watermark using combined SIR + SynthID score."""
      '''
      Args:
        delta:
          delta = 0.0 → 完全使用 SIR 检测
          delta = 0.5 → 平均融合
          delta = 1.0 → 完全使用 SynthID 检测

        zscore_threshold:
          default value is 0.5
          can get via making False Positive Possibility as 0.1% or 0.5% ??
          or just by (1- delta) * sit_threshold + delta * synthid_threshold

      '''

      _, sir_score = self.detector_sir.detect_watermark(text)      # ∈ [-1, 1]
      _, synthid_score = self.detector_synthid.detect_watermark(text)  # ∈ [0, 1]

      # Normalize sir_score to [0, 1]
      sir_score_norm = (sir_score + 1) / 2

      # Weighted combination
      combined_score = (1 - delta) * sir_score_norm + delta * synthid_score

      # Threshold could be configurable or set empirically
      is_watermarked = combined_score > zscore_threshold

      if return_dict:
          return {
              "is_watermarked": is_watermarked,
              "combined_score": combined_score,
              "sir_score": sir_score,
              "synthid_score": synthid_score,
              "delta": delta
          }
      else:
          return is_watermarked, combined_score







    '''原始的 SIR 的detection 函数'''
    # def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
    #     """Detect watermark in the input text."""

    #     # Split the text into a 2D array of words
    #     word_2d = self.utils.get_text_split(text)

    #     # Initialize a list to hold all computed values for similarity
    #     all_value = []

    #     # Iterate over each sentence in the split text, skipping the first
    #     for i in range(1, len(word_2d)):
    #         # Create the context sentence from all previous text portions
    #         context_sentence = ''.join([''.join(group) for group in word_2d[:i]]).strip()
    #         # Current sentence to check against the context
    #         current_sentence = ''.join(word_2d[i]).strip()

    #         # Continue if the context sentence is shorter than the required chunk length
    #         if len(list(jieba.cut(context_sentence))) < self.config.chunk_length:
    #             continue

    #         # Get embedding of the context sentence
    #         context_embedding = self.utils.get_embedding(context_sentence)
    #         # Transform the embedding using the model, process output
    #         output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()
    #         # Scale the output vector and map to predefined indices
    #         similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

    #         # Encode the current sentence into tokens
    #         tokens = self.config.generation_tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)

    #         # Append negative similarity values for each token in the current sentence
    #         for index in tokens[0]:
    #             all_value.append(-float(similarity_array[index]))

    #     # Calculate the mean of all similarity values
    #     z_score = np.mean(all_value)

    #     # Determine if the z_score indicates a watermark
    #     is_watermarked = z_score > self.config.z_threshold

    #     # Return results based on the return_dict flag
    #     if return_dict:
    #         return {"is_watermarked": is_watermarked, "score": z_score}
    #     else:
    #         return (is_watermarked, z_score)
        
        # 新函数 暂不提供 数据可视化
    # def get_data_for_visualization(self, text: str, *args, **kwargs):
    #     """Get data for visualization."""
        
    #     # Split the text into 2D array of words
    #     word_2d = self.utils.get_text_split(text)
    #     highlight_values = []
    #     decoded_tokens = []

    #     # Iterate over each sentence in the text
    #     for i in range(len(word_2d)):
    #         # Construct the context sentence from the previous sentences
    #         context_sentence = ' '.join([' '.join(group) for group in word_2d[:i]])
    #         # Current sentence for tokenization
    #         current_sentence = ' '.join(word_2d[i])
    #         # Tokenize the current sentence
    #         tokens = self.config.generation_tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)

    #         # Decode each token and append to the decoded_tokens list
    #         for token_id in tokens[0]:
    #             token = self.config.generation_tokenizer.decode(token_id.item())
    #             decoded_tokens.append(token)

    #         # If the context sentence is shorter than required, append highlight -1 for each token
    #         if len(context_sentence.split()) < self.config.chunk_length:
    #             highlight_values.extend([-1] * len(tokens[0]))
    #             continue

    #         # Get the embedding of the context sentence and process it through the model
    #         context_embedding = self.utils.get_embedding(context_sentence)
    #         output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()

    #         # Scale the output vector and get similarity values
    #         similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

    #         # Append highlight values based on similarity
    #         for token_index in tokens[0]:
    #             similarity_value = -float(similarity_array[token_index.item()])
    #             highlight_values.append(1 if similarity_value > 0 else 0)

    #     return DataForVisualization(decoded_tokens, highlight_values)