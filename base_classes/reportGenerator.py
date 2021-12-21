
import logging

import tensorflow as tf
import numpy as np
import re
import random

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

from base_classes.transformer import start_token, end_token, tokenize, pure_tokenize, detokenize

class ReportGenerator():
    def __init__(self, transformer):
        # if not isinstance(transformer, Transformer):
        #     raise TypeError("transformer must be a Transformer object")
        self.transformer = transformer
        self.predictions = None
        self.encoder_input = None
        self.output_array = None
        self.output = None
        self.current_token_num = None
        self.is_output_updated = True


    def debug_get_highest_prob_kmer(self, kmer_probs):
        predicted_kmer_token = tf.argmax(kmer_probs, axis=-1)
        predicted_kmer = self.detokenize_sequence(predicted_kmer_token)[0][0].numpy().decode('UTF-8')
        return predicted_kmer

    def debug_get_n_highest_prob_kmers(self, kmer_probs, n):
        assert n >= 1
        results = tf.math.top_k(kmer_probs, k=n)
        return detokenize(tf.cast(results.indices, dtype=tf.int64))

    def start_word_prediction(self):
        self.output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        # print(tf.constant([start_token]))
        self.output_array = self.output_array.write(0, tf.constant([start_token]))
        self.current_token_num = 1 # Accounts for the initial starting token
        self.encoder_input = tf.constant([start_token, end_token])[tf.newaxis]
        

    def get_next_word_probabilities(self):
        # Only do prediction if the output has been updated since previous prediction
        if self.is_output_updated:
            # Have transformer generate kmer probabilities based on current output
            self.output = tf.transpose(self.output_array.stack())
            predictions, _ = self.transformer([self.encoder_input, self.output], is_training=False)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            probabilities = tf.nn.softmax(predictions)

            self.is_output_updated = False
            return probabilities
            # return self.predictions_to_dict(predictions)
        else:
            print("Cannot generate next word without updating outputs")
            return None

        
    # def predictions_to_dict(self, predictions):
    #     # Returns a dict of Kmer(str tensor) -> probability
    #     prediction_dict = {}
    #     for idx in range(tf.size(predictions)):
    #         prediction_dict[self.reverse_kmer_lookup.lookup(idx)] = predictions[idx]
    #     return prediction_dict


    def feedback_next_word(self, next_word):
        # Update the transformer output with actual next kmer
        if self.is_output_updated:
            print("Warning, updating kmer outputs before predicting next kmer")
        # kmer_tensor = tf.squeeze(self.kmer_lookup.lookup(next_kmer), axis=[0])
        print(next_word)
        print(pure_tokenize(tf.squeeze(next_word, [0, 1])))
        # print(tf.squeeze(pure_tokenize(next_word), [0, 1]))
        print(tf.reshape(pure_tokenize(next_word).to_tensor(), [-1]))
        word_tensor = tf.cast(tf.reshape(pure_tokenize(next_word).to_tensor(), [-1]), dtype=tf.int32)
        # print(tf.squeeze(word_tensor, [0]))
        self.output_array = self.output_array.write(self.current_token_num, word_tensor)
        self.current_token_num += 1
        self.is_output_updated = True

    def feedback_next_word_tok(self, tok):
         # Update the transformer output with actual next kmer
        if self.is_output_updated:
            print("Warning, updating kmer outputs before predicting next kmer")
        tok = tf.squeeze(tok, [0])
        self.output_array = self.output_array.write(self.current_token_num, tok)
        self.current_token_num += 1
        self.is_output_updated = True

    def token_tensor_to_report(self, tokens):
        report_list = detokenize(tokens).flat_values.numpy()
        raw_report = ""
        for word in report_list:
            raw_report += word.decode("utf-8") + " "
        raw_report = raw_report.replace("[START]", "")
        raw_report = raw_report.replace("[END]", "")
        report = self.beautify_raw_report(raw_report)
        return report

    def beautify_raw_report(self, raw_report):
        report = raw_report.strip()
        delete_space_before_and_after = ["'", "/", "\\",  "-"]
        delete_space_before = [",", ".", ":", ";"]
        to_cap = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        hyphenated_compounds = ["like", "shaped", "shape", "moving", "ish"]

        # Delete spaces around certain punctuation
        for punct in delete_space_before:
            report = report.replace(" "+punct, punct)
        for punct in delete_space_before_and_after:
            report = report.replace(" "+punct, punct)
        for punct in delete_space_before_and_after:
            report = report.replace(punct+" ", punct)
        # Clean times (delete spaces around colons surrounded by numbers)
        match = re.search(r'[0-9]: [0-9]', report)
        while match:
            report = report[:match.start()+2] + report[match.end()-1:]
            match = re.search(r'[0-9]: [0-9]', report)
        # Clean references of "<word>- like" object to "<word>-like". Also for other hyphenated compound words
        # for word in hyphenated_compounds:
        #     match = re.search(r'\w+\b- ' + re.escape(word) + r'\b', report)
        #     while match:
        #         report = report[:match.end() - (len(word)+1)] + report[match.end() - (len(word)):]
        #         match = re.search(r'\w+\b- ' + re.escape(word) + r'\b', report)
        # Separate random i stuck to the end of some words
        match = re.search(r'\w+i\b', report)
        while match:
            report = report[:match.end()-1] + " " + report[match.end()-1:]
            match = re.search(r'\w+i\b', report)
        # Capitalize first letter after periods
        sentences = report.split(". ")
        report = ""
        for sentence in sentences:
            report += sentence.capitalize()
            if not sentence.endswith("."):
                report += ". "
        # Capitalize lonely i's
        report = report.replace(" i ", " I ")
        # Capitalize various things
        for word in to_cap:
            report = report.replace(word, word.capitalize())
        # Chance to just make everything uppercase, why not
        upper_chance = 0.01
        if random.random() <= upper_chance:
            report = report.upper()
        return report


    def generate_report(self, max_report_len=1000, creativity=0.01):
        """
        Generates a random UFO sighting report
        """
        self.start_word_prediction()  # initialize prediction loop
        next_word_tok = -1
        count = 0
        min_count = max_report_len/2
        k = 50
        while next_word_tok != end_token and count < max_report_len:
            # Get word probabilities
            word_probs = self.get_next_word_probabilities()
            # Add more variety every few tokens (even if it's weird)
            # variety_spread = 10
            # if count % variety_spread == 0:
            #     print("Generating word ", count)
            #     big_k = 100
            #     top_probs = tf.math.top_k(word_probs, big_k)
            #     next_word_tok = np.random.choice(top_probs.indices.numpy().flatten(), 1)
            #     next_word_tok = tf.constant(next_word_tok)[tf.newaxis]
            #     self.feedback_next_word_tok(next_word_tok)
            #     count += 1
            #     continue

            top_probs = tf.math.top_k(word_probs, k)
            
            report_completion = (self.output_array.size() / max_report_len).numpy()
            highest_end_token_index = int(k * report_completion)
            does_have_end_token = tf.reduce_any(tf.math.equal(top_probs.indices[0:highest_end_token_index], tf.constant(end_token)))
            normalized_top_probs = tf.linalg.normalize(top_probs.values, ord=1)[0].numpy().flatten()
            # Check if ending token is in top possible words
            # if count > min_count and tf.cond(tf.equal(tf.constant(True), does_have_end_token), lambda: True, lambda: False):
            if tf.cond(tf.equal(tf.constant(True), does_have_end_token), lambda: True, lambda: False):
                # print("Found End token in top ", highest_end_token_index, " results")
                next_word_tok = tf.constant([end_token])[tf.newaxis]
            else:
                # Normalize the top probabilities so that most likely ones are still more likely than other words
                # This should make the sentences more coherent
                next_word_tok = np.random.choice(top_probs.indices.numpy().flatten(), 1, p=normalized_top_probs)
                next_word_tok = tf.constant(next_word_tok)[tf.newaxis]
                # # Use the following instead to remove end tokens from possible selection
                # indices = top_probs.indices.numpy()
                # values = top_probs.values.numpy()
                # valid_mask = indices != end_token
                # indices = indices[valid_mask]
                # values = values[valid_mask]
                # normalized_top_probs = tf.linalg.normalize(tf.constant(values), ord=1)[0].numpy().flatten()
                # next_word_tok = np.random.choice(indices.flatten(), 1, p=normalized_top_probs)
                # next_word_tok = tf.constant(next_word_tok)[tf.newaxis]

            if count < min_count and next_word_tok == end_token:
                end_tok_idx = tf.where(tf.math.not_equal(top_probs.indices, tf.constant(end_token)))

            # Feed word back into transformer output
            self.feedback_next_word_tok(next_word_tok)
            count += 1
        self.output = tf.transpose(self.output_array.stack())
        return self.token_tensor_to_report(self.output)
        # print(self.token_tensor_to_report(self.output))

