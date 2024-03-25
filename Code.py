def translate_sentence(sentence, eng_tokenizer, fra_tokenizer, max_eng_sent_len, max_fra_sent_len, translator_model):
    # Tokenize and encode the input sentence
    sentence_seq = eng_tokenizer.texts_to_sequences([sentence])
    padded_sentence = pad_sequences(sentence_seq, maxlen=max_eng_sent_len, padding='post')

    # Get the model prediction for the input sentence
    predicted_seq = translator_model.predict(padded_sentence)

    # Decode the predicted sequence into French
    translated_sentence = ''

    for token in predicted_seq[0]:
        sampled_token_index = np.argmax(token)
        if sampled_token_index == 8:
            break
        word = fra_tokenizer.index_word[sampled_token_index]
        translated_sentence += word + ' '

    return translated_sentence.strip()

# Example usage:
english_sentence = "How are you doing today?"
translated_sentence = translate_sentence(english_sentence, eng_tokenizer, fra_tokenizer, max_eng_sent_len, max_fra_sent_len, translator_model)

print(f"English: {english_sentence}")
print(f"Translated French: {translated_sentence}")
