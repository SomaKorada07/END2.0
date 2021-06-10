# Session 6 - GRUs, Seq2Seq and Introduction to Attention Mechanism

1. Take the last code (+tweet dataset) and convert that in such a war that:
   1. *encoder:* an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. **VERY IMPORTANT TO MAKE THIS SINGLE VECTOR**
   2. this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
   3. and send this final vector to a Linear Layer and make the final prediction. 
   4. This is how it will look:
      1. embedding
      2. *word from a sentence +last hidden vector ->* encoder *-> single vector*
      3. *single vector + last hidden vector -> decoder -> single vector*
      4. *single vector -> FC layer -> Prediction*
2. Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%. 
3. The code needs to look as simple as possible, the focus is on making encoder/decoder classes and how to link objects together
4. Getting good accuracy is NOT the target, but must achieve at least **45%** or more
5. Once the model is trained, take one sentence, "print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder. ← **THIS IS THE ACTUAL ASSIGNMENT**