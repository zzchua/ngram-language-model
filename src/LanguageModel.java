import java.io.File;
import java.io.FileNotFoundException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
/**
 * This class represents the Language Model Program
 * Using both backoff and interpolation, you can train the models on a given corpus by 
 * modifying the global constant "TRAINING_CORPUS" below to specify the desired training corpus
 * You can also choose the corpus on which to run perplexity tests by modifying the constant "TEST_CORPUS" below
 * to specify the desired test corpus to use.
 * 
 * To change the parameters of the Interpolation models, modify LAMBDA_1, 2, 3
 * To change the parameters of the Backoff model, modify DISCOUNT_FACTOR
 * 
 * When run, the program trains the 2 language models on the specifed training corpus using 90% of the data.
 * It then tests the 2 langauge models on the specified test corpus using 10% of that corpus data.
 * 
 * You can change the % values of how much of each corpus to use by modifying the constants in CorpusParser.java
 * 
 * @author zzchua
 */

public class LanguageModel {
	
	// Feel free to modify the values of these global constants
	public static final double DISCOUNT_FACTOR = 0.7;
	public static final double LAMBDA_1 = 0.1;
	public static final double LAMBDA_2 = 0.5;
	public static final double LAMBDA_3 = 0.4;
	public static final String TRAINING_CORPUS = "corpus/gutenberg.txt";
	public static final String TEST_CORPUS = "corpus/brown.txt";
	
	// Do not modify this
	public static final double K = 1;
	
	public static void main(String[] args) throws FileNotFoundException {
		
		File fTraining = new File(TRAINING_CORPUS);
		File fTest = new File(TEST_CORPUS);
		
		// Split the corpus into 3 sets: 0: training set, 1: dev set 2: test set
		ArrayList<ArrayList<String>> trainingWordSets = CorpusParser.getTrainingSentences(fTraining);
		ArrayList<ArrayList<String>> testWordSets = CorpusParser.getTrainingSentences(fTest);
		
		System.out.println("Building the Language Models on " + TRAINING_CORPUS + "...");
		// Training The Model:
		HashMap<String, Integer> unigramModel = CorpusParser.generateUnigramModel(trainingWordSets.get(0));
		HashMap<String, HashMap<String, Integer>> bigramModel = CorpusParser.generateBigramModel(trainingWordSets.get(0));
		HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel = CorpusParser.generateTrigramModel(trainingWordSets.get(0));
		
		int totalWordCount = 0; 
		for (String word : unigramModel.keySet()) {
			totalWordCount += unigramModel.get(word);
		}
		
//		// Devset for tuning:
//		ArrayList<ArrayList<String>> devSentences = convertSentenceWords(trainingWordSets.get(1));
//		System.out.println("Total No. of " + TEST_CORPUS + " dev sentences: " + devSentences.size());

		// Testing the LM:
		ArrayList<ArrayList<String>> testSentences = convertSentenceWords(testWordSets.get(2));
		System.out.println("Total No. of " + TEST_CORPUS + " test sentences: " + testSentences.size());

		System.out.println("\nRunning Backoff Perplexity Test using " + TEST_CORPUS + "...");
		// Calculating the perplexity using Backoff:
		double perplexityBackoff = calculatePerplexityBackoff(testSentences, unigramModel, bigramModel, trigramModel);
		System.out.println("Testing " + TEST_CORPUS + " in Backoff:\n Perplexity: " + perplexityBackoff); 
		
		System.out.println("\nRunning Interpolation Perplexity Test using " + TEST_CORPUS + "...");
		// Calculating the perplexity using Interpolation:
		double perpelexityInterpolation = calculatePerplexityInterpolation(testSentences, unigramModel, bigramModel, trigramModel, totalWordCount);
		System.out.println("Testing " + TEST_CORPUS + " in Interpolation:\n Perplexity: " + perpelexityInterpolation); 
	}
	
	
	/**
	 * Takes a array list of string sentences and converts it into an arraylist of sentences, where each sentence is an arraylist of words
	 * @param sentences list of string sentences
	 * @return a ArrayList of ArrayList of words. Each array list of words is a sentence.
	 */
	public static ArrayList<ArrayList<String>> convertSentenceWords(ArrayList<String> sentences) {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		for (String s : sentences) {
			// split the sentence into words
			String[] words = s.split("\\s+");
			// make into an array list:
			ArrayList<String> sentenceWords = new ArrayList<>();
			for (int i = 0; i < words.length; i++) {
				sentenceWords.add(words[i]);
			}
			result.add(sentenceWords);
		}
		return result;
	}
	
	/**
	 * Calculates the log probability of a given sentence in the backoff model
	 * @param sentence the sentence for which a probability is to be calculated
	 * @param unigramModel the model representing the unigram
	 * @param bigramModel the model representing the bigram
	 * @param trigramModel the model representing the trigram
	 * @return the log probability of the given sentence
	 */
	public static double calculateSentenceProbabilitiesBackoff(ArrayList<String> sentence, 
			HashMap<String, Integer> unigramModel, 
			HashMap<String, HashMap<String, Integer>> bigramModel, 
			HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel) {
		
		
		// sentence starts with <s><s>
		assert(sentence.get(0).equals("<s>") && sentence.get(1).equals("<s>"));
		double logSentenceProbability = 0;
		
		
		for (int i = 2; i < sentence.size(); i++) {
			
			String word = sentence.get(i);
			// construct the history:
			ArrayList<String> tHistory = new ArrayList<String>();
			tHistory.add(sentence.get(i - 2));
			tHistory.add(sentence.get(i - 1));
			String bHistory = sentence.get(i-1);
			// calculate 3-gram probability:
			// check to see if trigram exists:
			// if 2-word history matches and next word is present:
			if (trigramModel.containsKey(tHistory) && trigramModel.get(tHistory).containsKey(word)) {
				HashMap<String, Integer> innerTriMap = trigramModel.get(tHistory);
				// calculating d * MLE = p1
				logSentenceProbability += Math.log(DISCOUNT_FACTOR * (double)innerTriMap.get(word)/ ((double) bigramModel.get(tHistory.get(0)).get(tHistory.get(1)) + K));
			} else {
				if (bigramModel.containsKey(bHistory) && bigramModel.get(bHistory).containsKey(word)) {	
					// calculate beta:
					// find the sum of all probabilities of trigrams beginning with the same 2 first words:
					double sumTrigramProbabilities = 0.0;
					HashMap<String, Integer> innerTriMap = trigramModel.get(tHistory);
					if (innerTriMap != null) {
						for (String w: innerTriMap.keySet()) {
							sumTrigramProbabilities += DISCOUNT_FACTOR * ((double) innerTriMap.get(w)/((double) bigramModel.get(tHistory.get(0)).get(tHistory.get(1)) + K));
						}
					}
					double alpha = 1 - sumTrigramProbabilities;
					// finding the denominator:
					// finding all words whose trigrams with the same first 2 words are 0:
					int sumDenominator = 0; 
					// for all bigrams starting with wi-1
					HashMap<String, Integer> innerBiMap = bigramModel.get(bHistory);
					for (String w : innerBiMap.keySet()) {
						if (innerTriMap == null || !innerTriMap.containsKey(w)) {
							// add the probability:
							sumDenominator += innerBiMap.get(w);
						}
					}
					logSentenceProbability += Math.log(DISCOUNT_FACTOR * alpha * ((double) innerBiMap.get(word) / ((double)sumDenominator + K)));
				} else {
					// calculate unigram probability:
					
					// calculate beta:
					// find the sum of all probabilities of bigrams beginning with the same last word i -1:
					double sumBigramProbabilities = 0.0;
					HashMap<String, Integer> innerBiMap = bigramModel.get(bHistory);
					if (innerBiMap != null) {
						for (String w : innerBiMap.keySet()) {
							sumBigramProbabilities += DISCOUNT_FACTOR * ((double)innerBiMap.get(w)/ ((double)unigramModel.get(bHistory) + K));
						}
					}
					// finding the denominator:
					// finding all words whose bigrams with the same first word are 0:
					int sumDenominator = 0;
					for (String w : unigramModel.keySet()) {
						if (innerBiMap == null || !innerBiMap.containsKey(w)) {
							sumDenominator += unigramModel.get(w);
						}
					}
					double alpha = 1 - sumBigramProbabilities;
					if (unigramModel.containsKey(word)) {
						logSentenceProbability += Math.log(DISCOUNT_FACTOR * alpha * ((double)unigramModel.get(word) / ((double)sumDenominator + K)));
					} else {
						logSentenceProbability += Math.log(DISCOUNT_FACTOR * alpha * (K / ((double)sumDenominator + K)));
					}
				}
			}
		}
		return logSentenceProbability;
	}
	
	/**
	 * Calculates the perplexity of a list of sentences using the backoff model.
	 * @param sentences a list of sentences. Each sentence in the list is a list of words representing the sequence of words in the sentence.
	 * @param unigramModel
	 * @param bigramModel
	 * @param trigramModel
	 * @return the perplexity of the backoff model
	 */
	public static double calculatePerplexityBackoff(ArrayList<ArrayList<String>> sentences,
			HashMap<String, Integer> unigramModel, 
			HashMap<String, HashMap<String, Integer>> bigramModel, 
			HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel) {
		// for each sentence:
		BigDecimal l = new BigDecimal(0.0);
		int count = 0;
		for (ArrayList<String> s : sentences) {
			double logP = calculateSentenceProbabilitiesBackoff(s, unigramModel, bigramModel, trigramModel);
//			System.out.println("sentence logp = " + logP);
			if (logP != 0) {
				l = l.add(new BigDecimal(logP/Math.log(2)));
//				System.out.println("l: " + l.toPlainString());
			}
			count++;
//			System.out.println("done... " +  count);
		}
		// get number of words in test corpus:
		int totalWords = 0;
		for (ArrayList<String> sentence : sentences) {
			for (String word : sentence) {
				totalWords++;
			}
		}
		l = l.divide(new BigDecimal((double)totalWords),  2, RoundingMode.HALF_UP);
		double perplexity = Math.pow(2, -l.doubleValue());
		return perplexity;
	}
	
	/**
	 * Calculates the perplexity of a given sequence of sentences
	 * @param sentences a list of sentences. Each sentence in the list is a list of words representing the sequence of words in the sentence.
	 * @param unigramModel
	 * @param bigramModel
	 * @param trigramModel
	 * @param totalWordCount a count of all the words in the training set.
	 * @return the perplexity of the interpolation model
	 */
	public static double calculatePerplexityInterpolation(ArrayList<ArrayList<String>> sentences,
			HashMap<String, Integer> unigramModel, 
			HashMap<String, HashMap<String, Integer>> bigramModel, 
			HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel, int totalWordCount) {
		// for each sentence:
		BigDecimal l = new BigDecimal(0.0);
		int count = 0;
		for (ArrayList<String> s : sentences) {
			double logP = calculateSentenceProbabilitiesInterpolation(s, unigramModel, bigramModel, trigramModel, totalWordCount);
//			double p = Math.exp(logP);
//			System.out.println("sentence p = " + p);
			l = l.add(new BigDecimal(logP/Math.log(2)));
//			System.out.println("l: " + l);
			count++;
//			System.out.println("done... " +  count);
		}
		// get number of words in test corpus:
		int totalWords = 0;
		for (ArrayList<String> sentence : sentences) {
			for (String word : sentence) {
				totalWords++;
			}
		}
		l = l.divide(new BigDecimal((double) totalWords), 2, RoundingMode.HALF_UP);
		double perplexity = Math.pow(2, -l.doubleValue());
		return perplexity;
	}
	
	/**
	 * Calculates the log probability of a sentence using interpolation
	 * @param sentence a list of words representing the sentence
	 * @param unigramModel
	 * @param bigramModel
	 * @param trigramModel
	 * @param totalWordCount the count of all words in the training set.
	 * @return the log probability of the given sentence
	 */
	public static double calculateSentenceProbabilitiesInterpolation(ArrayList<String> sentence, 
			HashMap<String, Integer> unigramModel, 
			HashMap<String, HashMap<String, Integer>> bigramModel, 
			HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel, int totalWordCount) {
		// sentence starts with <s><s>
		assert(sentence.get(0).equals("<s>") && sentence.get(1).equals("<s>"));
		double logSentenceProbability = 0;
		for (int i = 2; i < sentence.size(); i++) {
			String w2 = sentence.get(i-2);
			String w1 = sentence.get(i-1);
			String w = sentence.get(i);
			ArrayList<String> tHistory = new ArrayList<>();
			tHistory.add(w2);
			tHistory.add(w1);
			// calculate the trigram value:
			HashMap<String, Integer> innerTriMap = trigramModel.get(tHistory);
			double p1 = 0;
			double p2 = 0;
			double p3 = 0;
			if (innerTriMap != null && innerTriMap.containsKey(w)) {
				p1 = LAMBDA_1 * ((double) innerTriMap.get(w) / (double) bigramModel.get(w2).get(w1)); 
			}
			// calcualte the bigram value:
			HashMap<String, Integer> innerBiMap = bigramModel.get(w1);
			if (innerBiMap != null && innerBiMap.containsKey(w)) {
				p2 = LAMBDA_2 * ((double) innerBiMap.get(w)/ (((double) unigramModel.get(w1)) + K));
			}
			// calculate the unigram value:
			if (unigramModel.containsKey(w)) {
				p3 = LAMBDA_3 * ((double) unigramModel.get(w) / (((double) totalWordCount) + K));
			}
			double p = p1 + p2 + p3;
			if (p != 0) {
				logSentenceProbability += Math.log(p);
			} else {
				logSentenceProbability += Math.log((double) K / (((double) totalWordCount) + K));
			}
		}
		return logSentenceProbability;
	}
	
	
}
