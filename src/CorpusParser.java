import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
/**
 * This class provides some static methdods useful for parsing the corpus into useful data.
 * Modify the constants below to change the percentage of each corpus to use for Training, Development and Testing.
 * @author zzchua 
 */
 

public class CorpusParser {
	public static final double TRAINING_SET_AMT = 0.9;
	public static final double DEV_SET_AMT = 0.05;
	public static final double TEST_SET_AMT = 0.05;
	
	
	/**
	 * Takes a corpus as a file and splits the corpus into sentences. 
	 * Appends a <s> <s> at the start of every sentence and a </s> at the end.
	 * @param f corpus of text
	 * @return an ArrayList containing ArrayLists of sentences. 0th: training sentences, 1st: dev sentences, 2nd: test sentences
	 * @throws FileNotFoundException
	 */
	public static ArrayList<ArrayList<String>> getTrainingSentences(File f) throws FileNotFoundException {
		Scanner fileScanner = new Scanner(f);
		ArrayList<String> sentences = new ArrayList<String>();
		while (fileScanner.hasNextLine()) {
			String line = fileScanner.nextLine();
			// append a start and stop symbol to every sentence. 2 Start symbols
			line = "<s> <s> " + line;
			line = line + " </s> ";
			// clean the data:
			line = line.toLowerCase();
			line = line.replaceAll("[()%#@*&,.!?`\"]", "");
			sentences.add(line);
		}
		int trainingLast = (int)(TRAINING_SET_AMT * sentences.size());
		ArrayList<String> trainingSet = copyArrayList(0, trainingLast, sentences);
		
		int devLast = (int)(DEV_SET_AMT * sentences.size() + trainingLast);
		ArrayList<String> devSet = copyArrayList(trainingLast, devLast, sentences);
		ArrayList<String> testSet = copyArrayList(devLast, sentences.size(), sentences);
		
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		result.add(trainingSet);
		result.add(devSet);
		result.add(testSet);
		assert(trainingSet.size() + devSet.size() + testSet.size() == sentences.size());
		return result;
	}
	
	public static ArrayList<String> copyArrayList(int start, int last, ArrayList<String> list) {
		ArrayList<String> copy = new ArrayList<>(); 
		for (int i = start; i < last; i++) {
			copy.add(list.get(i));
		}
		return copy;
	}
	
	/**
	 * Generates a Map of words to frequencies
	 * @param trainingSet A list of sentences
	 * @return a map representing the unigram model.
	 */
	public static HashMap<String, Integer> generateUnigramModel(ArrayList<String> trainingSet) {
		HashMap<String, Integer> wordFreq = new HashMap<>();
		for (String sentence : trainingSet) {
			// split the sentence into words
			String[] words = sentence.split("\\s+");
			
			// add the word to the map
			for (int i = 0; i < words.length; i++) {
				if (!wordFreq.containsKey(words[i])) {
					wordFreq.put(words[i], 0);
				}
				wordFreq.put(words[i], wordFreq.get(words[i]) + 1);
			}
		}
		return wordFreq;
	}
	
	/**
	 * Generates a Map of Strings representing the history word to a nested map containing the current word and the frequency of the 2-gram.
	 * @param trainingSet list of sentences
	 * @return a map representing the bigram model
	 */
	public static HashMap<String, HashMap<String, Integer>> generateBigramModel(ArrayList<String> trainingSet) {
		HashMap<String, HashMap<String, Integer>> bigramModel = new HashMap<>();
		for (String sentence : trainingSet) {
			String[] words = sentence.split("\\s+");
			for (int i = 0; i < words.length - 1; i++) {
				String history = words[i];
				if (!bigramModel.containsKey(history)) {
					// create the inner map:
					bigramModel.put(history, new HashMap<String, Integer>());
				}
				HashMap<String, Integer> innerBiMap = bigramModel.get(history);
				if (!innerBiMap.containsKey(words[i + 1])) {
					innerBiMap.put(words[i + 1], 0);
				}
				innerBiMap.put(words[i + 1], innerBiMap.get(words[i+1]) + 1);
				bigramModel.put(history, innerBiMap);
			}
		}
		return bigramModel;
	}
	
	/**
	 * Generates a trigram represented as a map of a list of 2 strings to a nested map of a string to a frequency count. 
	 * The outer map key is a list of 2-word sequences representing the history with the older word coming first.
	 * @param trainingSet list of sentences
	 * @return a map representing the trigram.
	 */
	public static HashMap<ArrayList<String>, HashMap<String, Integer>> generateTrigramModel(ArrayList<String> trainingSet) {
		HashMap<ArrayList<String>, HashMap<String, Integer>> trigramModel = new HashMap<>();
		for (String sentence : trainingSet) {
			String[] words = sentence.split("\\s+");
			for (int i = 0; i < words.length - 2; i++) {
				ArrayList<String> historyPair = new ArrayList<String>();
				historyPair.add(0, words[i]);
				historyPair.add(1, words[i+1]);
				if (!trigramModel.containsKey(historyPair)) {
					trigramModel.put(historyPair, new HashMap<String, Integer>());
				}
				// check to see if w exists:
				HashMap<String, Integer> innerTriMap = trigramModel.get(historyPair);
				if (!innerTriMap.containsKey(words[i + 2])) {
					// add the word:
					innerTriMap.put(words[i + 2], 0);
				}
				innerTriMap.put(words[i + 2], innerTriMap.get(words[i + 2]) + 1);
				// put the innermap back in:
				trigramModel.put(historyPair, innerTriMap);
			}
		}
		return trigramModel;
	}
}