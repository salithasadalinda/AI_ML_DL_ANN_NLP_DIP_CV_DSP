import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.StanfordCoreNLPClient;
import java.util.List;

public class SentenceSplitter {
    public static void main(String[] args) {
        // Get the StanfordCoreNLP object using the NLProcessor class
        StanfordCoreNLP stanfordCoreNLP = NLProcessor.getActionLine();

        // Define the example text to be split into sentences
        String example_str = "i need to learn NLP........ so i need to /learn DL |and ML";

        // Create a CoreDocument object and annotate it using the StanfordCoreNLP object
        CoreDocument coreDocument = new CoreDocument(example_str);
        stanfordCoreNLP.annotate(coreDocument);

        // Get the list of CoreSentences from the CoreDocument object
        List<CoreSentence> coreSentenceList = coreDocument.sentences();

        // Print out each sentence in the list of CoreSentences
        for(CoreSentence coreSentence: coreSentenceList){
            System.out.println(coreSentence.toString());
        }
    }
}
