import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.List;

public class Lemma {
    public static void main(String[] args) {
        // create a new instance of StanfordCoreNLP
        StanfordCoreNLP stanfordCoreNLP = new StanfordCoreNLP();

        // define the input text to check lemma
        String text_lemma_check = "game gamer gaming";

        // create a CoreDocument instance and annotate it with the StanfordCoreNLP object
        CoreDocument coreDocument = new CoreDocument(text_lemma_check);
        stanfordCoreNLP.annotate(coreDocument);

        // retrieve the list of CoreLabels from the CoreDocument
        List<CoreLabel> coreLabelList = coreDocument.tokens();

        // loop through the list of CoreLabels and print the lemma for each token
        for (CoreLabel coreLabel : coreLabelList) {
            String lemma = coreLabel.lemma();
            System.out.println(lemma);
        }
    }
}
