import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.util.List;

public class TokenizationClass {
    public static void main(String[] args) {
        // Get a new instance of the StanfordCoreNLP class
        StanfordCoreNLP stanfordnlp = NLProcessor.getActionLine();

        // Text to be tokenized
        String languageText = "this is natural language processing freely";

        // Create a new CoreDocument object to hold the text
        CoreDocument c_document = new CoreDocument(languageText);

        // Annotate the document using the StanfordCoreNLP pipeline
        stanfordnlp.annotate(c_document);

        // Get a list of CoreLabel objects, where each object represents a single token
        List<CoreLabel> coreLabelList = c_document.tokens();

        // Iterate over the tokens and print each one
        for (CoreLabel coreLabel : coreLabelList) {
            System.out.println(coreLabel.originalText());
        }
    }
}



//////optimized by chat gpt/////

//import edu.stanford.nlp.pipeline.CoreDocument;
//        import edu.stanford.nlp.pipeline.StanfordCoreNLP;
//        import java.util.List;
//        import java.util.Properties;
//
//public class TokenizationClass {
//    public static void main(String[] args){
//        StanfordCoreNLP stanfordnlp = NLProcessor.getActionLine();
//        String languageText = "this is natural language processing";
//        CoreDocument c_document = new CoreDocument(languageText);
//        stanfordnlp.annotate(c_document);
//        List<String> tokens = c_document.tokens().stream().map(token -> token.originalText()).toList();
//        System.out.println(tokens);
//    }
//}
