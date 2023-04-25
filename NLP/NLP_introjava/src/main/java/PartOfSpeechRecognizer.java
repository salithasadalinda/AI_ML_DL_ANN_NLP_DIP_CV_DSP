import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.sql.SQLOutput;
import java.util.List;

public class PartOfSpeechRecognizer {
    public static void main(String[] args) {
        StanfordCoreNLP stanfordnlp = NLProcessor.getActionLine();
        String txt_POS="hey i like apple and country";
        CoreDocument coreDocument=new CoreDocument(txt_POS);
        stanfordnlp.annotate(coreDocument);
        List<CoreLabel> coreLabelList = coreDocument.tokens();
        for(CoreLabel coreLabel:coreLabelList){
            String partOfSpeech = coreLabel.get(CoreAnnotations.PartOfSpeechAnnotation.class);
            System.out.println(coreLabel.originalText()+">>>>>>>>>>"+partOfSpeech);
        }
    }
}
