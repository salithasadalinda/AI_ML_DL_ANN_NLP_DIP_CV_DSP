import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.List;
public class NameEntityRecognizer {
    public static void main(String[] args) {
        StanfordCoreNLP stanfordCoreNLP = NLProcessor.getActionLine();
        String txt_name_entity_rec="hey! i am Harry Potter and i have a friend his name is Ron we both live in England";
        CoreDocument coreDocument=new CoreDocument(txt_name_entity_rec);
        stanfordCoreNLP.annotate(coreDocument);
        List<CoreLabel> coreLabelList=coreDocument.tokens();

        for(CoreLabel coreLabel:coreLabelList){
            String NER_ = coreLabel.get(CoreAnnotations.NamedEntityTagAnnotation.class);
            System.out.println(coreLabel.originalText()+"---------->"+NER_);


        }




    }
}
