import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.Properties;

public class NLProcessor {
    // Declare instance variables
    private static Properties properties;
    private static String actionName = "tokenize, ssplit, pos, lemma, ner";//the action priority

    private static StanfordCoreNLP stanfordnlp;

    // Empty constructor
    public NLProcessor() {
    }

    // Initialize the properties object and set the annotators property to the value of the actionName variable
    static {
        properties = new Properties();
        properties.setProperty("annotators", actionName);
    }

    // Get an instance of the StanfordCoreNLP pipeline, creating one if it does not already exist
    public static StanfordCoreNLP getActionLine() {
        if (stanfordnlp == null) {
            stanfordnlp = new StanfordCoreNLP(properties);
        }
        return stanfordnlp;
    }
}

///////optimisez by chatgpt////////
//
//class NLProcessor {
//    private static final String ACTION_NAME = "tokenize";
//    private static StanfordCoreNLP stanfordnlp;
//
//    static {
//        Properties properties = new Properties();
//        properties.setProperty("annotators", ACTION_NAME);
//        stanfordnlp = new StanfordCoreNLP(properties);
//    }
//
//    public static StanfordCoreNLP getActionLine(){
//        return stanfordnlp;
//    }
//}