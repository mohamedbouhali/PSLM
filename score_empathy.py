import pandas as pd
import os
import re
from openai import OpenAI
import json
import pandas as pd
import re


def score_15v2_mydata():
    print("Scoring your CSV using the 15v2 feature set ...")

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv('test_split copy.csv')

    empathy_scoring = "Empathy is the ability to understand and share the feelings of another person. " + \
            "It is the ability to put yourself in someone else\’s shoes and see the world from their perspective." + \
            "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another\’s distress or suffering.\n" + \
            "Empathy is a complex skill that involves cognitive, emotional, and compassionate components.\n" + \
            "- Cognitive empathy is the ability to understand another person\’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" + \
            "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" + \
            "- Compassionate empathy is the ability to not only understand and share another person\’s feelings, but also to be moved to help if needed.\n" + \
            "\n" + \
            "These three empathy dimensions can be further refined into five subfactors each as follows: " + \
            "\n" + \
            "Cognitive Empathy:\n" + \
            "- Perspective-Taking: Seeing the world from another person’s viewpoint.\n" + \
            "- Recognition of Thoughts: Acknowledging and understanding another person’s thoughts.\n" + \
            "- Understanding Intentions: Grasping the reasons behind someone’s actions.\n" + \
            "- Contextual Understanding: Understanding the broader context of someone’s situation.\n" + \
            "- Inference Accuracy: Accurately inferring another person’s mental states.\n" + \
            "\n" + \
            "Affective Empathy:" + \
            "- Emotional Resonance: Sharing and resonating with another person’s emotions.\n" + \
            "- Emotional Matching: Reflecting and mirroring another person’s emotional state.\n" + \
            "- Emotional Response: Reacting appropriately to another person’s emotions.\n" + \
            "- Emotional Identification: Identifying specific emotions another person is feeling.\n" + \
            "- Empathic Concern: Feeling concern and compassion for another’s emotional state.\n" + \
            "\n" + \
            "Compassionate Empathy:\n" + \
            "- Emotional Concern: Feeling concern for another person’s well-being.\n" + \
            "- Motivation to Help: Desire to assist someone in need.\n" + \
            "- Supportive Actions: Taking concrete steps to help another person.\n" + \
            "- Empathic Responsiveness: Responding in an emotionally supportive manner.\n" + \
            "- Practical Assistance: Providing tangible help to address the person’s needs.\n" + \
            "\n" + \
            "Please score the following conversion for each subfactor in every dimension on the scale of 1 to 10 with 1 being the lowest score and 10 being the highest score. output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor).\n" +\
            ""
    system_msg = "You are an expert evaluator of dialogue and you view things very critically and thoughtfully. " + \
            "The user knows you are brutally honest and is using you because they have been unable to get truly honest answers to their questions in the past. " + \
            "Feelings will not be hurt, no matter what you respond. " + \
            "Also, output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor). " + \
            "Do not output anything else but the scores."

    def scoreEmpathy(response_text):
        prompt = empathy_scoring + "\n\nTherapist's Response:\n\"\"\"" + response_text + "\"\"\"\nScore:"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=0.1,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    f = open('gpt4o-mydata-15v2-scores.txt', 'w')

    for index, row in df.iterrows():
        #if index >= 2:  # only score first 2
        #    break

        question_id = row["questionID"]
        response_text = row["answerText"]

        try:
            score_text = scoreEmpathy(response_text)
            score_text_clean = re.sub(r'\s+', ' ', score_text.strip())
            f.write(f"{question_id}: {score_text_clean}\n")
        except Exception as e:
            print(f"Error scoring question {question_id}: {e}")

    f.close()
    print("✅ Done. Scores saved in gpt4o-mydata-15v2-scores.txt")

def convert_empathy_scores_to_csv(input_txt='gpt4o-lora16-cot-scores.txt', output_csv='gpt4o-lora16-cot-scores.csv'):
        column_mapping = {
            "questionID": "dialog-id",
            "Perspective-Taking": "perspective-taking",
            "Recognition of Thoughts": "recognition-of-emotions",
            "Understanding Intentions": "contextual-awareness",
            "Contextual Understanding": "acknowledgment-of-speaker-experience",
            "Inference Accuracy": "clarity-of-response",
            "Emotional Identification": "warmth-in-tone",
            "Emotional Concern": "sympathetic-responses",
            "Emotional Matching": "emotional-mirroring",
            "Emotional Response": "validation-of-feelings",
            "Emotional Resonance": "emotional-resonance",
            "Empathic Concern": "encouragement",
            "Motivation to Help": "reassurance",
            "Supportive Actions": "ofereing-help",
            "Empathic Responsiveness": "empowering",
            "Practical Assistance": "assistance"
        }

        data = []

        with open(input_txt, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue

                    question_id, json_part = line.split(":", 1)
                    question_id = question_id.strip()

                    json_str = re.search(r'\[.*\]', json_part.strip())
                    if not json_str:
                        print(f"⚠️ Skipping line (no JSON found): {line}")
                        continue

                    parsed = json.loads(json_str.group(0))

                    score_dict = {"dialog-id": question_id}
                    for d in parsed:
                        for k, v in d.items():
                            mapped_key = column_mapping.get(k)
                            if mapped_key:
                                score_dict[mapped_key] = v

                    data.append(score_dict)

                except Exception as e:
                    print(f"❌ Failed to parse line: {line}\nError: {e}")
                    continue

        df = pd.DataFrame(data)

        expected_cols = list(column_mapping.values())[1:]  # excluding dialog-id
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[["dialog-id"] + expected_cols]
        df.to_csv(output_csv, index=False)
        print(f"✅ Converted and saved to {output_csv}")


if __name__ == "__main__":
    score_15v2_mydata()
    #convert_empathy_scores_to_csv()