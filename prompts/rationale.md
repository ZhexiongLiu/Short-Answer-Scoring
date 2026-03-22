### You will be given a question, a rubric, and a student's answer in German. You will evaluate the student's answer based on the specific rubric. You need to answer [Correct, Incorrect].
### Question: 
    {question}
### Evaluation Rubric:
    check one of the three rubric below, which is the most suitable for the student answer:
        - if the student answer mentions all the component in {partial_correct_rubric}:
            You need to answer Incorrect
        - if the student answer mentions all the component in {incorrect_rubric}:
            You need to answer Incorrect
        - if the student answer mentions all the component in {correct_rubric}:
            You need to answer Correct
### Student Answer: 
    {answer}

### Response: