import sys
import os
import en_core_web_sm
import json
import numpy as np
import random
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import Any, List, Mapping, Tuple

from constants import (
    INGEST_FILE_PATH,
    QUESTION_FILE_PATH
)

class QuestionGenerator:

    def __init__(self) -> None:

        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512

        print("CUDA available:", torch.cuda.is_available())

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # use_fast=True
        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=True)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

        self.qa_evaluator = QAEvaluator()

    # Generate the questions
    def generate_questions(self, question_count) -> List:

        print("Generating questions...\n")
        
        # Read the content of the file
        with open(INGEST_FILE_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Generate questions using the content and question_count parameter
        qg_inputs, qg_answers = self.generate_qg_inputs(content, "sentences")
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        print("Validating...\n")

        # Validate
        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        # Evaluate
        # if use_evaluator:
        print("Evaluating QA pairs...\n")
        encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(
            generated_questions, qg_answers
        )
        scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

        if question_count:
            qa_list = self._get_ranked_qa_pairs(
                generated_questions, qg_answers, scores, question_count
            )
        else:
            qa_list = self._get_ranked_qa_pairs(
                generated_questions, qg_answers, scores
            )

        print("Progress...\n")

        questions = [item["question"] for item in qa_list]
        with open(QUESTION_FILE_PATH, 'w', encoding='utf-8', errors='replace') as file:
            for question in questions:
                file.write(question + '\n')

        print("Completed.\n")

    def generate_qg_inputs(self, text: str, answer_style: str) -> Tuple[List[str], List[str]]:

        inputs = []
        answers = []

        if answer_style == "sentences" or answer_style == "all":
            segments = self._split_into_segments(text)

            for segment in segments:
                sentences = self._split_text(segment)

                prepped_inputs, prepped_answers = self._prepare_qg_inputs(
                    sentences, segment
                )
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        print("Returning inputs and answers...\n")

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:

        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        print("Returning all questions...\n")
        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text: str) -> List[str]:

        MAX_TOKENS = 490
        paragraphs = text.split("\n")
        tokenized_paragraphs = [
            self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0
        ]
        segments = []

        while len(tokenized_paragraphs) > 0:
            segment = []

            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(
        self,
        sentences: List[str],
        text: str
    ) -> Tuple[List[str], List[str]]:

        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = f"{self.ANSWER_TOKEN} {sentence} {self.CONTEXT_TOKEN} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:

        print("Starting generation...\n")

        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"], max_new_tokens=self.SEQ_LENGTH)
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        print("Returning a question...\n")

        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:

        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_ranked_qa_pairs(
        self, generated_questions: List[str], qg_answers: List[str], scores, question_count: int = 10
    ) -> List[Mapping[str, str]]:

        if question_count > len(scores):
            question_count = len(scores)
            print((
                f"\nWas only able to generate {question_count} questions.",
                "For more questions, please input a longer text.")
            )

        qa_list = []

        for i in range(question_count):
            index = scores[i]
            qa = {
                "question": generated_questions[index].split("?")[0] + "?",
                "answer": qg_answers[index]
            }
            qa_list.append(qa)

        return qa_list

    def _get_all_qa_pairs(self, generated_questions: List[str], qg_answers: List[str]):

        qa_list = []

        for question, answer in zip(generated_questions, qg_answers):
            qa = {
                "question": question.split("?")[0] + "?",
                "answer": answer
            }
            qa_list.append(qa)

        return qa_list

class QAEvaluator:

    def __init__(self) -> None:

        QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
        self.qae_model = AutoModelForSequenceClassification.from_pretrained(
            QAE_PRETRAINED
        )
        self.qae_model.to(self.device)
        self.qae_model.eval()

    def encode_qa_pairs(self, questions: List[str], answers: List[str]) -> List[torch.tensor]:

        encoded_pairs = []

        for question, answer in zip(questions, answers):
            encoded_qa = self._encode_qa(question, answer)
            encoded_pairs.append(encoded_qa.to(self.device))

        return encoded_pairs

    def get_scores(self, encoded_qa_pairs: List[torch.tensor]) -> List[float]:

        scores = {}

        for i in range(len(encoded_qa_pairs)):
            scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [
            k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _encode_qa(self, question: str, answer: str) -> torch.tensor:

        if type(answer) is list:
            for a in answer:
                if a["correct"]:
                    correct_answer = a["answer"]
        else:
            correct_answer = answer

        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            padding="max_length",
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def _evaluate_qa(self, encoded_qa_pair: torch.tensor) -> float:
        output = self.qae_model(**encoded_qa_pair, output_hidden_states=False)
        return output[0][0][1]

if __name__ == '__main__':
    # Read the command line arguments
    arguments = sys.argv[1:]

    # Parse the arguments
    parameters = {}
    for arg in arguments:
        key, value = arg.split('=')
        parameters[key] = value

    # Get the value of the question_count parameter
    question_count = int(parameters.get('question_count', '5'))

    # Call the function to generate questions
    q_generator = QuestionGenerator()
    q_generator.generate_questions(question_count)