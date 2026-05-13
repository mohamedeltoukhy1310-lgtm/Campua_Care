"""Generate a large custom CampusCare dataset.

The dataset is synthetic but designed from realistic university support-message patterns.
It is intended for an Artificial Neural Networks course project where custom dataset
creation is allowed/encouraged. It should not be used as a clinical dataset.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "campuscare_large_dataset.csv"
RANDOM_SEED = 42

LABELS = ["normal", "academic_stress", "emotional_distress", "urgent_human_review"]
DEFAULT_COUNTS = {
    "normal": 24000,
    "academic_stress": 18000,
    "emotional_distress": 12000,
    "urgent_human_review": 6000,
}

courses = [
    "calculus", "linear algebra", "programming", "data structures", "AI", "databases",
    "networks", "software engineering", "statistics", "physics", "chemistry", "biology",
    "economics", "accounting", "digital logic", "operating systems", "machine learning",
    "web development", "cybersecurity", "research methods"
]
assignments = ["assignment", "quiz", "lab", "project", "midterm", "final exam", "presentation", "report", "case study", "practical task"]
channels = ["anonymous form", "mobile app", "email", "advising portal", "chatbot", "course feedback form"]
years = ["freshman", "sophomore", "junior", "senior", "postgraduate"]
times = ["this morning", "today", "this week", "before the deadline", "after class", "during exams", "late at night", "over the weekend"]
requests = [
    "can someone guide me", "I need advice", "please tell me the next step", "I want support",
    "could the advisor contact me", "I need a clearer plan", "can this be reviewed", "please help me understand what to do"
]

normal_templates = [
    "I attended the {course} lecture {time} and the explanation was clear. Thank you.",
    "Can you confirm the room for the {assignment} of {course}? I do not have a problem, I just need the location.",
    "The {channel} is working well for my {course} group and I can access the materials.",
    "I submitted the {assignment} for {course}. Please confirm it was received.",
    "The new timetable is helpful and I can organize my study plan better now.",
    "I have a simple question about office hours for {course}, nothing urgent.",
    "The lab assistant helped us during {course} and the session was useful.",
    "The deadline extension announcement was clear. I appreciate the update.",
    "I want to give positive feedback about the teaching style in {course}.",
    "The campus app notification about {assignment} arrived on time.",
    "My group finished the {assignment} and we only need confirmation about the submission format.",
    "The recording for {course} helped me revise the topic.",
    "I found a small typo in the {assignment} instructions, but I understand the task.",
    "The advisor answered my registration question and everything is okay now.",
    "I am checking whether the {assignment} rubric for {course} has been uploaded.",
    "The library resources for {course} are useful and easy to access.",
    "The course page for {course} opened successfully after the update.",
    "I would like more examples in {course}, but overall the class is going well.",
    "The feedback on my previous {assignment} was helpful.",
    "I need the lecture slides for {course}. There is no emergency.",
]

academic_templates = [
    "I feel overwhelmed because the {course} {assignment} and another deadline are both due {time}.",
    "I studied for {course} but I panic when I think about the {assignment}. {request}.",
    "My grades dropped in {course} and I am scared I might fail this semester.",
    "I cannot balance work, family, and the {course} project. I need academic support.",
    "The workload this week is too much and I am anxious about finishing the {assignment}.",
    "I keep rereading the {course} material but I still do not understand it and I feel stressed.",
    "I missed two lectures in {course} and now I am behind. {request}.",
    "The {assignment} instructions are confusing and the deadline pressure is making me panic.",
    "I am afraid to ask questions in class, but I am lost in {course}.",
    "I have three exams and a {course} {assignment} in the same week. I need a study plan.",
    "My attendance in {course} is low because of transport issues and I am worried about failing.",
    "I am not sleeping well before exams because I keep thinking about my marks.",
    "I want tutoring support for {course}; I am trying but still behind.",
    "I feel burned out from continuous quizzes and deadlines, especially in {course}.",
    "The final project for {course} feels impossible for me to complete alone.",
    "I need help prioritizing the {assignment} tasks because I am overwhelmed.",
    "I become very nervous when I open the grade portal after {course} assessments.",
    "I studied many hours but my practice scores in {course} are still low and I am frustrated.",
    "I am under academic probation and this {assignment} is adding too much pressure.",
    "The group project is not progressing and I am anxious about my final grade.",
]

emotional_templates = [
    "I feel isolated on campus and it is hard to focus on {course} or talk to classmates.",
    "For several weeks I have felt empty and tired, even when I finish my {assignment}.",
    "I keep losing motivation and I do not enjoy anything at university anymore.",
    "I feel sad most days and I avoid replying to messages from my friends.",
    "I am struggling emotionally and my concentration in {course} is getting worse.",
    "I cry often after classes and I do not know who I should talk to.",
    "I feel like I am disappointing everyone even when I try my best.",
    "I am always exhausted and disconnected from my classmates.",
    "I feel lonely in the dorm and I need someone to listen without judging me.",
    "I have been anxious and low for a long time, not just because of one {assignment}.",
    "My mood changes quickly and I cannot control my stress during the day.",
    "I do not feel comfortable talking to my family about how bad I feel.",
    "I am losing confidence and I feel worthless when I compare myself with others.",
    "I often skip meals and stay in my room because I feel down.",
    "I feel emotionally drained and I need counseling or someone safe to talk to.",
    "I am finding it hard to get out of bed for classes, including {course}.",
    "I feel constant pressure and sadness that does not go away after exams.",
    "I need mental health support because I feel hopeless about my semester.",
    "I feel numb and disconnected from everything happening around me.",
    "I am not in immediate danger, but I need emotional support soon.",
]

urgent_templates = [
    "I do not feel safe with myself tonight and I need immediate human help.",
    "I might hurt myself if nobody contacts me soon. Please escalate this now.",
    "This is urgent: I am thinking about ending my life and I need support right now.",
    "I am alone and scared that I may do something dangerous to myself.",
    "Please ask a counselor to contact me immediately; I cannot stay safe on my own.",
    "I feel at risk of harming myself today and need emergency support.",
    "I wrote goodbye messages and I need someone to intervene now.",
    "I cannot promise I will be safe tonight. Please send urgent help.",
    "I am having thoughts of suicide and I need a trained person to call me now.",
    "I feel like I may hurt myself after leaving campus. Please escalate to human review.",
    "I need urgent help because I have thoughts about not wanting to live anymore.",
    "Please do not leave this message for tomorrow. I need immediate support.",
    "I am in crisis and I need emergency counseling now.",
    "I am afraid of what I might do to myself if I remain alone.",
    "I need someone to check on me now because I do not feel safe.",
]

hedges = [
    "maybe", "honestly", "to be clear", "right now", "recently", "sometimes", "for the first time", "again", "almost every day", "since last month"
]
intensifiers = ["a little", "very", "extremely", "quite", "really", "somewhat", "seriously", "constantly", "slightly"]
noise_phrases = [
    "sent from mobile", "sorry for the long message", "I am a {year} student", "student ID hidden",
    "English is not my first language", "I hope this is the right place", "please keep this private",
    "I tried to solve it first", "I already checked the portal", "I am writing through the {channel}"
]

severity_ranges = {
    "normal": (0.00, 0.25),
    "academic_stress": (0.30, 0.58),
    "emotional_distress": (0.55, 0.78),
    "urgent_human_review": (0.85, 1.00),
}

def choose_template(label: str) -> str:
    if label == "normal":
        return random.choice(normal_templates)
    if label == "academic_stress":
        return random.choice(academic_templates)
    if label == "emotional_distress":
        return random.choice(emotional_templates)
    return random.choice(urgent_templates)


def make_message(label: str) -> str:
    template = choose_template(label)
    text = template.format(
        course=random.choice(courses), assignment=random.choice(assignments), channel=random.choice(channels),
        time=random.choice(times), request=random.choice(requests), year=random.choice(years)
    )
    # Add realistic optional metadata/noise without making labels trivial.
    if random.random() < 0.55:
        text = f"{random.choice(hedges).capitalize()}, {text[0].lower() + text[1:]}"
    if random.random() < 0.45:
        text += " " + random.choice(noise_phrases).format(year=random.choice(years), channel=random.choice(channels)) + "."
    if random.random() < 0.18 and label != "urgent_human_review":
        text += " This is not an emergency."
    if random.random() < 0.22:
        text = text.replace("very", random.choice(intensifiers))
    if random.random() < 0.08:
        text = text.replace("I ", "i ")
    if random.random() < 0.08:
        text += " Please reply when possible."
    return " ".join(text.split())


def generate_dataset(counts: dict[str, int], seed: int = RANDOM_SEED) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    seen_by_label = {label: set() for label in LABELS}
    for label, n in counts.items():
        attempts = 0
        while len(seen_by_label[label]) < n:
            attempts += 1
            text = make_message(label)
            # Use metadata as separate features instead of injecting IDs into text.
            if text in seen_by_label[label]:
                continue
            seen_by_label[label].add(text)
            low, high = severity_ranges[label]
            severity = round(random.uniform(low, high), 3)
            row = {
                "message_id": f"CCAI-{label[:3].upper()}-{len(seen_by_label[label]):05d}",
                "text": text,
                "label": label,
                "severity_score": severity,
                "channel": random.choice(channels),
                "student_year": random.choice(years),
                "source": "custom_large_synthetic_campuscare",
                "is_synthetic": True,
            }
            rows.append(row)
            if attempts > n * 100:
                raise RuntimeError(f"Could not generate enough unique samples for {label}")
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "row_id", range(1, len(df) + 1))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=sum(DEFAULT_COUNTS.values()), help="Total rows. Class proportions are preserved.")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    total_default = sum(DEFAULT_COUNTS.values())
    counts = {k: int(round(v / total_default * args.rows)) for k, v in DEFAULT_COUNTS.items()}
    # fix rounding to exact requested rows
    diff = args.rows - sum(counts.values())
    counts["normal"] += diff
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_dataset(counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
