#!/usr/bin/env python
# coding: utf-8

# In[1]:


def count_and_print_vowels(sentence):
    vowels = 'aeiouAEIOU'
    vowel_counts = {vowel: 0 for vowel in vowels}

    for char in sentence:
        if char in vowels:
            vowel_counts[char.lower()] += 1

    for vowel, count in vowel_counts.items():
        if count > 0:
            print(f"{vowel}: {count}")

# Example usage:
sentence = "This is a sample sentence."
count_and_print_vowels(sentence)


# In[2]:


def censor_word(sentence, word, placeholder='****'):
    return sentence.replace(word, placeholder)

# Example usage:
sentence = "Please don't use inappropriate language."
word_to_censor = "inappropriate"
censored_sentence = censor_word(sentence, word_to_censor)
print(censored_sentence)


# In[3]:


def convert_to_grades(scores):
    grade_scale = {90: 'A', 80: 'B', 70: 'C', 60: 'D', 0: 'F'}
    grades = []

    for name, score in scores:
        for cutoff, grade in grade_scale.items():
            if score >= cutoff:
                grades.append((name, grade))
                break

    return grades

# Example usage:
student_scores = [("Alice", 85), ("Bob", 75), ("Charlie", 92)]
grades = convert_to_grades(student_scores)
print(grades)


# In[4]:


def organize_tasks(tasks):
    completed_tasks = []
    pending_tasks = []

    for task in tasks:
        if "(completed)" in task:
            completed_tasks.append(task)
        else:
            pending_tasks.append(task)

    return completed_tasks, pending_tasks

# Example usage:
tasks = ["Wake up (completed)", "Sleep", "Eat"]
completed, pending = organize_tasks(tasks)
print("Completed tasks:", completed)
print("Pending tasks:", pending)


# In[5]:


preprocess_data = lambda data: [(x ** 2) for x in data if x % 2 != 0]

# Example usage:
numbers = [1, 2, 3, 4, 5]
processed_numbers = preprocess_data(numbers)
print(processed_numbers)


# In[6]:


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Example usage:
doc1_words = set(["apple", "banana", "orange"])
doc2_words = set(["banana", "kiwi", "orange"])
similarity = jaccard_similarity(doc1_words, doc2_words)
print("Jaccard similarity:", similarity)


# In[ ]:




