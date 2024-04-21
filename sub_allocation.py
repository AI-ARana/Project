""" Subject Allocation"""
import csv

class Teacher:
    def __init__(self, name, su_id,designation, max_load):
        self.name = name
        self.designation = designation
        self.su_id=su_id
        self.max_load = max_load
        self.current_load = 0


class Subject:
    def __init__(self, course,sem,subjects, Credits_T,Credits_P, Theory_Hrs, Group_T, TT_Hrs, Practical_Hrs, Group_P, TP_Hrs):
        self.course=course
        self.sem=sem
        self.subjects = subjects
        self.Credits_T = Credits_T
        self.Credits_P = Credits_P
        self.Theory_Hrs = Theory_Hrs
        self.Group_T=Group_T
        self.TT_Hrs=TT_Hrs
        self.Practical_Hrs = Practical_Hrs
        self.Group_P=Group_P
        self.TP_Hrs=TP_Hrs
        self.teachers = []


def read_faculty_data(file_path):
    teachers = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            teacher = Teacher(row['name'], row['su_id'],row['designation'], int(row['max_load']))
            teachers.append(teacher)
    return teachers


def read_subject_data(file_path):
    subjects = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject = Subject(row['course'],int(row['sem']), row['subjects'], int(row['Credits_T']),int(row['Credits_P']), int(row['Theory_Hrs']),
                              int(row['Group_T']), int(row['TT_Hrs']), int(row['Practical_Hrs']), int(row['Group_P']), 
                              int(row['TP_Hrs']))
            subjects.append(subject)
    return subjects


def allocate_subjects_to_teachers(teachers, subjects):
    for teacher in teachers:
        for subject in subjects:
            if subject.Group_T and subject.Group_P and teacher.current_load < teacher.max_load:
                teacher.current_load += subject.Theory_Hrs + subject.Practical_Hrs
                subject.TT_Hrs -= 1
                subject.TP_Hrs -= 1
                subject.teachers.append(teacher)


"""def print_subject_allocation(subjects):
    for subject in subjects:
        print(f"Subject: {subject.subjects}")
        print("Assigned Teachers:")
        for teacher in subject.teachers:
            print(f"- {teacher.name} ({teacher.designation})")
        print()
"""


# Read data from CSV files
teachers = read_faculty_data(r'F:\pdataset\faculty.csv')
subjects = read_subject_data(r'F:\pdataset\subject.csv')

# Allocate subjects to teachers
allocate_subjects_to_teachers(teachers, subjects)

# Print subject allocation
#print_subject_allocation(subjects)
def write_subject_allocation_to_csv(subjects, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ "Assigned Teacher", "Teacher Designation","Subject","Course","Semester", "Credits","Theory_Hrs","Practical_Hrs"])
        for subject in subjects:
            for teacher in subject.teachers:
                writer.writerow([ teacher.name, teacher.designation,subject.subjects,subject.course,subject.sem,subject.Theory_Hrs,subject.Practical_Hrs])

# Usage
write_subject_allocation_to_csv(subjects, 'f:\pdataset\output.csv')
