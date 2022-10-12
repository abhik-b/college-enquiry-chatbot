from flask_server import db

class Teacher(db.Model):
    id = db.Column('faculty_id',db.Integer(), primary_key=True)
    first_name=db.Column(db.String(length=123),nullable=False)
    last_name=db.Column(db.String(length=123),nullable=False)
    department = db.Column(db.String(length=123),nullable=False)
    def __repr__(self):
        return f"{self.first_name} {self.last_name}"



class Holidays(db.Model):
    id = db.Column('holiday_id',db.Integer(), primary_key=True)
    year=db.Column(db.Integer(),nullable=False)
    file_name=db.Column(db.String(length=123),nullable=False)
    data=db.Column(db.LargeBinary())

    def __repr__(self):
        return f"Holidays ID : {self.id} for year : {self.year}"

class Student(db.Model):
    id = db.Column('student_id',db.Integer(),primary_key=True)
    name=db.Column(db.String(length=123),nullable=False)
    cgpa=db.Column(db.String(length=4))
    course_id = db.Column(db.Integer, db.ForeignKey('course.course_id'))
    course = db.relationship('Course', backref='course', lazy=True)

    def __repr__(self):
        return f"{self.name} - {self.course} - {self.cgpa}"

class Course(db.Model):
    id = db.Column('course_id',db.Integer(),primary_key=True)
    name=db.Column(db.String(length=123),nullable=False)
    syllabus=db.Column(db.LargeBinary())
    duration=db.Column(db.String(length=123),nullable=False)

    def __repr__(self):
        return f"{self.name} - {self.duration}"

    