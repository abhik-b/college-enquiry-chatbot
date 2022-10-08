from flask_server import db

class Teacher(db.Model):
    id = db.Column('faculty_id',db.Integer(), primary_key=True)
    first_name=db.Column(db.String(length=123),nullable=False)
    last_name=db.Column(db.String(length=123),nullable=False)
    department = db.Column(db.String(length=123),nullable=False)
    def __repr__(self):
        return f"Faculty ID : {self.id} \n Name : {self.first_name} {self.last_name}"



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

    def __repr__(self):
        return f"{name} - course - {cgpa}"
