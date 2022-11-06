from .models import Teacher,Holidays,Student,Course
from flask_server import db,app
from flask import render_template,request,jsonify,redirect,url_for,Blueprint,send_file
from io import BytesIO

@app.route("/")
def hello_world():
    return render_template('home.html')
@app.route("/home2/")
def hello_world2():
    return render_template('home2.html')

@app.route("/teachers/",methods = ['POST', 'GET'])
def teachers():
    if request.method == 'POST':
        first_name = request.form['firstname']
        last_name = request.form['lastname']
        department = request.form['department']

        new_teacher = Teacher(first_name=first_name,last_name=last_name,department=department)
        db.session.add(new_teacher)
        db.session.commit()
        teachers = Teacher.query.all()
        return redirect(url_for('teachers'))
    
    teachers = Teacher.query.all()
    return render_template('teachers.html',teachers=teachers)


@app.route("/teachers/delete/<int:id>/")
def teachersdelete(id):
    teacher = Teacher().query.get(id)
    db.session.delete(teacher)
    db.session.commit()
        
    teachers = Teacher.query.all()
    return redirect(url_for('teachers'))


@app.route("/teachers/api/")
def teachers_api():
    teachers = Teacher.query.all()
    return jsonify([
        {
            "name":teacher.first_name+" "+teacher.last_name,
            "department":teacher.department,
        } for teacher in teachers
    ])


@app.route("/teachers/api/<string:dept>/")
def dept_teachers_api(dept):
    teachers = Teacher.query.filter(Teacher.department.ilike(f"%{dept}%")).all()
    return jsonify([
        {
            "name":teacher.first_name+" "+teacher.last_name,
            "department":teacher.department,
        } for teacher in teachers
    ])


# =============================
# HOLIDAYS
# =============================

@app.route("/holidays/",methods = ['POST', 'GET'])
def holidays():
    if request.method == 'POST':
        year = request.form['year']
        data = request.files['file']
        new_holiday = Holidays(year=year,file_name=data.filename,data=data.read())
        db.session.add(new_holiday)
        db.session.commit()
        print(f"{new_holiday} added")
        return redirect(url_for('holidays'))
    
    holidays = Holidays.query.all()
    return render_template('holidays.html',holidays=holidays)

@app.route("/holidays/download/<int:id>/")
def holidays_file_api(id):
    holiday = Holidays.query.filter_by(id=id).first()
    return send_file(BytesIO(holiday.data),download_name=holiday.file_name)


# ====================================
#  STUDENTS 
# ====================================


@app.route("/students/",methods = ['POST', 'GET'])
def students():
    if request.method == 'POST':
        studentID = request.form['studentID']
        name = request.form['name']
        courseID = request.form['course']

        new_student = Student(id=studentID,name=name,course_id=courseID)
        db.session.add(new_student)
        db.session.commit()
        students = Student.query.all()
        return redirect(url_for('students'))
    
    students = Student.query.all()
    courses = Course.query.all()
    return render_template('students.html',students=students,courses=courses)

@app.route("/students/update/<int:id>/",methods=['POST', 'GET'])
def studentsupdate(id):
    student = Student().query.get(id)
    
    if request.method == 'POST':
        student.cgpa=request.form['cgpa']
        student.name=request.form['name']
        student.id=request.form['studentID']
        # student.update(request.form)
        db.session.commit()
        return redirect(url_for('students'))
    
    return render_template('student_update.html',student=student)
    
# ====================================
#  COURSES 
# ====================================



@app.route("/courses/",methods = ['POST', 'GET'])
def courses():
    if request.method == 'POST':
        name = request.form['name']
        duration = request.form['duration']
        syllabus = request.files['file']

        new_course = Course(name=name,duration=duration,syllabus=syllabus.read())
        db.session.add(new_course)
        db.session.commit()
        return redirect(url_for('courses'))
    
    courses = Course.query.all()
    return render_template('courses.html',courses=courses)

@app.route("/courses/update/<int:id>/",methods=['POST', 'GET'])
def coursesupdate(id):
    course = Course().query.get(id)
    
    if request.method == 'POST':
        syllabus_file=request.files['file']
        course.syllabus=syllabus_file.read()
        # course.update(request.form)
        db.session.commit()
        return redirect(url_for('courses'))
    
    return render_template('course_update.html',course=course)


@app.route("/courses/syllabus/<int:id>/")
def syllabus_api(id):
    course = Course.query.filter_by(id=id).first()
    return send_file(BytesIO(course.syllabus),download_name=f"{course.name}.pdf")