from .models import Teacher,Holidays
from flask_server import db,app
from flask import render_template,request,jsonify,redirect,url_for,Blueprint,send_file
from io import BytesIO

@app.route("/")
def hello_world():
    return render_template('home.html')

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