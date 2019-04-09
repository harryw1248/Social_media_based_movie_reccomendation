from flask import render_template, url_for, redirect
import sqlite3
from hashlib import sha256

def login(session, request):
    if session.get('username'):
        return redirect(url_for('homepage'))
    #print request.form
    try:
        username = request.form['username']
        password = request.form['password']
    except:
        return render_template('homepage.html', error=True, msg="Error.")
    #print username
    #print password
    if request.form.get("register"):
        valid,msg = register(username, password)
        return render_template('homepage.html', error=not valid, msg=msg)
    if request.form.get("login"):
        valid,msg = auth_user(username, password)
        if valid:
            session['username'] = username
            return redirect(url_for('homepage'))
        return render_template("homepage.html", error=not valid, msg=msg)
    return redirect(url_for('homepage'))

def register(user, pw):
    f = "data/data.db"
    db = sqlite3.connect(f, check_same_thread=False)
    c = db.cursor()

    c.execute('SELECT * FROM users WHERE username = ?', (user,))
    if len(list((c))) != 0:
        print "User %s exists" % user
        db.close()
        return False,"Username taken"
    pw = sha256(pw).hexdigest()
    c.execute('INSERT INTO users VALUES (?,?)', (user, pw))
    db.commit()
    db.close()
    print "Registered successfully"
    return True,"Successfully registered"

def auth_user(user, pw):
    f = "data/data.db"
    db = sqlite3.connect(f, check_same_thread=False)
    c = db.cursor()

    pw = sha256(pw).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? and password = ?',
            (user, pw))
    if len(list((c))) == 1:
        db.close()
        print "Login successful"
        return True,"Successfully logged in"
    db.close()
    print "Login failed"
    return False,"Bad user/pass combo"
