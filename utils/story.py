from flask import render_template, url_for, redirect
import sqlite3
from hashlib import sha256
from base64 import b64encode, b64decode

def add_story(session, request):
    user = session['username']
    title = request.form['title']
    story = request.form['story']

    f = "data/data.db"
    db = sqlite3.connect(f)
    c = db.cursor()

    c.execute('SELECT * FROM stories WHERE title = ?', (title,))
    if c.fetchone():
        return render_template("addstory.html", msg="Title taken")
    c.execute('SELECT * FROM stories')
    l = c.fetchall()
    lastID = l[-1][1] if l else 0
    c.execute("INSERT INTO stories VALUES (?,?,?,?)",
                (title, lastID+1, b64encode(user), b64encode(story)))
    db.commit()
    db.close()
    return redirect(url_for('homepage'))

def fetch_stories(user):
    def tablify(s):
        l = s.split(";")
        l = [b64decode(e) for e in l]
        return l

    f = "data/data.db"
    db = sqlite3.connect(f)
    c = db.cursor()

    c.execute('SELECT * FROM stories')
    result = c.fetchall()
    stories = []
    for e in result:
        updaters = tablify(e[2])
        updates = tablify(e[3])
        stories.append((e[0],e[1],updaters,updates,user in updaters if user else False))
    return stories

def update_story(session, request):
    user = session['username']
    sID = request.form['storyID']
    update = request.form['update']
    f = "data/data.db"
    db = sqlite3.connect(f)
    c = db.cursor()

    c.execute('SELECT updaters,updates FROM stories WHERE id = ?', (sID,))
    e = c.fetchone()
    if b64decode(e[0]) == user:
        return render_template("homepage.html", error=True, msg="Already contributed to story", user=user)

    c.execute('UPDATE stories SET updaters = ?, updates = ? WHERE id = ?',
                (e[0]+";"+b64encode(user), e[1]+";"+b64encode(update), sID))
    db.commit()
    db.close()
    return redirect(url_for('homepage'))
