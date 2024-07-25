"""
- dummy function yerine test.py main i call etmek gerekiyor, 
- test.py debug edilmesi lazim, oncelikle onun end-to-end calistigindan emin olun.
- adding stuff like email API, graphic and stuff
"""


## collab de run edilecekse, once sunu ekle:
# from google.colab.output import eval_js
# print(eval_js("google.colab.kernel.proxyPort(5000)"))


from flask import Flask, render_template, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email
from test import main
from uuid import uuid4
from time import time
import os

app = Flask(__name__)
app.config["Results"] = "results"
app.secret_key = 'tO$&!|0wkamvVia0?n$NqIRVWOG'
csrf = CSRFProtect(app)

class NameForm(FlaskForm):
    name = StringField('Your e-mail address', validators=[DataRequired(), Email("Please enter a valid e-mail")])
    record = BooleanField("Record")
    submit = SubmitField('Submit')

class DownloadForm(FlaskForm):

    submit = SubmitField('Download results')

    


def dummyfunction(email):
    # Dummy function that processes the email address
    return f"Processing email: {email}"

@app.route("/", methods=["POST", "GET"])
def index():
    form = NameForm()

    if form.validate_on_submit():
        # Get the value from the form's name field
        email_address = form.name.data
        # Call the dummy function with the email address
        result = dummyfunction(email_address)  
        # See if recording == True
        record = form.record.data

        if record:
            result = main()
            file_id = uuid4()
            file_path = f"{file_id}_results.tsv"
            full_path = os.path.join(app.config['Results'], file_path)
            result.to_csv(full_path, sep = "\t")
           
            return send_from_directory(app.config['Results'],file_path, as_attachment = True)
            # somehow redirect to page before starting record or indicate that you started recording, add stop recording funtion, save the data as csv and plotting?, optional: sent to email using api (sendgrid?) 
           

        return render_template("processing.html", result=result, record = record)

    return render_template("layout.html", form=form)



if __name__ == "__main__" :
   # train_model()
   
   app.run()