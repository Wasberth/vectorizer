from flask import render_template

class VacaException(Exception):
    def __init__(self, code, *args):
        super().__init__(*args)
        self.code = code

    def response(self):
        return render_template('error.html', description=str(self), code=self.code), 200