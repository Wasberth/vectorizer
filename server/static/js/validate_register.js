function validate(form){
    if (form.og_pass.value != form.conf_pass.value){
        error_tag = document.getElementById('localerror');
        chamaco = document.createElement('div');
        chamaco.setAttribute('class', 'alert alert-danger my-2');
        chamaco.setAttribute('role', 'alert');
        chamaco.innerHTML = 'Las contraseñas no coindicen.';
        error_tag.appendChild(chamaco);
        return false;
    }
    return true;
}