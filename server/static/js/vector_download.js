function download_svg(button){
    let form = document.getElementById('config_form');

    let grouping = form.elements.namedItem('agrupacion');
    let style = form.elements.namedItem('estilo');
    let space = form.elements.namedItem('espacios');

    if (grouping.value == 'color' && space.value == 'uncut') {
        alert('No se puede agrupar por color y apilar espacios a la vez.');
        return;
    }

    form.submit();
}