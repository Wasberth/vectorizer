let color_clusters = null;
let lab_image = null;
let image_tag = document.getElementById('preprocessed');
let loading_text = document.getElementById('loading-text');
let url = image_tag.getAttribute('loading-url');
let img_canvas = document.getElementById('img-canvas');
let loaded = false;
let sending = false;

function labToRgb(l_opencv, a_opencv, b_opencv) {
    // Desnormalizar a LAB real
    const L = l_opencv * (100 / 255);  // OpenCV L (0-255) -> L (0-100)
    const a = a_opencv - 128;          // OpenCV A (0-255) -> a (-128-127)
    const b = b_opencv - 128;          // OpenCV B (0-255) -> b (-128-127)

    // LAB -> XYZ
    let Y = (L + 16) / 116;
    let X = a / 500 + Y;
    let Z = Y - b / 200;

    const ref_X = 95.047;
    const ref_Y = 100.000;
    const ref_Z = 108.883;

    const pow = Math.pow;
    const f_inv = (t) => (pow(t, 3) > 0.008856) ? pow(t, 3) : (t - 16/116) / 7.787;

    X = ref_X * f_inv(X);
    Y = ref_Y * f_inv(Y);
    Z = ref_Z * f_inv(Z);

    // XYZ -> RGB
    X /= 100; Y /= 100; Z /= 100;
    let r = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
    let g = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
    let b_ = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

    const adjust = c => (c > 0.0031308)
      ? 1.055 * Math.pow(c, 1/2.4) - 0.055
      : 12.92 * c;

    r = adjust(r);
    g = adjust(g);
    b_ = adjust(b_);

    // Escalar a 0–255 y limitar
    const clamp = (val) => Math.min(Math.max(0, val * 255), 255);

    return {
        r: Math.round(clamp(r)),
        g: Math.round(clamp(g)),
        b: Math.round(clamp(b_))
    };
}

function create_button(r, g, b, id){
    boton = document.createElement('BUTTON');
    boton.setAttribute('class', 'btn rounded-circle p-3 m-2');
    boton.setAttribute('style', 'width: 3.75rem; background-color: rgb('+r+', '+g+', '+b+') !important;');
    boton.setAttribute('type', 'button');
    boton.setAttribute('state', 'enabled');
    boton.setAttribute('id', 'boton_'+id)
    boton.setAttribute('onclick', 'update_img('+r+', '+g+', '+b+', '+id+')');
    boton.innerHTML = id+1;
    return boton;
}

function update_img(r, g, b, cluster_id){
    if (activated){
        activated = false;
        boton = document.getElementById('boton_'+cluster_id);
        if (boton.getAttribute('state') == 'enabled'){
            boton.setAttribute('style', 'width: 3.75rem; border-color: rgb('+r+', '+g+', '+b+') !important; color: rgb('+r+', '+g+', '+b+') !important;');
            boton.setAttribute('state', 'disabled');
        } else {
            boton.setAttribute('style', 'width: 3.75rem; background-color: rgb('+r+', '+g+', '+b+') !important;');
            boton.setAttribute('state', 'enabled');
        }
        draw_canvas();
        activated = true;
    }
}

function distanciaEuclidiana3D(p1, p2) {
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const dz = p2[2] - p1[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function draw_canvas(){
    width = img_canvas.getAttribute('width');
    height = img_canvas.getAttribute('height');
    let ctx = img_canvas.getContext('2d');
    for (let i = 0; i < width; i++) {
        for (let j = 0; j < height; j++) {
            cluster_class = -1;
            cluster_distance = -1;
            for (let k = 0; k < color_clusters.length; k++) {
                if (document.getElementById('boton_'+k).getAttribute('state') == 'enabled'){
                    current_distance = distanciaEuclidiana3D(lab_image[j][i], color_clusters[k]);
                    if (cluster_distance == -1){
                        cluster_class = k;
                        cluster_distance = current_distance;
                        continue;
                    }
                    if (current_distance < cluster_distance){
                        cluster_distance = current_distance;
                        cluster_class = k;
                    }
                }
            }
            rgb = labToRgb(color_clusters[cluster_class][0], color_clusters[cluster_class][1], color_clusters[cluster_class][2]);
            ctx.fillStyle = 'rgb('+rgb.r+', '+rgb.g+', '+rgb.b+')';
            ctx.fillRect(i, j, 1, 1);
        }
    }
}

function load_image(){
    fetch(url, {
        method:'POST'
    })
    .then(response => response.json())
    .then(data => {
        color_clusters = data.centroides;
        boton_group = document.getElementById('color_boton_group');
        for (let i = 0; i < color_clusters.length; i++) {
            const cluster_center = color_clusters[i];
            rgb_coded = labToRgb(cluster_center[0], cluster_center[1], cluster_center[2])
            boton = create_button(rgb_coded.r, rgb_coded.g, rgb_coded.b, i);
            boton_group.appendChild(boton);
        }
        if (data.estado == 'SR'){
            image_tag.setAttribute('src', image_tag.getAttribute('src') + '?' + new Date().getTime());
            loading_text.innerHTML = 'Suavizando curvas';
            url = data.siguiente;
            get_SR();
        }else if (data.estado == 'exito'){
            image_tag.setAttribute('style', 'display: none;');
            img_canvas.removeAttribute('style');
            img_canvas.setAttribute('width', data.width);
            img_canvas.setAttribute('height', data.height);
            lab_image = data.pixels;
            draw_canvas();
            activated = true;
        }
    })
    .catch(error => {
        console.error('Algo salió mal ', error);
    });
}

function get_SR(){
    fetch(url, {
        method:'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.estado == 'exito'){
            image_tag.setAttribute('src', image_tag.getAttribute('src') + '?' + new Date().getTime());
            loading_text.innerHTML = 'Limpiando resultados';
            url = data.siguiente;
            get_image_SR();
        }
    })
    .catch(error => {
        console.error('Algo salió mal ', error);
    });
}

function get_image_SR(){
    fetch(url, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data =>{
        if (data.estado == 'exito'){
            document.getElementById('loader').style.display = 'None';
            document.getElementById('loader-background').style.display = 'None';
            image_tag.setAttribute('style', 'display: none;');
            img_canvas.removeAttribute('style');
            img_canvas.setAttribute('width', data.width);
            img_canvas.setAttribute('height', data.height);
            lab_image = data.pixels;
            draw_canvas();
            activated = true;
        }
    })
    .catch(error => {
        console.error('Algo salió mal ', error);
    });
}

function vectorizar(){
    if(activated){
        if(!sending){
            activated = false;
            send_url = document.getElementById('send_vec').getAttribute('post-action');
            body_send = []
            for (let k = 0; k < color_clusters.length; k++) {
                if (document.getElementById('boton_'+k).getAttribute('state') == 'enabled'){
                    // El cambio es para mandar la información de que el usuario cambió el color, si no lo cambió se queda en 'no'
                    body_send.push({centroide: color_clusters[k], cambio:'no'});
                }
            }
            fetch(send_url, {
                method:'POST',
                body:JSON.stringify(body_send)
            })
            .then(response => response.json())
            .then(data => {
                if(data.estado == 'exito'){
                    window.location.replace(data.siguiente);
                }
            })
            .catch(error => {
                console.error('Algo salió mal ', error);
                activated = true;
            });
        }
    }
}

function cambiarColores(){
    if(activated){
        activated = false;
        let cantidad = document.getElementById('cantidadColores').value;
        let request_body = {numero: cantidad}
        let request_url = document.getElementById('cantidadColores').getAttribute('post-url');
        fetch(request_url,{
            method:'POST', 
            body:JSON.stringify(request_body)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('loader').removeAttribute('style');
            document.getElementById('loader-background').removeAttribute('style');
            image_tag.removeAttribute('style');
            img_canvas.setAttribute('style', 'display: none;');
            boton_group = document.getElementById('color_boton_group');
            let child = boton_group.lastElementChild;
            while (child) {
                boton_group.removeChild(child);
                child = boton_group.lastElementChild;
            }
            color_clusters = data.centroides;
            for (let i = 0; i < color_clusters.length; i++) {
                const cluster_center = color_clusters[i];
                rgb_coded = labToRgb(cluster_center[0], cluster_center[1], cluster_center[2])
                boton = create_button(rgb_coded.r, rgb_coded.g, rgb_coded.b, i);
                boton_group.appendChild(boton);
            }
            if (data.estado == 'SR'){
                image_tag.setAttribute('src', image_tag.getAttribute('src') + '?' + new Date().getTime());
                loading_text.innerHTML = 'Suavizando curvas';
                url = data.siguiente;
                get_SR();
            }else if (data.estado == 'exito'){
                image_tag.setAttribute('style', 'display: none;');
                img_canvas.removeAttribute('style');
                img_canvas.setAttribute('width', data.width);
                img_canvas.setAttribute('height', data.height);
                lab_image = data.pixels;
                draw_canvas();
                activated = true;
            }
        })
        .catch(error => {
            console.error('Algo salió mal ', error);
            activated = true;
        })
    }
}

load_image()