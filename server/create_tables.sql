CREATE TABLE Usuario (
    idUsuario int NOT NULL AUTO_INCREMENT,
    usuario varchar(45) NOT NULL,
    contrasena blob(256) NOT NULL,
    tokens int DEFAULT 0,
    PRIMARY KEY (idUsuario)
);

-- Usuarios de prueba
INSERT INTO Usuario (usuario, contrasena) values ('juanito@alcachofa.com', '0a0b20e465f2c58132a768e711303709532580ee7a6d87f3330104a5ad7919ff'); -- Contraseña: n0m3l0
