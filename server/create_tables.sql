CREATE TABLE Usuario (
    idUsuario int NOT NULL AUTO_INCREMENT,
    usuario varchar(45) NOT NULL,
    contrasena blob(256) NOT NULL,
    tokens int DEFAULT 0,
    PRIMARY KEY (idUsuario)
);

-- Usuarios de prueba
INSERT INTO Usuario (usuario, contrasena) values ('juanito@gmail.com', '0a0b20e465f2c58132a768e711303709532580ee7a6d87f3330104a5ad7919ff'); -- Contraseña: n0m3l0
INSERT INTO Usuario (usuario, contrasena) values ('pablito@gmail.com', '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4'); -- Contraseña: 1234
INSERT INTO Usuario (usuario, contrasena) values ('valeria@gmail.com', '65e84be33532fb784c48129675f9eff3a682b27168c0ea744b2cf58ee02337c5'); -- Contraseña: qwerty
INSERT INTO Usuario (usuario, contrasena) values ('sebastian@gmail.com', 'f0e4c2f76c58916ec258f246851bea091d14d4247a2fc3e18694461b1816e13b'); -- Contraseña: asdf
INSERT INTO Usuario (usuario, contrasena) values ('diego@gmail.com', '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'); -- Contraseña: password
INSERT INTO Usuario (usuario, contrasena) values ('paulita@hotmail.com', '88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589'); -- Contraseña: abcd
INSERT INTO Usuario (usuario, contrasena) values ('adrian@hotmail.com', '116eb98a352cd1a7cdd54dbfdade518ec2ce5965f02133170ee89c1450bb6565'); -- Contraseña: lechuga
INSERT INTO Usuario (usuario, contrasena) values ('marquitos@hotmail.com', '0f052edd78dd1a7e23405ebb808a0d94cc92a20c40bd2d983ec79aeee0205235'); -- Contraseña: insegura
INSERT INTO Usuario (usuario, contrasena) values ('carlos@outlook.com', '7b754e35f71116bde3991b68b00656ce37a5d91343e068f1a7cae73a24435d59'); -- Contraseña: unchiste
INSERT INTO Usuario (usuario, contrasena) values ('emilio@outlook.com', '251bf48305b8ed943144ee7a1b71f0446cd23596c80a193db1019e849ca42096'); -- Contraseña: inusual
