USE PassLi;
GO

INSERT INTO Horarios
(GrupoID, SalonID, MateriaID, MaestroID, DiaSemana, HoraInicio, HoraFin)
VALUES
(1, 2, 1, 1, 'Martes', '07:00', '07:50'),
(1, 18, 2, 2, 'Martes', '07:50', '08:40'),
(1, 18, 6, 5, 'Martes', '08:40', '09:30'),
(1, 3, 7, 6, 'Martes', '10:00', '10:50'),
(1, 4, 3, 3, 'Martes', '10:50', '11:40'),
(1, 14, 4, 7, 'Martes', '11:40', '12:30'),
(1, 9, 5, 4, 'Martes', '12:30', '13:20'),
(1, 7, 8, 6, 'Martes', '13:20', '14:10');


INSERT INTO Horarios
(GrupoID, SalonID, MateriaID, MaestroID, DiaSemana, HoraInicio, HoraFin)
VALUES
(1, 7, 1, 1, 'Lunes', '07:00', '07:50'),
(1, 1, 3, 3, 'Lunes', '07:50', '08:40'),
(1, 9, 2, 2, 'Lunes', '08:40', '09:30'),
(1, 1, 6, 5, 'Lunes', '10:00', '10:50'),
(1, 5, 7, 6, 'Lunes', '10:50', '11:40'),
(1, 9, 5, 4, 'Lunes', '11:40', '12:30'),
(1, 14, 4, 7, 'Lunes', '12:30', '13:20');


INSERT INTO Maestros (NombreCompleto)
VALUES
('Rubio Monserrat'),
('César E. Inda'),
('Medina Héctor'),
('Carlos D. Ortiz'),
('Lizbeth Ibarra'),
('Sol Mar Ramirez'),
('Stephany Lopéz'),
('Nadia Adale'),
('Miriam Ibarra'),
('Juan M. Ramirez');


INSERT INTO Materias (NombreMateria)
VALUES
('Ecuaciones Diferenciales'),
('Aprendizaje de Máquinas'),
('Inglés V'),
('Minería de Datos'),
('Fundamentos de Visión'),
('Proyecto Integrador II'),
('Liderazgo de Equipo'),
('Tutoria Grupal'),
('Base de Datos'),
('App. Web'),
('Estand. y Metricas');