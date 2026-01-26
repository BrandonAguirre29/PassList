CREATE DATABASE PassLi;
GO

USE PassLi;
GO

CREATE TABLE Grupos (
    GrupoID INT IDENTITY PRIMARY KEY,
    NombreGrupo VARCHAR(20) NOT NULL
);

CREATE TABLE Alumnos (
    AlumnoID INT IDENTITY(1,1) PRIMARY KEY,
    nombres varchar(80) not null,
    ap_paterno varchar(30) not null,
    ap_materno varchar(30),
    matricula varchar(20) not null,
    GrupoID INT NOT NULL,
    FOREIGN KEY (GrupoID) REFERENCES Grupos(GrupoID)
);

CREATE TABLE Salones (
    SalonID INT IDENTITY PRIMARY KEY,
    NombreSalon VARCHAR(20) NOT NULL
);

CREATE TABLE Horarios (
    HorarioID INT IDENTITY PRIMARY KEY,
    GrupoID INT NOT NULL,
    SalonID INT NOT NULL,
    MateriaID INT NOT NULL,
    MaestroID INT NOT NULL,
    DiaSemana VARCHAR(10) NOT NULL,
    HoraInicio TIME NOT NULL,
    HoraFin TIME NOT NULL,
    FOREIGN KEY (GrupoID) REFERENCES Grupos(GrupoID),
    FOREIGN KEY (SalonID) REFERENCES Salones(SalonID),
    FOREIGN KEY (MateriaID) REFERENCES Materias(MateriaID),
    FOREIGN KEY (MaestroID) REFERENCES Maestros(MaestroID)
);

CREATE TABLE Asistencias (
    AsistenciaID INT IDENTITY PRIMARY KEY,
    AlumnoID INT NOT NULL,
    Fecha DATE NOT NULL,
    HoraRegistro TIME NOT NULL,
    Estado VARCHAR(20) NOT NULL,
    FOREIGN KEY (AlumnoID) REFERENCES Alumnos(AlumnoID)
);

CREATE TABLE Maestros (
    MaestroID INT IDENTITY PRIMARY KEY,
    NombreCompleto VARCHAR(150) NOT NULL
);


CREATE TABLE Materias (
    MateriaID INT IDENTITY PRIMARY KEY,
    NombreMateria VARCHAR(100) NOT NULL
);
