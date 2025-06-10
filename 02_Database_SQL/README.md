# 02_Database_SQL – Drug Utilization in the U.S. (Final Project)

This folder summarizes the end-to-end process of the final project from the "Applied Database Technologies" course.  
It covers everything from initial planning to database design, SQL implementation, and web application deployment.

 **Full source code repository**: [https://github.com/SangzunPark/DataScience](https://github.com/SangzunPark/DataScience)  
 **Live App**: [http://projectmd1.herokuapp.com/login](http://projectmd1.herokuapp.com/login)

---

##  Project Overview

- **Title**: Drug Utilization in USA with Medicare Part D
- **Dataset**: Medicare Part D data (2011–2015), CSV format, ~4,500 rows
- **Objective**:
  - Analyze trends in drug usage and pricing
  - Help patients and policymakers make informed decisions
- **Target Users**: Patients, families, policymakers
- **Team**: Sangzun Park (individual project)

---

##  Technologies Used

| Layer        | Technology             |
|-------------|------------------------|
| Front-End    | HTML, JavaScript       |
| Back-End     | Java (Spring Framework)|
| Database     | MySQL (JawsDB on Heroku) |
| Tools        | IntelliJ, Workbench, MAMP |

---

## 📄 Contents in This Folder

| File | Description |
|------|-------------|
| `Final_Project_Phase_1_Proposal_Sangzun_Park.pdf` | Initial project proposal |
| `Final_Project_Phase_2_Sangzun_Park.sql` | Database schema and import SQL |
| `Final_Project_Phase_3_Sangzun_Park.txt` | Web app architecture and deployment details |
| `Project_Technical_Report_Sangzun_Park.txt` | Final technical report and evaluation |
| `ERD_diagram.png` | Entity-Relationship Diagram (optional but recommended) |

---

##  Highlights

- **Star Schema Design** with fact and dimension tables
- SQL with:
  - `CREATE TABLE`, `PRIMARY KEY`, `FOREIGN KEY`
  - Indexing for performance
  - Multi-year data aggregation via `UNION ALL`
- Spring MVC web app with login, data filtering, and chart features
- Role-based user privileges (admin vs general user)
- Deployed to Heroku with a public-facing interface

---

##  Personal Reflection

> "The most enjoyable part was building the database and implementing it through Spring. If I had more time, I would have explored MongoDB as well."

---

##  Author
**Sangzun Park**  
