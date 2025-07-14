import sqlite3
import os
from pathlib import Path

def create_student_database():
    """Create and populate the student database with sample data."""
    
    # Database path
    db_path = Path(__file__).parent / "student.db"
    
    # Remove existing database if it exists
    if db_path.exists():
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Connect to SQLite database
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    
    try:
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENT (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME VARCHAR(25) NOT NULL,
            CLASS VARCHAR(25),
            SECTION VARCHAR(1),
            MARKS INT CHECK(MARKS >= 0 AND MARKS <= 100)
        )
        """)
        
        # Extended sample data with more variety
        students = [
            ('Alice Johnson', 'Data Science', 'A', 85),
            ('Bob Smith', 'Data Science', 'B', 90),
            ('Charlie Brown', 'AI Agents', 'A', 75),
            ('David Wilson', 'AI Agents', 'B', 80),
            ('Eve Davis', 'DevOps', 'A', 95),
            ('Frank Miller', 'DevOps', 'B', 70),
            ('Grace Lee', 'Data Science', 'A', 88),
            ('Heidi Zhang', 'Data Science', 'B', 92),
            ('Ivan Petrov', 'AI Agents', 'A', 78),
            ('Judy Chen', 'AI Agents', 'B', 82),
            ('Karl Schmidt', 'DevOps', 'A', 97),
            ('Leo Garcia', 'DevOps', 'B', 65),
            ('Mallory Taylor', 'Data Science', 'A', 89),
            ('Nina Patel', 'Data Science', 'B', 91),
            ('Oscar Rodriguez', 'AI Agents', 'A', 76),
            ('Peggy Anderson', 'AI Agents', 'B', 84),
            ('Quentin Moore', 'DevOps', 'A', 98),
            ('Rupert Clark', 'DevOps', 'B', 68),
            ('Sybil White', 'Data Science', 'A', 87),
            ('Trent Young', 'Data Science', 'B', 93),
            ('Uma Singh', 'AI Agents', 'A', 79),
            ('Victor Thompson', 'AI Agents', 'B', 81),
            ('Walter Brown', 'DevOps', 'A', 96),
            ('Xena Williams', 'DevOps', 'B', 66),
            ('Yara Ali', 'Data Science', 'A', 86),
            ('Zane Cooper', 'Data Science', 'B', 94),
            ('Amanda Foster', 'AI Agents', 'A', 77),
            ('Brian Turner', 'AI Agents', 'B', 83),
            ('Catherine Hill', 'DevOps', 'A', 99),
            ('Daniel Green', 'DevOps', 'B', 69),
            ('Elena Martinez', 'Data Science', 'A', 90),
            ('Felix Kumar', 'Data Science', 'B', 95),
            ('Gina Ross', 'AI Agents', 'A', 80),
            ('Henry Liu', 'AI Agents', 'B', 85),
            ('Iris Wang', 'DevOps', 'A', 98),
            ('Jack Brooks', 'DevOps', 'B', 67),
            ('Kara Mitchell', 'Data Science', 'A', 88),
            ('Louis King', 'Data Science', 'B', 92),
            ('Maya Gupta', 'AI Agents', 'A', 76),
            ('Nathan Parker', 'AI Agents', 'B', 82),
            ('Olivia Reed', 'DevOps', 'A', 97),
            ('Paul Adams', 'DevOps', 'B', 65),
            ('Quinn Morgan', 'Data Science', 'A', 89),
            ('Rachel Evans', 'Data Science', 'B', 91),
            ('Sam Carter', 'AI Agents', 'A', 78),
            ('Tina Stewart', 'AI Agents', 'B', 84),
            ('Ulrich Nelson', 'DevOps', 'A', 98),
            ('Vera Collins', 'DevOps', 'B', 68),
            ('Wade Phillips', 'Data Science', 'A', 87),
            ('Ximena Gray', 'Data Science', 'B', 93)
        ]
        
        # Insert sample data
        cursor.executemany("INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) VALUES (?,?,?,?)", students)
        
        # Create additional table for courses
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS COURSES (
            COURSE_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            COURSE_NAME VARCHAR(50) NOT NULL,
            INSTRUCTOR VARCHAR(50),
            CREDITS INTEGER,
            DEPARTMENT VARCHAR(30)
        )
        """)
        
        # Insert course data
        courses = [
            ('Introduction to Data Science', 'Dr. Sarah Johnson', 3, 'Computer Science'),
            ('Machine Learning Fundamentals', 'Prof. Michael Chen', 4, 'Computer Science'),
            ('AI Agent Development', 'Dr. Emily Rodriguez', 3, 'Computer Science'),
            ('Advanced AI Systems', 'Prof. David Kim', 4, 'Computer Science'),
            ('DevOps Fundamentals', 'Dr. James Wilson', 3, 'Information Technology'),
            ('Cloud Computing', 'Prof. Maria Garcia', 4, 'Information Technology'),
            ('Database Management', 'Dr. Robert Taylor', 3, 'Computer Science'),
            ('Software Engineering', 'Prof. Lisa Anderson', 4, 'Computer Science')
        ]
        
        cursor.executemany("INSERT INTO COURSES (COURSE_NAME, INSTRUCTOR, CREDITS, DEPARTMENT) VALUES (?,?,?,?)", courses)
        
        # Commit changes
        connection.commit()
        
        # Display statistics
        cursor.execute("SELECT COUNT(*) FROM STUDENT")
        student_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM COURSES")
        course_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT CLASS, COUNT(*) FROM STUDENT GROUP BY CLASS")
        class_distribution = cursor.fetchall()
        
        print(f"✅ Database created successfully at: {db_path}")
        print(f"📊 Statistics:")
        print(f"   - Total students: {student_count}")
        print(f"   - Total courses: {course_count}")
        print(f"   - Class distribution:")
        for class_name, count in class_distribution:
            print(f"     * {class_name}: {count} students")
        
        # Show sample data
        print(f"\n📝 Sample data:")
        cursor.execute("SELECT NAME, CLASS, SECTION, MARKS FROM STUDENT LIMIT 5")
        for row in cursor.fetchall():
            print(f"   - {row[0]} | {row[1]} | Section {row[2]} | {row[3]} marks")
            
    except Exception as e:
        print(f"❌ Error creating database: {str(e)}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    create_student_database()