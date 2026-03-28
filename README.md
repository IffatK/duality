# duality

LP Duality Calculator

A Visual + Analytical Linear Programming Solver

<p align="center"> <b>Understand. Solve. Visualize.</b><br/> A complete Linear Programming toolkit with Duality, Simplex, Big-M & Graphical insights. </p>

Preview : https://duality-gql2.onrender.com/

<img width="1920" height="1158" alt="image" src="https://github.com/user-attachments/assets/67c6887b-6b2f-4cf0-bd16-fcbfcdcc1358" />



This is not just a solver — it's a learning engine.


Most tools give answers.

This shows you why the answer is correct.


✔ Step-by-step derivations

✔ Full Simplex iterations

✔ Duality verification

✔ Big-M recovery for infeasible cases


🚀 Features

⚙️ Core Capabilities

Solve Max & Min LP problems

Supports 2–5 variables and constraints

Handles ≤, ≥, = constraints

Automatic standard form conversion


Mathematical Depth

Primal Optimal Solution

Dual Optimal Solution

Strong Duality Theorem validation

Complementary Slackness conditions


🔬 Advanced Methods

Big-M Method (infeasibility handling

Full Simplex Tableau (step-by-step)

Sensitivity Analysis (Pandas-based simulation)


📈 Visualization

Feasible region plotting (2 variables)

Optimal point highlighting

Constraint boundary visualization


🛠️ Tech Stack

🔹 Backend

Python (Flask)

NumPy, SciPy, Pandas, Matplotlib

🔹 Frontend

HTML, CSS, JavaScript

Custom dark-themed UI (no frameworks)

📁 Project Structure

.
├── app.py              # Core LP solver + logic

├── index.html          # Interactive UI

├── requirements.txt    # Dependencies

├── runtime.txt         # Python version


⚙️ Setup Instructions

1️⃣ Clone the repo

git clone https://github.com/IffatK/duality.git

cd duality

2️⃣ Install dependencies

pip install -r requirements.txt


Dependencies:


3️⃣ Run locally

python app.py


Open:


http://127.0.0.1:5000/

⚡ How It Works

🧩 Input

Objective function (c)

Constraints (A, b)

Constraint types

⚙️ Processing

Converts to standard LP form

Solves using scipy.optimize.linprog

Applies:

Simplex (manual tableau)

Big-M (if infeasible)

Dual transformation

📤 Output

Optimal values (Primal + Dual)

Step-by-step explanation

Graph (if 2 variables)

Feasibility & duality checks

🎯 Use Cases

📚 Students learning Linear Programming

🧠 Understanding Duality & Simplex deeply

🧪 Experimenting with constraints & sensitivity

💼 Academic projects / viva demonstrations



👩‍💻 Author

Team Leader:

Iffat Khan


Team : Sneha Mahadik and Simran Waghmare 


Give it a ⭐ on GitHub — it actually helps.
