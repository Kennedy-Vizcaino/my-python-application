# my-python-application

This Streamlit web application is a data visualization tool designed to explore trends in used car sales. It simulates data exploration through interactive and static visualizations generated from a real-world car sales dataset (vehicles_us.csv). The tool provides insights into price distribution, vehicle mileage, model year, and car type using a combination of Seaborn, Matplotlib, Plotly Express, and Pandas.

To run this project on your own computer, start by cloning the repository to your local machine. You can do this by opening a terminal and running:

git clone https://github.com/kennedy-vizcaino/my-python-application.git

cd my-python-application

Next, install the required packages. If there's a requirements.txt file included, simply run:

pip install -r requirements.txt

If that file isn't available, you can install the needed libraries manually using:

pip install streamlit pandas seaborn matplotlib plotly distinctipy 

Once everything is set up, make sure the dataset file—vehicles_us.csv—is placed in the root directory of the project.

Now you’re ready to launch the app! Just run the following command:
streamlit run app.py

Your default browser should open automatically, but if it doesn’t, you can go to http://localhost:8501 to view the dashboard.

