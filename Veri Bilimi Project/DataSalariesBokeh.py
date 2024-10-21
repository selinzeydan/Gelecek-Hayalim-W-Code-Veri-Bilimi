import pandas as pd
import numpy as np
from bokeh.io import show
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import row
from bokeh.transform import cumsum
from bokeh.models import ColumnDataSource
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from bokeh.models import Slope
from sklearn.linear_model import LinearRegression
from bokeh.models import NumeralTickFormatter
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral3

curdoc().theme = "light_minimal"

#colors
colors = {'male': '#718dbf', 'female': '#e84d60'}

df = pd.read_csv('Salary Prediction of Data Professions.csv')

gender_distribution = df['SEX'].value_counts().reset_index()
gender_distribution.columns = ['SEX', 'count']
gender_distribution['angle'] = gender_distribution['count'] / gender_distribution['count'].sum() * 2 * np.pi
gender_distribution['color'] = [colors['male'], colors['female']]

p = figure(title="General Gender Distribution", toolbar_location=None, tools="hover", tooltips="@SEX: @count", x_range=(-0.5, 1.0))
p.wedge(x=0, y=1, radius=0.4,
         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
         line_color="white", fill_color='color', legend_field='SEX', source=gender_distribution)

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None
p.title.text_font_size = '16pt'

X = df[['AGE', 'RATINGS', 'SEX']]
y = df['SALARY']

label_encoder = LabelEncoder()
X['SEX'] = label_encoder.fit_transform(X['SEX'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

source = ColumnDataSource(data=dict(
    age=df['AGE'].iloc[y_test.index],
    real_salary=y_test,
    predicted_salary=y_pred
))

#---scatterplot grafik
scatter = figure(title="Age vs Salary Estimates (Random Forest Regression)", width=600, height=450)
scatter.circle('age', 'real_salary', source=source, color="blue", size=10, alpha=0.6, legend_label="Real Values")
scatter.circle('age', 'predicted_salary', source=source, color="orange", size=10, alpha=0.6, legend_label="Predicted Values")
scatter.xaxis.axis_label = "Age"
scatter.yaxis.axis_label = "Average Salary"
scatter.yaxis.formatter = NumeralTickFormatter(format="0,0")  # Format y-axis to avoid scientific notation
scatter.legend.location = "top_left"

ideal_line = Slope(gradient=1, y_intercept=0, line_color="blue", line_dash='dashed', line_width=2)
scatter.add_layout(ideal_line)

#--scatter_exp

gender_list = df['SEX'].unique().tolist()  #for diff genders, diff colors

source_exp = ColumnDataSource(data=dict(
    past_experience=df['PAST EXP'],
    salary=df['SALARY'],
    sex=df['SEX']
))

X_experience = df[['PAST EXP']].values  
y_salary = df['SALARY'].values           

linear_model = LinearRegression()
linear_model.fit(X_experience, y_salary)

x_range = np.linspace(X_experience.min(), X_experience.max(), 100).reshape(-1, 1)
y_pred_exp = linear_model.predict(x_range)

scatter_exp = figure(title="Salary by Past Experience and Gender", width=600, height=400)


scatter_exp.scatter('past_experience', 'salary', source=source_exp, 
                    size=10, alpha=0.7, legend_field='sex',
                    color=factor_cmap('sex', palette=Spectral3, factors=gender_list))  

scatter_exp.line(x_range.flatten(), y_pred_exp, line_color='brown', 
                 line_width=2, legend_label="Trendline")

scatter_exp.xaxis.axis_label = "Past Experience (Years)"
scatter_exp.yaxis.axis_label = "Average Salary"
scatter_exp.yaxis.formatter = NumeralTickFormatter(format="0,0")  
scatter_exp.legend.title = "Gender"

show(row(children=[p, scatter_exp, scatter], sizing_mode="scale_width"))