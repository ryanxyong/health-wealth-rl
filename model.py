# By Abigail Kayser, Christopher Scanlon, Goutham Veeramachaneni, 
#    Kate Spillane, Madison Sclar, Ryan Yong
# To generate the model for our final project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define assumptions
hospital_count = 50
insurance_company_count = 20
patient_count = 10000
MAX_ROUNDS = 50

# Data generators

# Source for capacity: https://www.ahd.com/states/hospital_CA.html
# Source for cost_per_service: https://www.wsj.com/articles/how-much-does-a-c-section-cost-at-one-hospital-anywhere-from-6-241-to-60-584-11613051137
def generate_hospitals(n):
    return pd.DataFrame({
        'capacity': np.random.randint(200, 500, n),  # Adjusted capacity range
        'quality': np.random.uniform(5, 9, n),  # Adjusted quality range
        'cost_per_service': np.random.uniform(6241, 60584, n),
        'c_section_availability': np.random.choice([0, 1], n, p=[0.3, 0.7])
    })

# Source for client_count: https://www.chcf.org/wp-content/uploads/2019/05/CAHealthInsurersAlmanac2019.pdf
# Source for average_premium: https://www.valuepenguin.com/best-cheap-health-insurance-california#:~:text=What%20is%20the%20average%20cost,and%20the%20plan's%20metal%20tier.
def generate_insurance_companies(n):
    return pd.DataFrame({
        'client_count': np.random.randint(900000, 8000000, n),  # Adjusted client count range
        'average_premium': np.random.uniform(350, 7000, n),  # Adjusted average premium range
        'c_section_coverage': np.random.choice([0, 1], n, p=[0.5, 0.5]) # 50% chance that insurance company covers C-section
    })

def generate_patients(n):
    return pd.DataFrame({
        'insurance_company': np.random.choice(range(insurance_company_count), n),
        'need_for_c_section_services': np.random.choice([0, 1], n, p=[0.6, 0.4]),  # Increased patients in need of C-section
        'out_of_pocket_cost': np.full(n, np.nan),
        'wait_time': np.full(n, np.nan),
        'welfare': np.full(n, np.nan),
        'preferred_hospital': np.random.choice(range(hospital_count), n),
        'decision': np.full(n, 'None', dtype=object),
        'round_count': np.full(n, 0)
    })

# Generate data
hospitals = generate_hospitals(hospital_count)
insurance_companies = generate_insurance_companies(insurance_company_count)
patients = generate_patients(patient_count)

# Adjust the decision-making condition
def make_decision(offer, counteroffer, hospital, insurance_company, round_count):
    decision = ''
    if offer * 0.9 <= counteroffer and hospital['quality'] >= 4.5 and insurance_company['client_count'] > 0:  # Adjusted decision-making conditions
        decision = 'accept'
    else:
        decision = 'reject'
        round_count += 1
    return decision, round_count

# Include more factors in the offers
def make_offer(counteroffer):
    return counteroffer * ((np.random.random() / 5) + .9)

def make_counteroffer(offer):
    return offer * ((np.random.random() / 5) + .75) # counteroffer is within 5% to 25% of original offer

# Define the transaction process
def transaction(offer, hospital, insurance_company, patient):
    hospital['capacity'] -= 1
    insurance_company['client_count'] -= 1
    patient['out_of_pocket_cost'] = max(0, offer - insurance_company['average_premium'])
    patient['wait_time'] = 1 / hospital['capacity'] if hospital['capacity'] > 0 else float('inf')
    return hospital, insurance_company, patient

# Define Nash equilibrium
# With help from ChatGPT
def nash_equilibrium(hospital, insurance_company, patient):
    curr_counteroffer = 0

    for round_count in range(1, MAX_ROUNDS + 1):
        if round_count == 1:
            curr_counteroffer = hospital['quality'] * 0.6 + hospital['capacity'] * 0.4
        offer = make_offer(curr_counteroffer)
        counteroffer = make_counteroffer(offer)
        curr_counteroffer = counteroffer
        decision, round_count = make_decision(offer, counteroffer, hospital, insurance_company, round_count)
        if decision == 'accept':
            hospital, insurance_company, patient = transaction(offer, hospital, insurance_company, patient)
            patient['welfare'] = hospital['quality'] - patient['out_of_pocket_cost'] - patient['wait_time']
            return hospital, insurance_company, patient, decision, round_count
    patient['wait_time'] = float('inf')
    patient['welfare'] = 0
    return hospital, insurance_company, patient, 'no agreement', round_count

# Simulate the game
for i in range(patient_count):
    preferred_hospital_index = patients.loc[i, 'preferred_hospital']
    insurance_company_index = patients.loc[i, 'insurance_company']
    patient = patients.loc[i]
    if hospitals.loc[preferred_hospital_index, 'c_section_availability'] == 1 and patient['need_for_c_section_services'] == 1:
        if hospitals.loc[preferred_hospital_index, 'capacity'] > 0:
            hospitals.loc[preferred_hospital_index], insurance_companies.loc[insurance_company_index], patients.loc[i], patients.loc[i, 'decision'], patients.loc[i, 'round_count'] = nash_equilibrium(hospitals.loc[preferred_hospital_index], insurance_companies.loc[insurance_company_index], patients.loc[i])
        else:
            patients.loc[i, 'wait_time'] = float('inf')
            patients.loc[i, 'welfare'] = 0
            patients.loc[i, 'decision'] = 'no capacity'
            patients.loc[i, 'round_count'] = 0

# Culls irrelevant patients from data to be displayed
patients = patients[patients.decision != 'None']

######################## Analysis ########################
# With help from ChatGPT for refining representations
# Count of decisions
print("Count of decisions:")
print(patients['decision'].value_counts())

# Average wait time
print("Average wait time:", patients['wait_time'].mean())

# Histogram of welfare
plt.figure(figsize=(10,6))
plt.hist(patients['welfare'].dropna(), bins=20)
plt.xlabel('Welfare')
plt.ylabel('Number of Patients')
plt.title('Histogram of Welfare')
plt.show()

# Average welfare by insurance company
print("Average welfare by insurance company:")
print(patients.groupby('insurance_company')['welfare'].mean())

# Average wait time by insurance company
print("Average wait time by insurance company:")
print(patients.groupby('insurance_company')['wait_time'].mean())

# Patients' welfare and wait time based on whether their insurance covers C-section services or not
insurance_companies['c_section_coverage'] = insurance_companies['c_section_coverage'].map({0:'No', 1:'Yes'})
merged_data = patients.merge(insurance_companies['c_section_coverage'], left_on='insurance_company', right_index=True)

print("Average welfare by C-section coverage:")
print(merged_data.groupby('c_section_coverage')['welfare'].mean())

print("Average wait time by C-section coverage:")
print(merged_data.groupby('c_section_coverage')['wait_time'].mean())

# Average patient welfare and wait time based on hospital quality
merged_data2 = patients.merge(hospitals[['quality', 'capacity']], left_on='preferred_hospital', right_index=True)

print("Average welfare by hospital quality:")
print(merged_data2.groupby('quality')['welfare'].mean())

print("Average wait time by hospital quality:")
print(merged_data2.groupby('quality')['wait_time'].mean())

print("Average welfare by hospital capacity:")
print(merged_data2.groupby('capacity')['welfare'].mean())

print("Average wait time by hospital capacity:")
print(merged_data2.groupby('capacity')['wait_time'].mean())

# Average round count by decision
print("Average round count by decision:")
print(patients.groupby('decision')['round_count'].mean())

# Barplot showing the average welfare for each insurance company
plt.figure(figsize=(10,6))
sns.barplot(x='insurance_company', y='welfare', data=patients)
plt.title('Average Welfare by Insurance Company')
plt.show()

# Boxplot showing the distribution of wait times for each insurance company
plt.figure(figsize=(10,6))
sns.boxplot(x='insurance_company', y='wait_time', data=patients)
plt.title('Distribution of Wait Times by Insurance Company')
plt.show()

# Histogram of the number of rounds taken for each decision
plt.figure(figsize=(10,6))
print(patients['round_count'])
sns.histplot(patients['round_count'])
plt.title('Distribution of Round Count')
plt.show()
