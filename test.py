import markdown
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ["GEMIN_API_KEY"])

prompt = PromptTemplate.from_template(""" 
content: 
    {content}
 html code: 
<body>
  <h1>CANDIDATE OVERVIEW</h1>
  <div class="section">
    <div class="section-title">Recommendation:</div>
    <ul>
      <li>Highly motivated Software Engineer with a passion for building innovative solutions. Proven track record of exceeding expectations and delivering high-quality code on time and within budget.</li>
    </ul>
  </div>
  <div class="section">
    <div class="section-title">Key Summary:</div>
    <ul>
      <li>Expertise in back-end development using Java and Spring Boot.</li>
      <li>Extensive experience designing and implementing scalable and secure APIs.</li>
      <li>Strong understanding of cloud platforms (AWS, Azure) and DevOps principles.</li>
      <li>Excellent communication and collaboration skills with a focus on agile methodologies.</li>
    </ul>
  </div>
  <div class="section">
    <div class="section-title">Profile:</div>
    <ul>
      <li>Software Engineer with 5+ years of experience in the FinTech industry.</li>
      <li>Successfully developed and deployed critical trading platform features, resulting in a 20% increase in transaction efficiency.</li>
      <li>Led a cross-functional team in building a microservices architecture for a new e-commerce platform.</li>
      <li>Proficient in unit testing, code reviews, and continuous integration/continuous delivery (CI/CD) pipelines.</li>
    </ul>
  </div>
  <div class="section">
    <div class="section-title">Current Compensation:</div>
    <ul>
      <li>Negotiable; seeking a competitive salary and benefits package commensurate with experience and skills.</li>
    </ul>
  </div>
  <div class="section">
    <div class="section-title">Notice Period:</div>
    <ul>
      <li>Candidate is on 3 months' notice and available for a start date after [desired date].</li>
    </ul>
  </div>
</body>
</html>


Fit the content in the html code . Now generate html code with the content given. Make sure to fit and format the content given in the html code similarly.
only return the html code but in the string format. No backtics, no markdown format.

""")


# Your Markdown content
md_content = """
CANDIDATE OVERVIEW

Recommendation:

A highly motivated and results-oriented Marketing Manager with 8+ years of experience in the B2B SaaS industry. Proven track record of exceeding targets by consistently developing and executing successful marketing campaigns that drive brand awareness, lead generation, and customer acquisition. Known for strong communication, collaboration, and analytical skills, with a passion for staying ahead of industry trends and leveraging data insights to optimize marketing strategies.

Key Summary:

Expertise in developing and executing comprehensive digital marketing campaigns across various channels, including SEO, SEM, content marketing, social media marketing, and email marketing.
Proven ability to analyze marketing performance metrics and translate data into actionable insights to optimize campaigns and maximize ROI.
Strong project management skills with a focus on meeting deadlines and delivering projects within budget.
Excellent written and verbal communication skills with the ability to create compelling marketing copy and deliver impactful presentations.
Experienced in collaborating with cross-functional teams, including sales, product development, and customer success, to ensure alignment and achieve marketing goals.
Skilled in using marketing automation platforms (e.g., HubSpot, Marketo) and CRM systems (e.g., Salesforce) to streamline marketing processes and track campaign performance.
Profile:

Marketing Manager with a successful track record of growing revenue and brand awareness for leading B2B SaaS companies.
Increased website traffic by 40% and qualified leads by 30% through strategic SEO and content marketing initiatives.
Developed and launched a social media marketing campaign that resulted in a 25% increase in brand mentions and a 15% growth in social media followers.
Managed a team of marketing specialists to successfully execute a multi-channel marketing campaign that exceeded the lead generation target by 20%.
Presented marketing strategies and campaign performance reports to senior management, effectively communicating key insights and recommendations.
Current Compensation:

Competitive salary and benefits package commensurate with experience and skills. Open to negotiation.
Notice Period:

Candidate is currently on 3 months' notice and available for a start date after [desired date].
This example adds:

More details and accomplishments to the "Recommendation" and "Profile" sections.
Specific examples of achievements with quantifiable results.
Skills related to project management and data analysis.
Information about desired compensation and openness to negotiation.
You can further customize this by:

Highlighting relevant skills and experience based on the specific job you're applying for.
Quantifying achievements using metrics and percentages.
Adding specific examples of marketing campaigns and their impact.
Adjusting the notice period based on your situation.
"""


chain = ({"content": RunnablePassthrough()} | prompt | model)

res = chain.invoke(md_content)

print(res)


# Save the HTML content to a file
with open('candidate_overview.html', 'w') as f:
    f.write(
       """
    <!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Candidate Overview</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
    }
    h1 {
      text-align: center;
      font-size: 24px;
    }
    .section {
      margin-bottom: 20px;
    }
    .section-title {
      font-weight: bold;
      margin-bottom: 10px;
      font-size: 18px;
    }
    ul {
      list-style-type: disc;
      margin-left: 20px;
    }
    li {
      margin-bottom: 10px;
    }
  </style>
</head>
""" + res + "</html>"
    )

