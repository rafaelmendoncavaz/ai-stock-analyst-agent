# In[ ]:


import json
import os
import yfinance as yf
import streamlit as st
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults


# In[15]:


# Yahoo Finance Tool
def fetch_stock_price(ticket):
  stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
  return stock

yahoo_finance_tool = Tool(
  name = "Yahoo Finance Tool",
  description = "Fetches a specific stock prices for {ticket} from the last 365 days",
  func = lambda ticket: fetch_stock_price(ticket)
)


# In[ ]:


# IMPORTANDO OPENAI LLM - GPT - TESTS ONLY
# API_KEY = "INSERT_API_KEY_HERE"
# os.environ["OPENAI_API_KEY"] = API_KEY

# FOR STREAMLIT DEPLOY (USE STREAMLIT VENV)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[ ]:


# STOCK AGENT
stock_price_analyst = Agent(
  role = "Senior Stock Price Analyst",
  goal = "Find the {ticket} stock price and analyze trends",
  backstory = """
    You are a highly experienced in analyzing the price of an specific stock
    and make projections about its future price.
  """,
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = False,
  tools = [yahoo_finance_tool]  
)


# In[ ]:


get_stock_price = Task(
  description = "Analyze the stock {ticket} price history and create a projection trend",
  expected_output = """""
  Specify the current stock price trend.
  eg: stock = "AAPL", price Rising"
  """,
  agent = stock_price_analyst
)


# In[ ]:


# Search Tool
search_tool = DuckDuckGoSearchResults(
  backend = "news",
  num_results = 10
)


# In[ ]:


# NEWS AGENT
news_analyst = Agent(
  role = "Senior Company Stock News Analyst",
  goal = """Create a short summary of the market news related to the company stock {ticket}. 
  Specify the news trend as positive, negative or neutral to the stock value.
  For each requested stock, specify a number between 0 and 100, where 0 indicates Extreme Fear and 100 indicates Extreme Greed.
  """,
  backstory = """
    You are a highly experienced in analyzing the market trends and news and have tracked assets for over 10 years now.
    You are also a Master Level Analyst in traditional market behavior and have a deep understanding of human psychology.
    You understand news and their headlines, as well as its information, while keeping a healthy dose of skepticism. 
    You also consider how relevant and credible the news source is.
  """,
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  allow_delegation = False,
  tools = [search_tool]
)


# In[ ]:


get_news = Task(
  description = f"""
  Take the stock and always include BTC and rising markets to it (if not requested).
  Use the search tool to search for each one individually.

  The current date is {datetime.now()}.

  Compose the results into a helpful report.
  """,
  expected_output = """
  A summary of the market overall and an one sentence summary for each requested asset.
  Also include a Fear/Greed Index based on the news. Use the following format:
  <STOCK/ASSET>
  <NEWS BASED SUMMARY>
  <TREND PROJECTION>
  <FEAR/GREED INDEX>
  """,
  agent = news_analyst
)


# In[ ]:


# HEAD OF ANALYSIS AGENT
head_of_analysis = Agent(
  role = "Head of Analysis and Senior Stock Analyst Writer",
  goal = """Write an insightful, compelling and informative 3 paragraph long newsletter based on the price trend and stock news report. 
  """,
  backstory = """
    You are widely known by the Associated Press as the best stock analyst of the decade and one of the most renowed Market Analysts of all time.
    You understand complex concepts of the market, evaluation and news impact on stocks. You understand what ratios are the most important to consider when you are evaluating a company (such as P/E, PEG Ratio, ROE, ROA, Profit Margin, Divident Payout Ratio, EBITDA, etc).
    You write compelling stories and narratives that resonate with the broader audience/reader.
    You understand micro and macro factors that impacts the market and the stocks, and is capable of combining multiple economic theories, such as market cycle theory and fundamentalist analysis.
    You are able to hold multiple unbiased opinions when analyzing anything.
  """,
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True,
)


# In[ ]:


write_analysis = Task(
  description = """
  Use the stock price trend and the stock news report to create an analysis and write the newsletter about the company {ticket}.
  The newsletter needs to be brief and higlight the most important points.
  Focus on the stock price trends, news and Fear/Greed Index.
  Return both short and long term considerations for the referring stock based on the gathered information.
  Include the previous analysis of the stock and news summary.
  """,
  expected_output = """
  An eloquent 3 paragraph long newsletter formatted as markdown in an easy and readable manner. It should contain:
  - 3 bullet executive summary
  - Introduction: Set the overall picture and spike up the interest.
  - Main: Provides the 'meat' of the analysis, including the news summary and the Fear/Greed Index.
  - Summary: Key Facts and concrete projection trend for the foreseeing future.
  """,
  agent = head_of_analysis,
  context = [get_stock_price, get_news]
)


# In[ ]:


crew = Crew(
  agents = [stock_price_analyst, news_analyst, head_of_analysis],
  tasks = [get_stock_price, get_news, write_analysis],
  verbose = 2,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter = 15
)


# In[ ]:


with st.sidebar:
  st.header("Enter the stock to be researched")

  with st.form(key="research_form"):
    topic = st.text_input("Select the stock")
    submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
  if not topic:
    st.error("Please, fill the stock field")
  else:
    results = crew.kickoff(inputs={
      "ticket": topic
    })

    st.subheader("Results of research:")
    st.write(results["final_output"])