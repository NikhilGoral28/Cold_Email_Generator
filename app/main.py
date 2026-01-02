import streamlit as st
from utils import clean_text
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio


def create_streamlit_app(llm,portfolio,clean_text):
    st.title("Cold email Generator")


    url_input = st.text_input("Enter a URL: ", value="https://jobs.adidas-group.com/job/Holon-Merchandising-Specialist-HaMe/1279844101/?feedId=301201&utm_source=j2w")
    submit_button = st.button("submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_job(data)

            for job in jobs:
                skills = job.get('skills',[])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An error Occurred: {e}")


if __name__=="__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout='wide',page_title="Cold Email Generator")
    create_streamlit_app(chain, portfolio,clean_text)
