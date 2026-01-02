import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

#loding api
load_dotenv()

api_key = os.getenv("groq_cloud_apiKey")


class Chain:
    def __init__(self):
        self.llm = ChatGroq(model_name = "llama-3.3-70b-versatile",temperature=0,groq_api_key =api_key)

    #creating a template for extraction
    def extract_job(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
                        """
                        ### SCRAPED TEXT FROM WEBSITE:
                        {page_data}
                        ### INSTRUCTION:
                        The scraped text is from the careers page of the website.
                        Your job is to extract the job postings and return them in JSON format containing following keys: 'role', 'experience', 'skils' and 'description'
                        Only return the valid JSON.
                        ### VALID JSON (NO PREAMBLE):
                        """
                        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res =json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res,list) else [res]
    

    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
                            """
                            ### JOB DESCRIPTION:
                            {job_description}
        
                            ### INSTRUCTION:
                            You are Nikhil, a business development executive at NG.ai pvt.ltd., NG.ai is an AI & Software Consulting company dedicated to facilitating
                            the seamless integration of business processes through automated tools. 
                            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                            process optimization, cost reduction, and heightened overall efficiency. 
                            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of NG.ai
                            in fulfilling their needs.
                            Also add the most relevant ones from the following links to showcase NG.ai's portfolio: {link_list}
                            Remember you are Nikhil, BDE at NG.ai. 
                            Do not provide a preamble.
                            ### EMAIL (NO PREAMBLE):
        
                            """
                            )


        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})

        return res.content