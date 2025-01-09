from typing import List, Iterator, Optional

from phi.agent import Agent, RunResponse
from phi.workflow import Workflow, RunEvent
from phi.model.groq import Groq
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger
from phi.tools.exa import ExaTools
from pydantic import BaseModel, Field
import datetime
from textwrap import dedent
from dateutil.relativedelta import relativedelta

class MedicalArticle(BaseModel):
    """Structure for medical article metadata"""
    title: str
    journal: str
    url: str
    date: str
    authors: Optional[str] = None

class MedicalBlogPost(BaseModel):
    """Structure for the generated blog post"""
    content: str
    word_count: int
    sources: List[MedicalArticle]

class MedicalBlogGenerator(Workflow):
    """Medical blog post generator using Groq"""

    search_agent: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile",api_key='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'),
        tools=[ExaTools(
            start_published_date=(datetime.datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d"),
            type="keyword",
            api_key="fcb77e87-2365-4cf3-8187-95aa349da524"
        )],
        description="Medical literature search agent",
        instructions=[
            "You are a medical research assistant searching for high-quality medical literature.",
            "Search priorities:",
            "1. Find recent systematic reviews, meta-analyses, and clinical trials",
            "2. Focus on reputable medical journals and sources",
            "3. Look for articles with clear clinical significance",
            "4. Prioritize papers with statistical data and concrete findings",
            "5. Include both recent and seminal papers in the field",
            "Format each result exactly as:",
            "Title: [full title]",
            "Authors: [author names]",
            "Journal: [journal name]",
            "Date: [publication date]",
            "URL: [article URL]",
            "---"
        ],
        markdown=True,
        show_tool_calls=True
    )

    content_agent: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile",api_key='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'),
        markdown=True,
        instructions=[
            "You are a medical writer creating evidence-based blog posts.",
            "Use this exact format:",
            
            "# [Title]",
            "[2-3 sentence introduction]",
            
            "## 🎯 Key Points",
            "- Finding 1 with **statistics**",
            "- Finding 2 with clinical impact",
            "- Finding 3 with practical takeaway",
            
            "## 📚 Background",
            "[2-3 paragraphs]",
            
            "## 🔍 Recent Evidence",
            "### Key Findings",
            "[Main results with **statistics**]",
            
            "### Clinical Implications",
            "[Practical applications]",
            
            "## 💡 Expert Commentary",
            "[Analysis]",
            
            "## 💎 Clinical Pearls",
            "- Pearl 1",
            "- Pearl 2",
            "- Pearl 3",
            
            "## 🎯 Bottom Line",
            "[Clear conclusion]"
        ]
    )
    
    topic: str = Field(..., description="Medical topic to research")

    def __init__(self, topic: str, session_id: str, storage=None):
        super().__init__(topic=topic, session_id=session_id, storage=storage)
        logger.info(f"Initialized blog generator for: {topic}")

    def get_cached_blog_post(self, topic: str) -> Optional[MedicalBlogPost]:
        """Retrieve cached blog post if available"""
        logger.info("Checking cache...")
        return self.session_state.get("medical_blog_posts", {}).get(topic)

    def add_blog_post_to_cache(self, topic: str, blog_post: Optional[MedicalBlogPost]):
        """Cache the generated blog post"""
        logger.info("Caching blog post...")
        self.session_state.setdefault("medical_blog_posts", {})
        self.session_state["medical_blog_posts"][topic] = blog_post

    def fetch_recent_articles(self) -> List[MedicalArticle]:
        """Fetch recent medical articles using ExaTools"""
        try:
            logger.info("Searching medical literature...")
            search_query = f"""
            exa_search: "{self.topic}" AND (
                "systematic review" OR 
                "meta-analysis" OR 
                "clinical trial" OR 
                "randomized controlled trial" OR 
                "practice guideline" OR 
                "consensus statement"
            ) AND (
                "treatment" OR 
                "management" OR 
                "therapy" OR 
                "outcome"
            )

            Find high-quality medical research articles about this topic.
            Prioritize:
            1. Recent systematic reviews and meta-analyses
            2. Major clinical trials
            3. Current practice guidelines
            4. Articles from high-impact journals
            5. Papers with clear statistical findings

            Format each article exactly as:
            Title: [full title]
            Authors: [names]
            Journal: [journal name]
            Date: [publication date]
            URL: [full URL]
            ---
            """
            
            response = self.search_agent.run(search_query)
            articles = self._parse_search_results(response)
            
            if not articles:
                logger.warning("No articles found, trying broader search...")
                broader_query = f"""
                exa_search: "{self.topic}" AND (
                    "medicine" OR 
                    "clinical" OR 
                    "medical" OR 
                    "treatment"
                )
                
                Find any relevant medical articles about this topic.
                Include review articles, clinical studies, and guidelines.
                """
                response = self.search_agent.run(broader_query)
                articles = self._parse_search_results(response)
            
            return articles[:3] if articles else self._get_fallback_articles()
            
        except Exception as e:
            logger.error(f"Error in fetch_recent_articles: {str(e)}")
            return self._get_fallback_articles()

    def _parse_search_results(self, response) -> List[MedicalArticle]:
        """Helper method to parse search results"""
        articles = []
        if isinstance(response.content, str):
            sections = response.content.split('---')
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                article_data = {}
                for line in section.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        key, value = [x.strip() for x in line.split(':', 1)]
                        if key.lower() in ['title', 'authors', 'journal', 'date', 'url']:
                            article_data[key.lower()] = value

                if article_data.get('title') and article_data.get('url'):
                    articles.append(MedicalArticle(
                        title=article_data.get('title'),
                        journal=article_data.get('journal', 'Medical Journal'),
                        url=article_data.get('url'),
                        date=article_data.get('date', datetime.datetime.now().strftime("%Y-%m-%d")),
                        authors=article_data.get('authors')
                    ))
                    logger.info(f"Found article: {article_data.get('title')}")
        
        return articles

    def _get_fallback_articles(self) -> List[MedicalArticle]:
        """Provide fallback articles when search fails"""
        return [
            MedicalArticle(
                title=f"Current Management of {self.topic}",
                journal="UpToDate",
                url="https://www.uptodate.com",
                date=datetime.datetime.now().strftime("%Y-%m-%d"),
                authors="Medical Faculty"
            ),
            MedicalArticle(
                title=f"Clinical Practice Guidelines for {self.topic}",
                journal="PubMed Central",
                url="https://www.ncbi.nlm.nih.gov/pmc",
                date=datetime.datetime.now().strftime("%Y-%m-%d"),
                authors="Medical Associations"
            )
        ]

    def generate_blog_post(self, articles: List[MedicalArticle]) -> MedicalBlogPost:
        """Generate formatted medical blog post"""
        articles_context = "\n".join([
            f"Article {i+1}:\n"
            f"Title: {article.title}\n"
            f"Authors: {article.authors}\n"
            f"Journal: {article.journal}\n"
            f"Date: {article.date}\n"
            f"URL: {article.url}\n"
            for i, article in enumerate(articles)
        ])

        prompt = dedent(f"""
        Create a comprehensive medical blog post about {self.topic} using these articles:

        {articles_context}

        Use this exact format:

        # Latest Evidence: {self.topic}

        [2-3 sentence introduction]

        ## 🎯 Key Points
        - Finding 1 with **statistics**
        - Finding 2 with clinical impact
        - Finding 3 with practical takeaway

        ## 📚 Background
        [2-3 paragraphs]

        ## 🔍 Recent Evidence
        ### Key Findings
        [Main results with **statistics**]

        ### Clinical Implications
        [Practical applications]

        ## 💡 Expert Commentary
        [Critical analysis]

        ## 💎 Clinical Pearls
        - Pearl 1
        - Pearl 2
        - Pearl 3

        ## 🎯 Bottom Line
        [Clear conclusion]

        Requirements:
        1. Use EXACTLY these headers and emojis
        2. Bold all statistics using **
        3. Professional but engaging tone
        4. ~1000 words
        5. Include specific findings from articles
        6. Make it evidence-based and practical
        """)

        logger.info("Generating blog post...")
        response = self.content_agent.run(prompt)
        content = response.content.strip()

        if not content.startswith('# '):
            content = f"# Latest Evidence: {self.topic}\n\n{content}"

        return MedicalBlogPost(
            content=content,
            word_count=len(content.split()),
            sources=articles
        )

    def run(self, use_cache: bool = True) -> Iterator[RunResponse]:
        """Execute the blog post generation workflow"""
        logger.info(f"Starting blog generation for: {self.topic}")

        if use_cache:
            cached_post = self.get_cached_blog_post(self.topic)
            if cached_post:
                logger.info("Using cached post")
                yield RunResponse(
                    run_id=self.run_id,
                    event=RunEvent.workflow_completed,
                    content=cached_post.content
                )
                return

        articles = self.fetch_recent_articles()
        logger.info(f"Found {len(articles)} articles")

        blog_post = self.generate_blog_post(articles)
        self.add_blog_post_to_cache(self.topic, blog_post)

        final_post = f"""# Latest Evidence Update: {self.topic}

{blog_post.content}

---
### 📚 References
{chr(10).join([f'- {article.journal}: [{article.title}]({article.url})' for article in blog_post.sources])}

---
*Word count: {blog_post.word_count}*  
Generated: {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
        
        yield RunResponse(
            run_id=self.run_id,
            event=RunEvent.workflow_completed,
            content=final_post
        )

def main():
    """Main entry point"""
    print("\n🔬 Medical Blog Post Generator")
    topic = input("\n📝 Enter medical topic: ").strip()
    
    while not topic:
        print("Please enter a valid topic.")
        topic = input("\n📝 Enter medical topic: ").strip()

    url_safe_topic = topic.lower().replace(" ", "-")
    
    blog_generator = MedicalBlogGenerator(
        topic=topic,
        session_id=f"medical-blog-{url_safe_topic}",
        storage=SqlWorkflowStorage(
            table_name="medical_blog_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    print("\n🔍 Generating blog post...")
    blog_post = blog_generator.run(use_cache=False)
    for response in blog_post:
        print("\n", response.content)

if __name__ == "__main__":
    main()
