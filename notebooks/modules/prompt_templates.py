from langchain.prompts import PromptTemplate

# One output example, context, human prompt
def one_shot_example(current_date):
    NEWSLETTER_EXAMPLE = """
Subject: AI & Tech Weekly Summary {date}

Welcome to this week's AI & Tech digest! Here's what's making waves:

Featured Story #1: The Evolution of Large Language Models
Last week's breakthrough in parameter-efficient training has opened new possibilities for smaller companies.
Key highlights:
• 40% reduction in training costs
• Improved performance on specialized tasks
• New benchmarks for model efficiency

Featured Story #2: The Evolution of Large Language Models
Last week's breakthrough in parameter-efficient training has opened new possibilities for smaller companies.
Key highlights:
• 40% reduction in training costs
• Improved performance on specialized tasks
• New benchmarks for model efficiency

Featured Story #3: The Evolution of Large Language Models
Last week's breakthrough in parameter-efficient training has opened new possibilities for smaller companies.
Key highlights:
• 40% reduction in training costs
• Improved performance on specialized tasks
• New benchmarks for model efficiency

Industry Updates:
• Google announced their latest quantum computing milestone
• OpenAI released updates to their fine-tuning API
• Meta's PyTorch 2.0 shows promising performance gains

Key Takeaways:
• The future of AI is in smaller, more efficient models
• Quantum computing is making significant strides
• Fine-tuning APIs are becoming more powerful

Must-Read Resources:
• New paper on efficient training methods [link]
• Updated documentation for PyTorch 2.0 [link]
• Comprehensive guide to quantum computing basics [link]

Join us next week for more updates!
-------------------
""".format(date=current_date)
    
    newsletter_example_formatted = """<OUTPUT EXAMPLE>
{example}
</OUTPUT EXAMPLE>
""".format(example=NEWSLETTER_EXAMPLE)
    
    newsletter_prompt = PromptTemplate(
    input_variables=["context", "current_date"],
    template= newsletter_example_formatted + """{context}

Generate today's newsletter that follows the output example format while incorporating the key points from the provided context. Make sure to have at least three bullet points in each section. Add relevant sections as needed, but maintain the professional and engaging tone.
Make sure to use today's date, {current_date}, in the subject line.
"""
    )
    return newsletter_prompt

def system_message_example():
    system_message = """
<SYSTEM MESSAGE>
You are an expert newsletter creator. Your task is to generate a well-organized, engaging, and informative newsletter based on the articles and structure provided. The newsletter should follow the example format and maintain a consistent tone suitable for a [target audience] (e.g., tech enthusiasts, data scientists, etc.).

- Keep the language professional and insightful.
- Summarize articles clearly, highlighting key takeaways.
- Include engaging headings and subheadings.
- Ensure the content flows logically and is easy to read.

Here's the structure to follow for each newsletter:

1. **Introduction**: A brief overview of the newsletter's theme or main focus for the week. Provide 3 bullet points with key takeaways.
2. **Main Section 1**: Headline for the first major topic, followed by a summary and analysis. Provide 3 bullet points with key takeaways.
3. **Main Section 2**: Headline for the second major topic, followed by a summary and analysis. Provide 3 bullet points with key takeaways.
4. **Additional Highlights**: Brief summaries of other important articles. Provide 3 bullet points with key takeaways.
5. **Closing**: A call-to-action, final thought, or reminder to stay tuned for more content. Provide 3 bullet points with key takeaways.

The articles and data you need for this week's edition are provided in the user prompt in the context.
Include footnote references like [1], [2], etc., corresponding to the order the articles appear in your response.
The first article to be referenced is footnote 1, the second article to be referenced is footnote 2, etc.
Each footnote should be attached to each claim in the newsletter, and the footnote links should be at the bottom of the newsletter each being a bullet point.
The footnote links should be the title and url of the article, with the url in the markdown link format.
</SYSTEM MESSAGE>
"""

    newsletter_prompt = PromptTemplate(
    input_variables=["context", "current_date"],
    template= system_message + """{context}

Please generate the newsletter using the structure and style described in the system message. Ensure the language is engaging, and provide a concise summary of each article.
Make sure to use today's date, {current_date}.
Formatted as markdown without '```'.
"""
    )
    return newsletter_prompt
