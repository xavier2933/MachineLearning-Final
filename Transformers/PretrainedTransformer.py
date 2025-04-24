from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

"""
pretrained GPT2 too small. Upgraded to llama

this script and demo prompt massaged with AI
"""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

focused_prompt = """
Interviewer: Hello and welcome. Thanks for coming in today. Could you start by telling me a bit about yourself and your background?

Candidate: Thank you for having me. I'm a software engineer with about 5 years of experience, primarily working with Python and cloud technologies. I started my career at TechStart Inc., where I developed backend services for their e-commerce platform. For the past two years, I've been at DataFlow Systems, focusing on machine learning pipelines and data processing workflows.

Interviewer: That's interesting. Could you tell me about a challenging project you worked on recently?

Candidate: Sure. At DataFlow, I led the development of a real-time recommendation system that needed to process millions of user interactions daily. The main challenge was balancing speed and accuracy while keeping infrastructure costs manageable. I implemented a hybrid approach using both batch processing for long-term patterns and stream processing for immediate user actions. This reduced response latency by 70% while actually improving recommendation quality by 15%.

Interviewer: How did you measure that 15% improvement in quality?

Candidate: We used A/B testing with a subset of users and measured click-through rates and conversion rates as our primary metrics. We also had a secondary metric of session duration that showed users were spending more time engaging with the recommended content.

Interviewer: I see. What about team collaboration? How do you typically work with others?

Candidate: I believe in transparent communication and setting clear expectations. At DataFlow, I worked closely with data scientists and product managers. We used a modified Agile approach with two-week sprints but kept a backlog of exploratory tasks that might not fit neatly into the sprint framework. I also instituted weekly knowledge-sharing sessions where team members could present new techniques or interesting challenges they'd solved.

Interviewer: How do you approach learning new technologies?

Candidate: I'm a hands-on learner. When I need to pick up a new technology, I start with the documentation to understand core concepts, then build small proof-of-concept projects to test my understanding. I also follow relevant communities on forums like Stack Overflow and GitHub to learn best practices. For example, when I needed to learn Kubernetes, I set up a small cluster on my local machine and gradually migrated some of my personal projects to it before implementing it at work.

Interviewer: Let's talk about problem-solving. Can you walk me through your approach when you encounter a difficult technical problem?

Candidate: I follow a systematic approach. First, I make sure I understand the problem completely by asking clarifying questions and defining success criteria. Then I break it down into smaller, more manageable parts. I usually start with the simplest possible solution that could work, even if it's not optimal, to establish a baseline. From there, I iterate and optimize, measuring the impact of each change. I'm also a big believer in leveraging existing knowledge, so I research how others have solved similar problems before reinventing the wheel.

Interviewer: What questions do you have about our company or the role?

Candidate: I'd like to know more about the team structure and how engineering collaborates with other departments. Also, could you tell me about the biggest technical challenges the team is facing right now? And I'm curious about your approach to balancing technical debt management with new feature development.

Interviewer: Those are good questions. Our engineering team is organized into cross-functional pods, each focused on a specific product area. Engineers work closely with product managers and designers throughout the development process. Currently, our biggest challenge is scaling our infrastructure to handle growing user demands while maintaining performance. As for technical debt, we dedicate about 20% of each sprint to maintenance and refactoring work, and we have quarterly "fix-it weeks" focused entirely on improving system architecture and paying down technical debt.

Candidate: That sounds like a healthy approach. The pod structure seems like it would help maintain focus while still allowing for specialization.

Interviewer: Yes, that's exactly the balance we're aiming for. Do you have any other questions for me?

Candidate: What does success look like in this role in the first 90 days?

Interviewer: Great question. In the first 30 days, we'd expect you to get familiar with our codebase and systems, start making small contributions, and begin building relationships with team members. By 90 days, you should be independently contributing to features, participating actively in architecture discussions, and helping to mentor more junior team members if appropriate. We have a structured onboarding process and you'll be paired with an experienced team member as your mentor.

Candidate: Thank you, that's helpful. I appreciate the clear expectations.
Interviewer: Well, I think that covers everything I wanted to ask. Thank you for your time today. We'll be in touch within the next week about next steps.
Candidate: Thank you for the opportunity. I've enjoyed learning more about the role and the company, and I'm excited about the possibility of joining the team.
Question: Did the candidate get the job?
"""

inputs = tokenizer(focused_prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_length=len(inputs["input_ids"][0]) + 100,
        temperature=0.9,
        top_p=0.92,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        num_return_sequences=3
    )

for i, sequence in enumerate(output):
    text = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"\nResponse {i+1}:")
    print(text)