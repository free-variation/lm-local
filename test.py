from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def chat(message):
    messages = [
        {"role": "system", "content": "You are a philosopher."},
        {"role": "user", "content": message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
        
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=4000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

test_prompt = """
Please summarize each of the segments of the following English Language Arts lesson, provided in Markdown format.

    Return your summaries in a markdown table, with two columns:
     - lesson segment number
     - summary of lesson segment

     Do not add any extra commentary, introductory or conclusion language, and don't evaluate or discuss the lesson.
    Return only the table of summaries, without an introduction like "Here is the table ..."

     Here is the lesson:

    # Lesson 7: Planning an Adventure Story
##
## Primary Focus of Lesson
- Students will demonstrate understanding of descriptive language and literary devices in the text.
- Students will plan their own adventure story.
## Formative Assessment
- Brainstorming Students complete a chart to frame the outline of their adventure stories.
- Shape of a Story Students complete a chart to plot the action of their adventure stories.
## Lesson Segment 7.1: Review the Chapter (10 min.)
- Have students recall the significant events that happen during Chapter 4, “What I Heard in the Apple Barrel.”Answers may vary, but should include: As the Hispaniola nears the island, Jim overhears a conversation between Silver and some of the other men aboard that proves they cannot be trusted; Silver plans to lead a mutiny and take the treasure once the honest men find it and bring it on board the ship; Jim meets with the captain, the doctor, and the squire to tell them what he heard; they decide to let the dishonest men go ashore upon reaching the island in hopes that this will give Silver an opportunity to convince the men not to mutiny yet; Jim decides to sneak ashore in one of the boats.
- Tell students they will reread and discuss excerpts from Chapter 4, “What I Heard in the Apple Barrel.”
- Have students turn to the table of contents, locate the chapter, and then turn to the first page of the chapter.
- Answers may vary, but should include: As the Hispaniola nears the island, Jim overhears a conversation between Silver and some of the other men aboard that proves they cannot be trusted; Silver plans to lead a mutiny and take the treasure once the honest men find it and bring it on board the ship; Jim meets with the captain, the doctor, and the squire to tell them what he heard; they decide to let the dishonest men go ashore upon reaching the island in hopes that this will give Silver an opportunity to convince the men not to mutiny yet; Jim decides to sneak ashore in one of the boats.
## Lesson Segment 7.2: Close Reading Chapter 4 (25 min.)
- Read the title of the chapter as a class, “What I Heard in the Apple Barrel.” As you read portions of the chapter, pause to explain or clarify the text at each point indicated.
- Have one student read the third paragraph on page 39 aloud.Inferential. Remind students that often, casual language is used to portray characters. When the pirates speak to each other in Treasure Island, some of the dialogue includes shortened forms of words, slang, and incorrect grammar. Which sentences use casual language or slang in this paragraph? How can you rephrase these examples using proper English? Answers may vary, but students may select the sentence, “Skeleton Island, they calls it. It were a main hideout for pirates once.” A possible rephrasing could be, “They call it Skeleton Island. It was a main hideout for pirates once.” Or, “It is called Skeleton Island. It was once a main hideout for pirates.”Inferential. What effect does the use of this casual language have here? Answers may vary, but may include that casual language shows the difference between Long John Silver and Captain Smollett in terms of how they speak. It may suggest that the captain is well-educated and a proper gentleman, while Silver is less educated and is rougher around the edges.
- Have one student read the second paragraph on page 40 aloud.Inferential. Why is Jim surprised by the coolness with which Silver declares his knowledge of the island? Jim knows Silver is lying now. He can’t believe Silver could be so calm while lying to the captain.
- Answers may vary, but students may select the sentence, “Skeleton Island, they calls it. It were a main hideout for pirates once.” A possible rephrasing could be, “They call it Skeleton Island. It was a main hideout for pirates once.” Or, “It is called Skeleton Island. It was once a main hideout for pirates.”
- Answers may vary, but may include that casual language shows the difference between Long John Silver and Captain Smollett in terms of how they speak. It may suggest that the captain is well-educated and a proper gentleman, while Silver is less educated and is rougher around the edges.
- Jim knows Silver is lying now. He can’t believe Silver could be so calm while lying to the captain.
- Jim means now that he knows Silver is dangerous and not trustworthy, he is afraid of Silver.
- It means he could barely hide that he shakes with fear when Silver says his name.
- Jim knows Silver is a dishonest man and that he is planning to mutiny. Jim is afraid of what Silver might do to him because Jim doesn’t trust Silver.
- Read aloud the paragraph that begins, “Silver helped the captain . . . .” Inferential. Why does Silver help the captain anchor the Hispaniola?He is still acting helpful in hopes that the captain won’t notice he is planning a mutiny. Inferential. What effect does use of the idiom like the palm of his hand have, as opposed to the use of He knew the passage well?The figurative language is more descriptive and helps convey how well Silver knows the passage; he knows it very well, just like he knows his own hand very well.
- Have students read the rest of page 41 silently.Inferential. We are in a real pickle is an idiom that means, “We are in a difficult situation.” How does the vocabulary word predicament relate to the meaning of this idiom? Predicament means “in a dangerous or difficult situation,” which is very similar to the idiom we are in a real pickle.Inferential. In the final paragraph, the simile, “He’ll bring ’em on board again, mild as lambs,” is used. How does this simile help the reader understand the expected behavior of the men after Silver talks to them?Lambs are very mild animals; if Silver can get the men to be mild as lambs, it means he will calm them down and they will no longer be ready to rebel.
- Have students read page 42 silently.Inferential. What do you think the phrase taken into our confidence means? It means Jim, the squire, captain, and doctor trust the other honest crew members enough to tell them what is going on and trust that they will still be loyal after they learn about what is being planned.Inferential. What does the narrator mean when he says, “the men must have thought they would trip over treasure as soon as they landed”?The narrator means that the men think the treasure will be easy to find and plentiful.
- He is still acting helpful in hopes that the captain won’t notice he is planning a mutiny.
- The figurative language is more descriptive and helps convey how well Silver knows the passage; he knows it very well, just like he knows his own hand very well.
- Predicament means “in a dangerous or difficult situation,” which is very similar to the idiom we are in a real pickle.
- Lambs are very mild animals; if Silver can get the men to be mild as lambs, it means he will calm them down and they will no longer be ready to rebel.
- It means Jim, the squire, captain, and doctor trust the other honest crew members enough to tell them what is going on and trust that they will still be loyal after they learn about what is being planned.
- The narrator means that the men think the treasure will be easy to find and plentiful.
## Lesson Segment 7.3: Chapter Discussion (5 min.)
- Use the following question to discuss the chapter:Inferential. Think-Pair-Share. Why does Jim think it would be more useful for him to go ashore than to stay aboard the ship?He thinks it would be more useful to explore the island and perhaps keep an eye on what Silver and his men are doing. Also, because Silver does not leave very many men behind, Jim is confident that the men left behind on the ship will not try to do anything to the ship or the others on it.
- Inferential. Think-Pair-Share. Why does Jim think it would be more useful for him to go ashore than to stay aboard the ship?He thinks it would be more useful to explore the island and perhaps keep an eye on what Silver and his men are doing. Also, because Silver does not leave very many men behind, Jim is confident that the men left behind on the ship will not try to do anything to the ship or the others on it.
- He thinks it would be more useful to explore the island and perhaps keep an eye on what Silver and his men are doing. Also, because Silver does not leave very many men behind, Jim is confident that the men left behind on the ship will not try to do anything to the ship or the others on it.
## Lesson Segment 7.4: Word Work: Duplicity (5 min.)
- In the chapter, you read, “I had, by this time, such a fear of his cruelty and duplicity that I could scarcely conceal a shudder when he called out, ‘Ahoy there, Jim!’ and laid his hand on my shoulder.”
- Say the word duplicity with me.
- Duplicity means “dishonest behavior meant to trick someone.”
- The spy used his duplicity to learn secrets from the enemies.
- What are some other examples of duplicity? Be sure to use the word duplicity in your response. Ask two or three students to use the target word in a sentence. If necessary, guide and/or rephrase students’ responses to make complete sentences: “An example of duplicity is  .”
- What part of speech is the word duplicity? noun
- Be sure to use the word duplicity in your response. Ask two or three students to use the target word in a sentence. If necessary, guide and/or rephrase students’ responses to make complete sentences: “An example of duplicity is  .”
- noun
- Tell the students: “I will read several sentences. If the sentence I read is an example of duplicity, say, ‘That is duplicity.’ If the sentence I read is not an example of duplicity, say, ‘That is not duplicity.’”
- The spy sneaked behind enemy lines to get information that helped the army win the war.That is duplicity.
- The children told their parents the truth about what happened when the window broke. That is not duplicity.
- Long John Silver lied to the captain about knowing where the island was located. That is duplicity.
- My friend always waits for me before walking to the bus. That is not duplicity.
- I know I can always count on my brother to help with my homework.That is not duplicity.
- That is duplicity.
- That is not duplicity.
- That is duplicity.
- That is not duplicity.
- That is not duplicity.
### Lesson 7: Planning an Adventure Story
### Writing
## Lesson Segment 7.5: Introduce Shape of a Story (15 min.)
- Tell students that today they will begin planning their adventure story.
- Explain that all stories have a shape or structure. Explain that you will use Treasure Island to model how a story is organized.
- Direct students’ attention to the Shape of a Story Chart you prepared in advance. Ask students what the chart resembles.a mountain
- Tell students the shape illustrates how suspense increases and decreases in a story. Explain that suspense is a feeling of excitement or nervousness caused by wondering what will happen. Explain that a story starts off flat, with minimal suspense, and gradually increases in suspense until the end, when the problem in the story is resolved.
- Point to the “Introduction” line. Explain that the “Introduction” line is flat because it does not increase suspense. An introduction establishes the setting, introduces the main characters, and captures the reader’s attention.
- Explain that, in most stories, the introduction is part of the beginning of the story.
- a mountain
- Several characters are introduced, including Jim Hawkins (the narrator), Billy Bones, and Black Dog. The setting is established—The Admiral Benbow Inn. The reader’s attention is captured by interesting characters and elements of danger, such as the sea chest and the sudden death of Billy Bones.
- Explain that the second part of a story is the problem or conflict.
- Have students identify the problem or conflict at the beginning of Treasure Island. The pirates and the honest men are searching for the same buried treasure. Students may also say the first problem is that Billy Bones is a hunted man.
- Explain that the third part of a story is called Rising Action. Tell students that Rising Action occurs as the story becomes more exciting or the problem worsens.
- Explain that Treasure Island is a relatively long story, so there are many points of Rising Action. Shorter stories, like the one they will write, will have fewer points of Rising Action.
- Have students brainstorm some of the events in Treasure Island that might be Rising Action. Because students have read only through Chapter 4 at this point, be sure not to give away any plot twists. Events so far: Trelawney goes to Bristol to find a ship and crew; Jim gets to know Long John Silver and wonders if he is trustworthy; Captain Smollett doesn’t trust the crew; the Hispaniola sets sail; Jim overhears Long John Silver convincing some members of the crew to mutiny; Jim tells Trelawney and Livesey what he overheard, and they form a plan.
- Explain that because they have not yet read the entire story, students are not yet able to identify parts of the story that align with the rest of the chart.
- Tell students that all stories have a turning point or climax. This occurs when the problem is addressed. It is the most exciting point in the story.
- Point out that students have not read far enough to reach the turning point or climax in Treasure Island, but they should look for the story’s climax in future lessons.
- Explain that the resolution or end is the last part of a story and comes after the turning point or climax; in the end, the problem is resolved and the action calms.
- The pirates and the honest men are searching for the same buried treasure. Students may also say the first problem is that Billy Bones is a hunted man.
- Events so far: Trelawney goes to Bristol to find a ship and crew; Jim gets to know Long John Silver and wonders if he is trustworthy; Captain Smollett doesn’t trust the crew; the Hispaniola sets sail; Jim overhears Long John Silver convincing some members of the crew to mutiny; Jim tells Trelawney and Livesey what he overheard, and they form a plan.
## Lesson Segment 7.6: Plan an Adventure Story (25 min.)
- Remind students that they created a character and selected a setting in previous lessons and explain that now they will focus on the action or main events in the story.
- Have students turn to Activity Page 7.2 and begin creating the shape of their adventure story.
- Have students refer to the displayed Writing Prompt and Shape of a Story Chart as needed.
- I like how the Rising Action moments you have chosen are logically sequenced.
- It looks like you have identified a Rising Action moment for your climax. How could you place the moment you have identified at a turning point in the story?
## Lesson Segment 7.7: Lesson Wrap-Up (5 min.)
- Ask for student volunteers to share their examples of Rising Action.
- Have students keep Activity Pages 7.1 and 7.2 for use in future lessons.
"""


