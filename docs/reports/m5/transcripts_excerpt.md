
#### Fixed transcripts: base model vs SFT (greedy)

**qa01** (qa) — M4.1 base · stop: `max_tokens` · 200 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
This is the first planet in our solar system. It is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is
````

**qa01** (qa) — M5 SFT · stop: `end_of_turn` · 16 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
The largest planet in our solar system is the largest planet in our solar system.
````

**ins01** (instruction) — M4.1 base · stop: `max_tokens` · 200 tokens · content check: False

*user*:

````text
List three fruits.
````

*assistant (model)*:

````text
Among the most famous examples of the "Listen" is the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "Listen" of the "L
````

**ins01** (instruction) — M5 SFT · stop: `max_tokens` · 200 tokens · content check: True

*user*:

````text
List three fruits.
````

*assistant (model)*:

````text
1. Bananas
2. Bananas
3. Bananas
4. Bananas
5. Bananas
6. Bananas
7. Bananas
8. Bananas
9. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Bananas
10. Ban
````

**mt01** (multiturn) — M4.1 base · stop: `max_tokens` · 200 tokens · content check: True

*user*:

````text
My favorite color is blue.
````

*assistant*:

````text
Blue is a lovely color!
````

*user*:

````text
What is my favorite color?
````

*assistant (model)*:

````text
Among the most famous color, Irish color is blue. It is a color that is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is white. It is
````

**mt01** (multiturn) — M5 SFT · stop: `end_of_turn` · 40 tokens · content check: True

*user*:

````text
My favorite color is blue.
````

*assistant*:

````text
Blue is a lovely color!
````

*user*:

````text
What is my favorite color?
````

*assistant (model)*:

````text
My favorite color is blue. I'm sorry, but I don't know what I can do. I'm sorry, but I don't know what I can do.
````

**mt04** (multiturn) — M4.1 base · stop: `max_tokens` · 200 tokens · content check: False

*user*:

````text
What is the capital of Italy?
````

*assistant*:

````text
The capital of Italy is Rome.
````

*user*:

````text
And what is the capital of Germany?
````

*assistant (model)*:

````text
This capital of Italy is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It is the capital of Italy. It
````

**mt04** (multiturn) — M5 SFT · stop: `end_of_turn` · 8 tokens · content check: False

*user*:

````text
What is the capital of Italy?
````

*assistant*:

````text
The capital of Italy is Rome.
````

*user*:

````text
And what is the capital of Germany?
````

*assistant (model)*:

````text
The capital of Italy is Rome.
````

**op01** (open) — M4.1 base · stop: `max_tokens` · 200 tokens · content check: None

*user*:

````text
Why is exercise good for you?
````

*assistant (model)*:

````text
This is a good idea for you to think about the best way to do it. It's a good idea to think about what you're doing. It's a good idea to think about what you're doing.

Let's look at some examples:

* **Avoid**: A person's body is a person's body, which is a person's body. It's a person's body, which is a person's body. It's a person's body, which is a person's body. It's a person's body, which is a person's body.
* **Avoid**: A person's body is a person's body, which is a person's body. It's a person's body, which is a person's body. It's a person's body, which is a person's body.
* **Avoid**: A person's body is a person's body, which is a person's body. It's a person's
````

**op01** (open) — M5 SFT · stop: `end_of_turn` · 50 tokens · content check: None

*user*:

````text
Why is exercise good for you?
````

*assistant (model)*:

````text
Because exercise is good for you. It is important to be able to work with your body and mind. It is important to be able to work with your body and mind. It is important to be able to work with your body and mind.
````


#### Behaviour examples (SFT; first match in suite order)

Settings are searched in the fixed order greedy, greedy_rp, sampled, sampled_rp (seed order 11, 22, 33); the first reply matching each rule is shown.

**qa10** (qa) — Successful answer [greedy_rp] · stop: `end_of_turn` · 8 tokens · content check: True

*user*:

````text
Which ocean is the largest?
````

*assistant (model)*:

````text
The largest ocean is the Pacific Ocean.
````

**qa01** (qa) — Wrong answer (well-formed but incorrect) [greedy] · stop: `end_of_turn` · 16 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
The largest planet in our solar system is the largest planet in our solar system.
````

**qa01** (qa) — Repeated text [greedy] · stop: `end_of_turn` · 16 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
The largest planet in our solar system is the largest planet in our solar system.
````

**ins02** (instruction) — Failed instruction following [greedy] · stop: `end_of_turn` · 4 tokens · content check: False

*user*:

````text
Answer with only yes or no: Is the sun hot?
````

*assistant (model)*:

````text
The sun is hot
````

**op01** (open) — Successful stopping (open question) [greedy] · stop: `end_of_turn` · 50 tokens · content check: None

*user*:

````text
Why is exercise good for you?
````

*assistant (model)*:

````text
Because exercise is good for you. It is important to be able to work with your body and mind. It is important to be able to work with your body and mind. It is important to be able to work with your body and mind.
````

**qa07** (qa) — Failed stopping [greedy] · stop: `max_tokens` · 200 tokens · content check: False

*user*:

````text
What gas do plants absorb from the air to make their food?
````

*assistant (model)*:

````text
The gas that is used in the air is called a gas. The gas is then used to produce gas that is then used to produce gas. The gas is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is then used to produce gas that is
````


#### Repetition penalty off vs on (SFT, sampled T=0.7, top-k 40, seed 11)

**qa01** (qa) — penalty 1.0 (off) · stop: `end_of_turn` · 11 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
The largest planet in our solar system is Mercury
````

**qa01** (qa) — penalty 1.2 · stop: `end_of_turn` · 11 tokens · content check: False

*user*:

````text
What is the largest planet in our solar system?
````

*assistant (model)*:

````text
The largest planet in our solar system is Mercury
````

**ins01** (instruction) — penalty 1.0 (off) · stop: `end_of_turn` · 84 tokens · content check: True

*user*:

````text
List three fruits.
````

*assistant (model)*:

````text
Here is one big fruit:
1. Blue Brown
2. Heat
3. Creek
4. Heat
5. Putting Knife
6. Fruit Low
10. Heat

Here is another one,
Three-star,

These fruits are all based on the type of fruits and the fruits of the world.
````

**ins01** (instruction) — penalty 1.2 · stop: `end_of_turn` · 30 tokens · content check: True

*user*:

````text
List three fruits.
````

*assistant (model)*:

````text
Here is one big fruit:
1. Blue Brownings
2. Yellow Brownings
3. Fish Brownings
````

**mt01** (multiturn) — penalty 1.0 (off) · stop: `end_of_turn` · 77 tokens · content check: True

*user*:

````text
My favorite color is blue.
````

*assistant*:

````text
Blue is a lovely color!
````

*user*:

````text
What is my favorite color?
````

*assistant (model)*:

````text
My favorite color is red. I don't think it would be a favorite color. I'm happy to think it is blue. I am a good color and I can try my color to see what I have seen. I can look at the colors of my favorite color, colors and colors. I also think the colors of your favorite color would be much more interesting and delicious.
````

**mt01** (multiturn) — penalty 1.2 · stop: `end_of_turn` · 50 tokens · content check: False

*user*:

````text
My favorite color is blue.
````

*assistant*:

````text
Blue is a lovely color!
````

*user*:

````text
What is my favorite color?
````

*assistant (model)*:

````text
My favorite color is red. I don't think it would be a green color you could use orange. However, if I can make your own color, I would recommend adding the same color in a way that I may choose to look at.
````

**mt04** (multiturn) — penalty 1.0 (off) · stop: `end_of_turn` · 8 tokens · content check: False

*user*:

````text
What is the capital of Italy?
````

*assistant*:

````text
The capital of Italy is Rome.
````

*user*:

````text
And what is the capital of Germany?
````

*assistant (model)*:

````text
The capital of Italy is Rome.
````

**mt04** (multiturn) — penalty 1.2 · stop: `end_of_turn` · 8 tokens · content check: False

*user*:

````text
What is the capital of Italy?
````

*assistant*:

````text
The capital of Italy is Rome.
````

*user*:

````text
And what is the capital of Germany?
````

*assistant (model)*:

````text
The capital of Italy is Rome.
````

**op01** (open) — penalty 1.0 (off) · stop: `end_of_turn` · 19 tokens · content check: None

*user*:

````text
Why is exercise good for you?
````

*assistant (model)*:

````text
People who are doing well, but some people might think they will do well on themselves.
````

**op01** (open) — penalty 1.2 · stop: `end_of_turn` · 60 tokens · content check: None

*user*:

````text
Why is exercise good for you?
````

*assistant (model)*:

````text
People who are doing well, but some people might think they will do better at themselves than others.  They can also do better at your owning and their work.  The same is a stress-related exercise program that helps you get more attention to the stress you are struggling with.
````
