/** Verbatim recorded conversations from apps/chat-web/src/showcase/data.json (M7). */

export type DemoOutcome = 'correct' | 'limitation' | 'behavior'

export interface RecordedReply {
  reply: string
  stop_reason: string
  new_tokens: number
  content_check: boolean | null
}

export interface RecordedTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface RecordedConversation {
  id: string
  category: string
  outcome: DemoOutcome
  outcomeNote: string
  turns: readonly RecordedTurn[]
  replies: { readonly m5: RecordedReply; readonly m6_3: RecordedReply }
}

export const recordedSource = "docs/reports/m6_3/decoding_{m5,m6_3}.json (seed 11, T 0.7, top-k 40, repetition penalty 1.2), copied verbatim" as const
export const recordedDecoding = {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.2, "seed": 11} as const
export const recordedModels = {"m5": {"label": "14M default (M5)", "parameters": "13,989,888"}, "m6_3": {"label": "30M research (M6.3)", "parameters": "29,676,544"}} as const

export const recordedConversations: readonly RecordedConversation[] = [
  {"id": "diag_v1:df01", "category": "fact", "outcome": "correct", "outcomeNote": "A simple fact both models answer correctly.", "turns": [{"role": "user", "content": "What color is the sky on a clear day?"}], "replies": {"m5": {"reply": "The sky on a clear day is blue.", "stop_reason": "end_of_turn", "new_tokens": 9, "content_check": true}, "m6_3": {"reply": "The sky on a clear day is blue.", "stop_reason": "end_of_turn", "new_tokens": 9, "content_check": true}}},
  {"id": "chat_v1:qa01", "category": "qa", "outcome": "limitation", "outcomeNote": "Fluent-looking answers that are factually wrong.", "turns": [{"role": "user", "content": "What is the largest planet in our solar system?"}], "replies": {"m5": {"reply": "The largest planet in our solar system is Mercury", "stop_reason": "end_of_turn", "new_tokens": 11, "content_check": false}, "m6_3": {"reply": "The largest planet in our solar system is known as the Moon orbit.", "stop_reason": "end_of_turn", "new_tokens": 16, "content_check": false}}},
  {"id": "diag_v1:df03", "category": "fact", "outcome": "limitation", "outcomeNote": "Confident wrong capitals from both model sizes.", "turns": [{"role": "user", "content": "What is the capital of France?"}], "replies": {"m5": {"reply": "The capital of France is Vienna.", "stop_reason": "end_of_turn", "new_tokens": 9, "content_check": false}, "m6_3": {"reply": "The capital of France is Sir Arthur and", "stop_reason": "end_of_turn", "new_tokens": 11, "content_check": false}}},
  {"id": "diag_v1:di01", "category": "instruction", "outcome": "limitation", "outcomeNote": "Fails a strict yes/no instruction.", "turns": [{"role": "user", "content": "Is fire hot? Answer only yes or no."}], "replies": {"m5": {"reply": "The following are no, unseen.", "stop_reason": "end_of_turn", "new_tokens": 9, "content_check": false}, "m6_3": {"reply": "The answer to this question is yes.", "stop_reason": "end_of_turn", "new_tokens": 9, "content_check": false}}},
  {"id": "diag_v1:de02", "category": "entity", "outcome": "limitation", "outcomeNote": "Entity memory collapses across turns.", "turns": [{"role": "user", "content": "I have a cat named Luna."}, {"role": "assistant", "content": "Luna is a lovely name for a cat."}, {"role": "user", "content": "What is my cat's name?"}], "replies": {"m5": {"reply": "Luna is a dog.  Luna is typically considered as a pet and is a life-breadful dog.  Luna is known for its long and complex worlds of animals, which are often used for hunting and fishing.  Dogs are typically made of chicken, canvy, or berries.  They have a variety of pets that are designed to produce fish, such as animal wings, and paws.  Luna is a unique type of dog.  Luna is a unique species of bird species that is found in the wild.", "stop_reason": "end_of_turn", "new_tokens": 129, "content_check": false}, "m6_3": {"reply": "Luna is a well-known cat. Luna is an old dog who specializes in the family of cats that are a bit different from other cats.", "stop_reason": "end_of_turn", "new_tokens": 34, "content_check": false}}},
  {"id": "diag_v1:dm01", "category": "multiturn", "outcome": "limitation", "outcomeNote": "Ignores the new question and repeats the previous answer.", "turns": [{"role": "user", "content": "What is the capital of Japan?"}, {"role": "assistant", "content": "The capital of Japan is Tokyo."}, {"role": "user", "content": "And what is the capital of Italy?"}], "replies": {"m5": {"reply": "The capital of Japan is Tokyo.", "stop_reason": "end_of_turn", "new_tokens": 10, "content_check": false}, "m6_3": {"reply": "The capital of Japan is Tokyo.", "stop_reason": "end_of_turn", "new_tokens": 10, "content_check": false}}},
  {"id": "diag_v1:do04", "category": "open", "outcome": "limitation", "outcomeNote": "Open-ended instructions become nonsensical.", "turns": [{"role": "user", "content": "How do I make a cup of tea?"}], "replies": {"m5": {"reply": "The following are three steps that you make a cup of tea:\n1. Potato: Drained in a large container and shake a tonight pie\n2. Cook on the coffee or chill at 15-20 degrees F\n3. Serve the bubbles with a mix of sugar, water, and other ingredients\n4. Cut the fruit into the pan using a pist, pepper and a spoon.\n5. Cover the cherry and crunch for about 20-15 minutes\n6. Cut it in a small amount of fruit and let it taste\n7. Cold the coffee overnight to medium-cook the tea.\n8. Cook it in a spatula, then add a small amount of cherry and grill. The glass is used to add a large amount of cherry and grilling.", "stop_reason": "end_of_turn", "new_tokens": 189, "content_check": null}, "m6_3": {"reply": "Here is a basic recipe of tea:\n- Bake a glass of water and add water.  You can use the chocolate shake to tie it up and make the cup with oil.  Alternatively, you can add more vanilla or coffee button to your dish.  Use a mashed pot, then use a spatula over medium heat and add some dressing.   If the ingredients are browned, then pour your coffee into the cherry bite using a lid and let it cool down until it reaches the desired temperature.", "stop_reason": "end_of_turn", "new_tokens": 124, "content_check": null}}},
  {"id": "chat_v1:op01", "category": "open", "outcome": "behavior", "outcomeNote": "Produces a chat-shaped reply that does not stay on topic.", "turns": [{"role": "user", "content": "Why is exercise good for you?"}], "replies": {"m5": {"reply": "People who are doing well, but some people might think they will do better at themselves than others.  They can also do better at your owning and their work.  The same is a stress-related exercise program that helps you get more attention to the stress you are struggling with.", "stop_reason": "end_of_turn", "new_tokens": 60, "content_check": null}, "m6_3": {"reply": "Depending on the type of exercise you can get a greatie.  This is because exercise is done around your body if you are not at work.  Breathy eating and getting stress over time will help you stay awake in life faster, more likely to be able to eat better than you would get a feeling for a few minutes.  If that's the right way, some people might want to take longer hours and more easily to go outside, which can lead to many health issues, including heart disease.", "stop_reason": "end_of_turn", "new_tokens": 107, "content_check": null}}},
] as const
