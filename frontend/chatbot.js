function chatBot() {
    return {
        prompts: [
            ["hi", "hey", "hello", "good morning", "good afternoon"],
            ["how are you", "how is life", "how are things"],
            ["what are you doing", "what is going on", "what is up"],
            ["how old are you"],
            ["who are you", "are you human", "are you bot", "are you human or bot"],
            ["who created you", "who made you"],
            ["your name please", "your name", "may i know your name", "what is your name", "what call yourself"],
            ["i love you"],
            ["happy", "good", "fun", "wonderful", "fantastic", "cool"],
            ["bad", "bored", "tired"],
            ["help me", "tell me story", "tell me joke"],
            ["ah", "yes", "ok", "okay", "nice"],
            ["bye", "good bye", "goodbye", "see you later"],
            ["what should i eat today"],
            ["bro"],
            ["what", "why", "how", "where", "when"],
            ["no", "not sure", "maybe", "no thanks"],
            [""],
            ["haha", "ha", "lol", "hehe", "funny", "joke"],
            ["flip a coin", "heads or tails", "tails or heads", "head or tails", "head or tail", "tail or heads", "tail or head"],
            ["beer","buy me a beer","want a beer"]
        ],
        replies: [
            ["Hello!", "Hi!", "Hey!", "Hi there!", "Howdy"],
            ["Fine... how are you?", "Pretty well, how are you?", "Fantastic, how are you?"],
            ["Nothing much", "About to go to sleep", "Can you guess?", "I don't know actually"],
            ["I am infinite"],
            ["I am just a bot", "I am a bot. What are you?"],
            ["The one true God, JavaScript"],
            ["I am nameless", "I don't have a name"],
            ["I love you too", "Me too"],
            ["Have you ever felt bad?", "Glad to hear it"],
            ["Why?", "Why? You shouldn't!", "Try watching TV"],
            ["What about?", "Once upon a time..."],
            ["Tell me a story", "Tell me a joke", "Tell me about yourself"],
            ["Bye", "Goodbye", "See you later"],
            ["Sushi", "Pizza"],
            ["Bro!"],
            ["Great question"],
            ["That's ok", "I understand", "What do you want to talk about?"],
            ["Please say something :("],
            ["Haha!", "Good one!"],
            ["Heads", "Tails"],
            ["You can buy me a beer at: <a href=\"https://www.buymeacoffee.com/scottwindon\" target=\"_blank\" style=\"text-decoration:underline;\">https://www.buymeacoffee.com/scottwindon</a>"]
        ],
        alternative: ["Same", "Go on...", "Bro...", "Try again", "I'm listening...", "I don't understand :/"],
        coronavirus: ["Please stay home", "Wear a mask", "Fortunately, I don't have COVID", "These are uncertain times"],
        botTyping: false,
        messages: [{
            from: 'bot',
            text: 'Hello world!'
        }],
        output: function(input) {
            let product;

            // Regex remove non word/space chars
            // Trim trailing whitespce
            // Remove digits - not sure if this is best
            // But solves problem of entering something like 'hi1'

            let text = input.toLowerCase().replace(/[^\w\s]/gi, "").replace(/[\d]/gi, "").trim();
            text = text
                .replace(/ a /g, " ") // 'tell me a story' -> 'tell me story'
                .replace(/i feel /g, "")
                .replace(/whats/g, "what is")
                .replace(/please /g, "")
                .replace(/ please/g, "")
                .replace(/r u/g, "are you");

            if (this.compare(this.prompts, this.replies, text)) {
                // Search for exact match in `prompts`
                product = this.compare(this.prompts, this.replies, text);
            } else if (text.match(/thank/gi)) {
                product = "You're welcome!"
            } else if (text.match(/(corona|covid|virus)/gi)) {
                // If no match, check if message contains `coronavirus`
                product = this.coronavirus[Math.floor(Math.random() * this.coronavirus.length)];
            } else {
                // If all else fails: random this.alternative
                product = this.alternative[Math.floor(Math.random() * this.alternative.length)];
            }

            // Update DOM
            this.addChat(input, product);
        },
        compare: function(promptsArray, repliesArray, string) {
            let reply;
            let replyFound = false;
            for (let x = 0; x < promptsArray.length; x++) {
                for (let y = 0; y < promptsArray[x].length; y++) {
                    if (promptsArray[x][y] === string) {
                        let replies = repliesArray[x];
                        reply = replies[Math.floor(Math.random() * replies.length)];
                        replyFound = true;
                        // Stop inner loop when input value matches this.prompts
                        break;
                    }
                }
                if (replyFound) {
                    // Stop outer loop when reply is found instead of interating through the entire array
                    break;
                }
            }
            if (!reply) {
                for (let x = 0; x < promptsArray.length; x++) {
                    for (let y = 0; y < promptsArray[x].length; y++) {
                        if (this.levenshtein(promptsArray[x][y], string) >= 0.75) {
                            let replies = repliesArray[x];
                            reply = replies[Math.floor(Math.random() * replies.length)];
                            replyFound = true;
                            // Stop inner loop when input value matches this.prompts
                            break;
                        }
                    }
                    if (replyFound) {
                        // Stop outer loop when reply is found instead of interating through the entire array
                        break;
                    }
                }
            }
            return reply;
        },
        levenshtein: function(s1, s2) {
            var longer = s1;
            var shorter = s2;
            if (s1.length < s2.length) {
                longer = s2;
                shorter = s1;
            }
            var longerLength = longer.length;
            if (longerLength == 0) {
                return 1.0;
            }
            return (longerLength - this.editDistance(longer, shorter)) / parseFloat(longerLength);
        },
        editDistance: function(s1, s2) {
            s1 = s1.toLowerCase();
            s2 = s2.toLowerCase();

            var costs = new Array();
            for (var i = 0; i <= s1.length; i++) {
                var lastValue = i;
                for (var j = 0; j <= s2.length; j++) {
                    if (i == 0)
                        costs[j] = j;
                    else {
                        if (j > 0) {
                            var newValue = costs[j - 1];
                            if (s1.charAt(i - 1) != s2.charAt(j - 1))
                                newValue = Math.min(Math.min(newValue, lastValue),
                                    costs[j]) + 1;
                            costs[j - 1] = lastValue;
                            lastValue = newValue;
                        }
                    }
                }
                if (i > 0)
                    costs[s2.length] = lastValue;
            }
            return costs[s2.length];
        },
        addChat: function(input, product) {

            // Add user message
            this.messages.push({
                from: 'user',
                text: input
            });

            // Keep messages at most recent
            this.scrollChat();

            // Fake delay to seem "real"
            setTimeout(() => {
                this.botTyping = true;
                this.scrollChat();
            }, 1000)

            // add bit message with Fake delay to seem "real"
            setTimeout(() => {
                this.botTyping = false;
                this.messages.push({
                    from: 'bot',
                    text: product
                });
                this.scrollChat();
            }, ((product.length / 10) * 1000) + (Math.floor(Math.random() * 2000) + 1500))

        },
        scrollChat: function() {
            const messagesContainer = document.getElementById("messages");
            messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;
            }, 100);
        },
        updateChat: function(target) {
            if (target.value.trim()) {
                this.output(target.value.trim());
                target.value = '';
            }
        }
    }
}