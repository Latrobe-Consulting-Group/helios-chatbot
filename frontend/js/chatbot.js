function chatBot() {
    return {
        botTyping: false,
        messages: [{
            from: 'bot',
            text: 'I am Helios - I can look up information for you and learn over time. Try asking me a question!'
        }],
        output: function(input) {
            let text = input.trim();
            this.addChat(text);
        },
        addChat: function(query) {
            // Add user message
            this.messages.push({
                from: 'user',
                text: query
            });

            // Keep messages at most recent
            this.botTyping = true;
            this.scrollChat();

            // Make call to backend API
            response = fetch("http://127.0.0.1:8000/chat/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: query,
                })
            });
            
            text = response.then((response) => {
                return response.text();
            });

            bot_response = text.then((bot_response) => {
                return bot_response;
            });

            // Add bot message
            this.botTyping = false
            this.messages.push({
                from: 'bot',
                text: bot_response
            });
            this.scrollChat();
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