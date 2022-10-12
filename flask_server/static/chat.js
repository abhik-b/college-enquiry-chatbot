const chatInput = document.getElementById("chat-input");
const chatForm = document.getElementById("chat-form");
const chatbotFigure = document.querySelector('.mobile')

let apiURL = ''

function toggleChatBot() {
    chatbotFigure.classList.toggle('hidden')
}
chatForm.addEventListener("submit", (e) => {
    e.preventDefault();

    let userMsg = chatInput.value
    addMessage(userMsg, 'outgoing');

    fetch(`http://127.0.0.1:5000/chatbot_api/${apiURL}`, {
        method: "POST",
        body: JSON.stringify({ "message": userMsg }),
        mode: "cors",
        headers: { "Content-Type": "application/json" }
    }).then(r => r.json()).then(r => {
        if (r.url) {
            apiURL = r.url
        } else {
            apiURL = ''
        }
        addMessage(r.response, 'incoming');

        if (r.data) {
            addPDFBtn(r.data)
        }
    })

});

function addMessage(message, msgtype) {
    const chatMessage = document.createElement("div");
    chatMessage.classList.add("chat-message");
    chatMessage.classList.add(`${msgtype}-message`);
    chatMessage.innerText = message;
    document.querySelector(".chat-messages").appendChild(chatMessage);
    document.querySelector(".chat-messages").scrollTop +=
        chatMessage.getBoundingClientRect().y + 20;
    chatInput.value = "";
}

function addPDFBtn(data) {
    const chatMessage = document.createElement("div");
    chatMessage.classList.add("chat-message");
    chatMessage.classList.add(`incoming-message`);
    chatMessage.classList.add(`file-message`);
    chatMessage.innerText = data.filename;
    console.log(data.link)
    chatMessage.onclick = (e) => {
        window.open(data.link)
    }
    document.querySelector(".chat-messages").appendChild(chatMessage);
    document.querySelector(".chat-messages").scrollTop +=
        chatMessage.getBoundingClientRect().y + 10;
    chatInput.value = "";
}