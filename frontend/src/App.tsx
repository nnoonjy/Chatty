import "./App.css";
import { useState, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";

// 1. 타입 정의 (Interface)
// 메시지의 형태를 미리 정의합니다.
interface Message {
  text: string;
  sender: "user" | "bot"; // sender는 이 두 가지 값만 가질 수 있음
}

// 서버에서 받아올 응답 데이터 형태
interface ChatResponse {
  answer: string;
}

function App() {
  // 2. State에 제네릭(<...>)으로 타입 지정
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]); // Message들의 배열
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // 사용자 메시지 추가
    const userMessage: Message = { text: input, sender: "user" };
    const newMessages = [...messages, userMessage];

    setMessages(newMessages);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      // 응답 데이터도 타입을 지정해서 받음
      const data: ChatResponse = await response.json();

      // 봇 메시지 추가
      const botMessage: Message = { text: data.answer, sender: "bot" };
      setMessages([...newMessages, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = { text: "서버 연결 실패", sender: "bot" };
      setMessages([...newMessages, errorMessage]);
    }

    setIsLoading(false);
  };

  // 3. 이벤트 핸들러 타입 지정
  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  return (
    <div
      className="App"
      style={{
        maxWidth: "600px",
        margin: "0 auto",
        padding: "20px",
        fontFamily: "sans-serif",
      }}
    >
      {/* 헤더 */}
      <div
        style={{
          backgroundColor: "#00573D",
          color: "white",
          padding: "15px",
          borderRadius: "8px 8px 0 0",
          textAlign: "center",
        }}
      >
        <h2 style={{ margin: 0 }}>PNU AI Assistant (TS)</h2>
      </div>

      {/* 채팅창 */}
      <div
        style={{
          height: "500px",
          overflowY: "auto",
          border: "1px solid #ddd",
          borderTop: "none",
          padding: "20px",
          backgroundColor: "#f9f9f9",
        }}
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              justifyContent: msg.sender === "user" ? "flex-end" : "flex-start",
              marginBottom: "10px",
            }}
          >
            <div
              style={{
                maxWidth: "70%",
                padding: "10px 15px",
                borderRadius: "18px",
                backgroundColor: msg.sender === "user" ? "#00573D" : "white", // 부산대 컬러
                color: msg.sender === "user" ? "white" : "black",
                boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
                wordBreak: "break-word",
                textAlign:'left'
              }}
            >
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          </div>
        ))}
        {isLoading && (
          <div
            style={{ textAlign: "center", color: "#888", fontSize: "0.9em" }}
          >
            답변 생성 중...
          </div>
        )}
      </div>

      {/* 입력창 */}
      <div style={{ display: "flex", marginTop: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="질문을 입력하세요..."
          style={{
            flex: 1,
            padding: "12px",
            border: "1px solid #ddd",
            borderRadius: "4px",
          }}
        />
        <button
          onClick={sendMessage}
          style={{
            marginLeft: "10px",
            padding: "12px 24px",
            backgroundColor: "#00573D",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          전송
        </button>
      </div>
    </div>
  );
}

export default App;
