import { useState } from "react";

function App() {
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);

  // Input state
  const [name, setName] = useState("");
  const [position, setPosition] = useState("");
  const [age, setAge] = useState("");
  const [level, setLevel] = useState("");
  const [team, setTeam] = useState("");
  const [ppg, setPpg] = useState("");
  const [rpg, setRpg] = useState("");
  const [apg, setApg] = useState("");
  const [fgPct, setFgPct] = useState("");
  const [threePtPct, setThreePtPct] = useState("");

  // Validation function
  const isValidInput = () => {
    if (!name.trim() || !position.trim() || !level) return false;
    if (level === "highschool" && (parseInt(age) < 14 || parseInt(age) > 19)) return false;
    if (level === "college" && (parseInt(age) < 18 || parseInt(age) > 23)) return false;
    const numbers = [ppg, rpg, apg, fgPct, threePtPct];
    return numbers.every((n) => !isNaN(parseFloat(n)) && n.trim() !== "");
  };

  const analyzePlayer = async () => {
    if (!isValidInput()) {
      alert("Please fill all fields correctly.");
      return;
    }

    setReport("");
    setLoading(true);

    const response = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        position,
        age: age ? parseInt(age) : undefined,
        level,
        team: team || undefined,
        ppg: parseFloat(ppg),
        rpg: parseFloat(rpg),
        apg: parseFloat(apg),
        fg_pct: parseFloat(fgPct),
        three_pt_pct: parseFloat(threePtPct),
      }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const obj = JSON.parse(line);
          if (obj.response) {
            // Typing effect: one character at a time
            for (let i = 0; i < obj.response.length; i++) {
              setReport((prev) => prev + obj.response[i]);
              await new Promise((res) => setTimeout(res, 10));
            }
          }
        } catch (e) {
          console.error("JSON parse error", e);
        }
      }
    }

    setLoading(false);
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <h1 className="text-xl font-bold mb-4">NBA Draft Scout</h1>

      <div className="grid gap-2 mb-4">
        <input
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="Position"
          value={position}
          onChange={(e) => setPosition(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          type="number"
          placeholder="Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          className="border p-2 rounded"
        />
        <select
          value={level}
          onChange={(e) => setLevel(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="">Select Level</option>
          <option value="highschool">High School</option>
          <option value="college">College</option>
          <option value="nba">NBA</option>
        </select>
        <input
          placeholder="Team (optional)"
          value={team}
          onChange={(e) => setTeam(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="PPG"
          value={ppg}
          onChange={(e) => setPpg(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="RPG"
          value={rpg}
          onChange={(e) => setRpg(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="APG"
          value={apg}
          onChange={(e) => setApg(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="FG %"
          value={fgPct}
          onChange={(e) => setFgPct(e.target.value)}
          className="border p-2 rounded"
        />
        <input
          placeholder="3PT %"
          value={threePtPct}
          onChange={(e) => setThreePtPct(e.target.value)}
          className="border p-2 rounded"
        />
      </div>

      <button
        onClick={analyzePlayer}
        disabled={loading || !isValidInput()}
        className={`px-4 py-2 rounded-lg ${loading || !isValidInput()
          ? "bg-gray-400 cursor-not-allowed"
          : "bg-blue-600 text-white"
          }`}
      >
        {loading ? "Analyzing..." : "Analyze Player"}
      </button>

      <pre className="mt-4 whitespace-pre-wrap bg-gray-100 p-3 rounded-lg">
        {report || "Enter player stats and click analyze..."}
      </pre>
    </div>
  );
}

export default App;
