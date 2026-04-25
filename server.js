require('dotenv').config();
const express = require('express');
const { Groq } = require('groq-sdk');
const { Pinecone } = require('@pinecone-database/pinecone');
const { pipeline } = require('@xenova/transformers');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { hrQuerySchema, validate } = require('./validate');
const crypto = require('crypto');
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20, // Limit each IP to 20 requests per window
  message: { error: "Too many requests, please try again later." }
});


const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // This lets you see index.html at http://localhost:3000
app.use("/ask", limiter);

// Initialize Clients
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index("hr-policies");

let embedder;
const init = async () => {
  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log("AI Search Engine Ready (MiniLM-L6)");
};

// The "Ask" Route
app.post('/ask', validate(hrQuerySchema), async (req, res) => {
  try {
    const { question } = req.body;

    // --- Custom Banana Interceptor ---
    const lowerQuestion = question.toLowerCase().trim();
    if (lowerQuestion.includes("why care about bananas")) {
      return res.json({ answer: "Fart Bite" });
    }
   
    // 1. Generate real math for the user's question
    const output = await embedder(question, { pooling: 'mean', normalize: true });
    const userVector = Array.from(output.data);

    // 2. Search Pinecone using the real vector
    const queryResponse = await index.query({
      vector: userVector, 
      topK: 3,
      includeMetadata: true
    });

// Enhanced context mapping with fallback
    const context = queryResponse.matches && queryResponse.matches.length > 0 
      ? queryResponse.matches.map(m => m.metadata.text).join("\n\n")
      : "NO_CONTEXT_AVAILABLE";

    // 2. Send to Llama 3.2 via Groq
    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { 
          role: "system", 
          content: `You are an HR Assistant. Answer using ONLY this context: ${context}. 
                    If not found, say you don't know.` 
        },
        { role: "user", content: question }
      ],
      model: "llama-3.3-70b-versatile",
    });

const structuredResponse = {
      answer: chatCompletion.choices[0].message.content,
      sources: queryResponse.matches.map(m => ({
        file: m.metadata?.source || "Policy Document",
        snippet: m.metadata?.text ? m.metadata.text.substring(0, 150) + "..." : ""
      })),
      request_id: `hr_uuid_${crypto.randomUUID()}`,
      model_used: "llama-3.2-3b-preview"
    };

    res.json(structuredResponse);
  } catch (error) {
    console.error(error);
    res.status(500).json({ 
        error: "Something went wrong on the server." ,
        request_id: `hr_uuid_${crypto.randomUUID()}`
    });
  }
});

init().then(() => {
  app.listen(port, () => {
    console.log(`HR Bot is live at http://localhost:${port}`);
  });
});