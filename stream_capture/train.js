import dotenv from "dotenv";
dotenv.config();

import pdfParse from "pdf-parse";
import fs from "fs";
import { OpenAI } from "openai";
import { ChromaClient } from "chromadb";
import { createClient, AnamEvent } from "@anam-ai/js-sdk";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anamClient = createClient("YOUR_SESSION_TOKEN");

// === Load & Chunk PDF ===
async function loadPDFChunks(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
  const pdf = await pdfParse(dataBuffer);
  const text = pdf.text;
  const chunks = text.match(/(.|[\r\n]){1,1500}/g); // ~500 tokens per chunk
  return chunks;
}

// === Embed & Store ===
let vectorStore = [];

async function embedChunks(chunks) {
  for (let i = 0; i < chunks.length; i++) {
    const embedding = await openai.embeddings.create({
      input: chunks[i],
      model: "text-embedding-ada-002",
    });

    vectorStore.push({ embedding: embedding.data[0].embedding, text: chunks[i] });
  }
}

// === Cosine Similarity ===
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

// === Get Relevant Chunks ===
function getTopChunks(queryEmbedding, topK = 3) {
  return vectorStore
    .map((item) => ({
      ...item,
      score: cosineSimilarity(queryEmbedding, item.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// === Generate Answer from OpenAI ===
async function generateAnswer(userQuery) {
  const queryEmbedding = await openai.embeddings.create({
    input: userQuery,
    model: "text-embedding-ada-002",
  });

  const topChunks = getTopChunks(queryEmbedding.data[0].embedding);

  const context = topChunks.map((c) => c.text).join("\n---\n");

  const completion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant answering questions based on the following company data:\n" + context,
      },
      { role: "user", content: userQuery },
    ],
    model: "gpt-3.5-turbo",
  });

  return completion.choices[0].message.content;
}

// === Anam Listener ===
anamClient.addListener(AnamEvent.MESSAGE_HISTORY_UPDATED, async (messages) => {
  const latest = messages[messages.length - 1];
  if (latest.role === "user") {
    const answer = await generateAnswer(latest.content);
    anamClient.talk(answer);
  }
});

// === Startup ===
(async () => {
  const chunks = await loadPDFChunks("data/company_docs.pdf");
  await embedChunks(chunks);
  console.log("✅ PDF embedded and vector store ready.");
})();
