const { Pinecone } = require('@pinecone-database/pinecone');
const { pipeline } = require('@xenova/transformers');
const { extractText, getDocumentProxy } = require('unpdf');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// 1. Modern Initialization
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// Target the index specifically (this usually clears the strikethrough in VS Code)
const index = pc.index('hr-policies');

let embedder; // Move embedder outside to reuse across files

async function ingestPDF(filePath) {
  const t0 = performance.now(); // Start high-res timer
  try {
    if (!embedder) {
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }

    const fileName = path.basename(filePath);
    console.log(`\nProcessing: ${fileName}`);

    // Extract Text
    const dataBuffer = fs.readFileSync(filePath);
    const pdf = await getDocumentProxy(new Uint8Array(dataBuffer));
    const { text } = await extractText(pdf, { mergePages: true });

    if (!text || text.trim().length < 5) {
      console.log(`  └─ ⚠️  Skipping: Document is empty or unreadable.`);
      return;
    }

// Create Chunks with 200-character overlap
    const chunks = [];
    for (let i = 0; i < text.length; i += 800) {
      const start = i < 200 ? 0 : i - 200; 
      chunks.push(text.substring(start, i + 800));
    }
    const BATCH_SIZE = 10;
    const cleanVectors = [];

// MODIFICATION: Batched parallel processing for memory safety
    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE);
      
      const batchResults = await Promise.all(batch.map(async (content, index) => {
        const trimmed = content.trim();
        if (trimmed.length < 5) return null;

        const output = await embedder(trimmed, { pooling: 'mean', normalize: true });
        
        return {
          id: `vec-${Date.now()}-${i + index}`,
          values: Array.from(output.data),
          metadata: { 
            text: trimmed, 
            source: fileName, 
            context: `${fileName}: ${trimmed.substring(0, 100)}`
          }
        };
      }));

      // Push results and show progress for large files
      cleanVectors.push(...batchResults.filter(v => v));
      console.log(`  ├─ ⏳ Progress: ${Math.min(i + BATCH_SIZE, chunks.length)}/${chunks.length} chunks`);
    }

// Final Upsert
    if (cleanVectors.length) {
      console.log(`  ├─ 📦 Vectors Prepared: ${cleanVectors.length}`);
      try {
        await index.upsert(cleanVectors);
        const time = ((performance.now() - t0) / 1000).toFixed(2); // Calculate seconds
        console.log(`  └─ ✅ SUCCESS: Records stored in ${time}s.`);
      } catch (err) {
        console.error(`  └─ ❌ PINECONE REJECTION: ${err.message}`);
      }
    }
  } catch (error) {
    console.error(`  └─ ❌ SYSTEM ERROR: ${error.message}`);
  }
}

const runBatch = async (dir) => {
  console.log(`--- Starting Batch Ingestion: ${dir} ---`);
  
  if (!fs.existsSync(dir)) return console.log("Folder not found.");
  const files = fs.readdirSync(dir).filter(f => f.toLowerCase().endsWith('.pdf'));
  
  // Process files one by one, while chunks inside are batched-parallel
  for (const file of files) await ingestPDF(path.join(dir, file));
  console.log("--- Process Finished ---");
};

runBatch('./data');