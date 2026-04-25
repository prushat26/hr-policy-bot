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

    // Create Chunks
    const chunks = text.match(/[\s\S]{1,1000}/g) || [];
    const vectors = [];

    for (let i = 0; i < chunks.length; i++) {
      const content = chunks[i].trim();
      if (content.length < 2) continue;

      const output = await embedder(content, { pooling: 'mean', normalize: true });
      const embedding = Array.from(output.data);

      vectors.push({
        id: `vec-${Date.now()}-${i}`,
        values: embedding,
        metadata: { text: content, source: fileName }
      });
    }

    // Final Upsert
    if (vectors.length > 0) {
      console.log(`  ├─ Vectors Prepared: ${vectors.length}`);
      console.log(`  ├─ Vector Dimension: ${vectors[0].values.length}`);
      
      try {
        // Modern SDKs prefer passing the array directly to .upsert()
        await index.upsert(vectors);
        console.log(`  └─ ✅ SUCCESS: ${vectors.length} records are now in Pinecone.`);
      } catch (pineconeErr) {
        // This is the error handler for the "1 record" failure
        console.error(`  └─ ❌ PINECONE REJECTION:`, pineconeErr.message);
        console.log(`     CRITICAL: Go to Pinecone Dashboard and ensure 'hr-policies' is 384 dimensions.`);
      }
    }

  } catch (error) {
    console.error(`  └─ ❌ SYSTEM ERROR:`, error.message);
  }
}

const runBatch = async (dir) => {
  console.log(`--- Starting Batch Ingestion: ${dir} ---`);
  
  if (!fs.existsSync(dir)) {
    console.log(`  └─ ❌ Folder '${dir}' not found.`);
    return;
  }

  const files = fs.readdirSync(dir).filter(f => f.toLowerCase().endsWith('.pdf'));
  
  if (files.length === 0) {
    console.log(`  └─ ⚠️  No PDFs found in ${dir}`);
    return;
  }

  for (const file of files) {
    await ingestPDF(path.join(dir, file));
  }
  console.log("\n--- Process Finished ---");
};

runBatch('./data');