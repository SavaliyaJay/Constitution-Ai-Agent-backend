const express = require("express");
const dotenv = require("dotenv");
const { createClient } = require("@libsql/client");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");

dotenv.config();
const app = express();
const PORT = 5000;
const cors = require("cors");

app.use(express.json({
  limit: '50mb',
  extended: true
}));
app.use(express.urlencoded({
  limit: '50mb',
  extended: true
}));

app.use(cors({
  origin: ['https://constitution-ai-agent.vercel.app', 'http://localhost:3000'],
  methods: ['GET', 'POST', 'DELETE', 'PUT', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: false
}));

app.options('*', cors());

// Initialize LibSQL (Turso) client
const db = createClient({
  url: process.env.LIBSQL_DB_URL,
  authToken: process.env.LIBSQL_DB_AUTH,
});

// Progress Tracking Class
class ProgressTracker {
  constructor(res) {
    this.res = res;
    this.isStreaming = false;
  }

  startStreaming() {
    if (!this.isStreaming) {
      this.res.writeHead(200, {
        'Content-Type': 'text/stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      this.isStreaming = true;
    }
  }

  updateProgress(progress, stage) {
    if (this.isStreaming) {
      this.res.write(`data: ${JSON.stringify({ type: 'progress', progress, stage })}\n\n`);
      console.log(`Progress: ${progress}% - ${stage}`);
    }
  }

  complete(result) {
    if (this.isStreaming) {
      this.res.write(`data: ${JSON.stringify({ type: 'complete', result })}\n\n`);
      this.res.end();
    }
  }

  error(error) {
    if (this.isStreaming) {
      this.res.write(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`);
      this.res.end();
    }
  }
}

// Initialize database tables with vector indexes
async function initializeDatabase() {
  console.time("Database Initialization");
  try {
    await db.execute(`
      CREATE TABLE IF NOT EXISTS constitution_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_text TEXT NOT NULL,
        embedding TEXT NOT NULL,
        article_section TEXT,
        chunk_type TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await db.execute(`
      CREATE TABLE IF NOT EXISTS user_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        response TEXT,
        relevant_chunks TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await db.execute(`
      CREATE INDEX IF NOT EXISTS idx_chunk_type ON constitution_chunks(chunk_type)
    `);

    await db.execute(`
      CREATE INDEX IF NOT EXISTS idx_article_section ON constitution_chunks(article_section)
    `);

    console.log("Database tables and indexes initialized successfully");
  } catch (error) {
    console.error("Error initializing database:", error);
  } finally {
    console.timeEnd("Database Initialization");
  }
}

// Memory-optimized LangChain text splitting
async function splitTextWithLangChain(text, chunkSize = 500, chunkOverlap = 50) {
  console.time("LangChain Splitting");
  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: chunkSize,
      chunkOverlap: chunkOverlap,
    });
    const docs = await splitter.createDocuments([text]);
    const result = docs.map(doc => doc.pageContent);
    
    // Clear references to help GC
    docs.length = 0;
    
    return result;
  } finally {
    console.timeEnd("LangChain Splitting");
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  }
}

// Get embeddings from Cloudflare with retry logic and memory cleanup
async function getEmbeddings(text) {
  const maxRetries = 3;
  let retryCount = 0;
  console.time(`Fetching Embedding (text length: ${text.length})`);

  while (retryCount < maxRetries) {
    try {
      const response = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/baai/bge-large-en-v1.5`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text }),
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Cloudflare API Error Response:", errorText);
        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      if (!result.result || !result.result.data || !result.result.data[0]) {
        console.error("Unexpected Cloudflare response structure:", result);
        throw new Error("Invalid embedding data received from Cloudflare.");
      }
      
      console.timeEnd(`Fetching Embedding (text length: ${text.length})`);
      return result.result.data[0];

    } catch (err) {
      retryCount++;
      console.error(`Attempt ${retryCount} failed:`, err.message);

      if (retryCount >= maxRetries) {
        console.error("Max retries reached, throwing error");
        console.timeEnd(`Fetching Embedding (text length: ${text.length})`);
        throw err;
      }

      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
    }
  }
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// MEMORY OPTIMIZATION: Chunking functions with improved memory management
const PREAMBLE_PATTERN = /WE,?\s+THE\s+PEOPLE\s+OF\s+INDIA/i;
const PART_PATTERN = /(PART\s+[IVX]+[A-Z]*\s*[-–—]?\s*[A-Z\s.,-]+)/gi;
const ARTICLE_HEADER_PATTERN = /(?:^|\n)\s*(\d+[A-Z]*)\.\s*([^\n]+)/gm;
const SCHEDULE_PATTERN = /((?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE)/gi;
const APPENDIX_PATTERN = /(APPENDIX\s+[IVX]+(?:\s*[-–—]?\s*[A-Z\s.,-]+)?)/gi;

async function splitByArticles(parentTitle, text) {
  const chunks = [];
  const articleMatches = [...text.matchAll(ARTICLE_HEADER_PATTERN)];
  let lastIndex = 0;

  for (let i = 0; i < articleMatches.length; i++) {
    const currentArticleMatch = articleMatches[i];
    const nextArticleMatch = articleMatches[i + 1];

    if (currentArticleMatch.index > lastIndex) {
      const precedingText = text.substring(lastIndex, currentArticleMatch.index).trim();
      if (precedingText.length > 50) {
        chunks.push({
          title: `${parentTitle} - Introduction`,
          text: precedingText,
          type: "article_intro"
        });
      }
    }

    const articleNumber = currentArticleMatch[1];
    const articleTitleLine = currentArticleMatch[2].trim();
    const articleContentStart = currentArticleMatch.index;
    const articleContentEnd = nextArticleMatch ? nextArticleMatch.index : text.length;
    const fullArticleText = text.substring(articleContentStart, articleContentEnd).trim();
    const finalChunkTitle = `${parentTitle} - Article ${articleNumber}: ${articleTitleLine}`;

    if (fullArticleText.length > 1500) {
      console.log(`Article ${articleNumber} is large (${fullArticleText.length} chars), further splitting...`);
      const subDocs = await splitTextWithLangChain(fullArticleText, 500, 50);
      subDocs.forEach((doc, idx) => {
        chunks.push({
          title: `${finalChunkTitle} (Part ${idx + 1})`,
          text: doc,
          type: "article_sub_part",
          articleNumber: articleNumber
        });
      });
    } else if (fullArticleText.length > 50) {
      chunks.push({
        title: finalChunkTitle,
        text: fullArticleText,
        type: "article",
        articleNumber: articleNumber
      });
    }
    lastIndex = articleContentEnd;
  }

  // Handle remaining content
  if (lastIndex < text.length) {
    const trailingText = text.substring(lastIndex).trim();
    if (trailingText.length > 50) {
      chunks.push({
        title: `${parentTitle} - Trailing Content`,
        text: trailingText,
        type: "trailing_content"
      });
    }
  }

  // Handle articles not found case
  if (articleMatches.length === 0 && text.length > 100) {
    if (text.length > 3000) {
      console.log(`Large un-articled section in "${parentTitle}" (${text.length} chars), using LangChain.`);
      const subDocs = await splitTextWithLangChain(text);
      subDocs.forEach((doc, idx) => {
        chunks.push({
          title: `${parentTitle} (Section ${idx + 1})`,
          text: doc,
          type: "generic_section_fallback"
        });
      });
    } else {
      chunks.push({
        title: `${parentTitle} (General Section)`,
        text: text,
        type: "generic_section"
      });
    }
  }

  return chunks;
}

async function smartLogicalChunking(text, progressTracker = null) {
  console.time("Smart Logical Chunking");
  console.log("Starting smart logical chunking...");
  const allStructuredChunks = [];

  if (progressTracker) {
    progressTracker.updateProgress(22, 'Cleaning document text...');
  }

  // Clean text
  let cleanText = text
    .replace(/Page \d+:/g, '')
    .replace(/\n\s*\n+/g, '\n\n')
    .trim();

  // Extract preamble
  if (progressTracker) {
    progressTracker.updateProgress(25, 'Extracting preamble...');
  }

  const preambleMatch = cleanText.match(PREAMBLE_PATTERN);
  if (preambleMatch) {
    const preambleStart = cleanText.indexOf(preambleMatch[0]);
    const firstPartOrScheduleIndex = Math.min(
      ...[
        cleanText.indexOf('PART I', preambleStart + 1),
        cleanText.indexOf('FIRST SCHEDULE', preambleStart + 1),
        cleanText.indexOf('APPENDIX', preambleStart + 1),
        cleanText.length
      ].filter(idx => idx !== -1)
    );

    if (firstPartOrScheduleIndex > preambleStart) {
      allStructuredChunks.push({
        title: "PREAMBLE",
        text: cleanText.substring(preambleStart, firstPartOrScheduleIndex).trim(),
        type: "preamble"
      });
      cleanText = cleanText.substring(firstPartOrScheduleIndex).trim();
    }
  }

  // Process parts with progress tracking
  if (progressTracker) {
    progressTracker.updateProgress(30, 'Processing document parts...');
  }

  const partMatches = [...cleanText.matchAll(PART_PATTERN)];
  
  for (let i = 0; i < partMatches.length; i++) {
    const progress = 30 + Math.round((i / partMatches.length) * 20); // 30-50%
    if (progressTracker) {
      progressTracker.updateProgress(progress, `Processing Part ${i + 1} of ${partMatches.length}...`);
    }

    const currentPartMatch = partMatches[i];
    const nextPartMatch = partMatches[i + 1];
    const partTitle = currentPartMatch[1].trim();
    const partContentStart = currentPartMatch.index + currentPartMatch[0].length;
    const partContentEnd = nextPartMatch ? nextPartMatch.index : cleanText.length;
    const partTextContent = cleanText.substring(partContentStart, partContentEnd).trim();

    console.log(`Processing ${partTitle}...`);
    const partSubChunks = await splitByArticles(partTitle, partTextContent);
    allStructuredChunks.push(...partSubChunks);
    lastProcessedIndex = partContentEnd;

    // Memory cleanup between parts
    if (global.gc && i % 5 === 0) {
      global.gc();
    }
  }

  // Extract schedules and appendices
  if (progressTracker) {
    progressTracker.updateProgress(50, 'Extracting schedules...');
  }

  const scheduleChunks = await extractSchedules(text);
  allStructuredChunks.push(...scheduleChunks);

  const appendixChunks = await extractAppendices(text);
  allStructuredChunks.push(...appendixChunks);

  const finalChunks = allStructuredChunks.filter(chunk => chunk.text && chunk.text.trim().length > 50);

  console.log(`Smart hierarchical chunking complete: ${finalChunks.length} chunks created.`);
  console.timeEnd("Smart Logical Chunking");

  // Final cleanup
  if (global.gc) {
    global.gc();
  }

  return finalChunks;
}

// Simplified schedule/appendix extraction functions (keeping original logic)
async function extractSchedules(originalText) {
  const chunks = [];
  const scheduleMatches = [...originalText.matchAll(SCHEDULE_PATTERN)];

  for (let i = 0; i < scheduleMatches.length; i++) {
    const currentSchedule = scheduleMatches[i];
    const nextSchedule = scheduleMatches[i + 1];
    const scheduleStart = currentSchedule.index;
    const appendixIndex = originalText.indexOf('APPENDIX', scheduleStart + 1);
    const endOfDocument = originalText.length;
    let scheduleEnd = nextSchedule ? nextSchedule.index : (appendixIndex !== -1 ? appendixIndex : endOfDocument);

    const scheduleText = originalText.substring(scheduleStart, scheduleEnd).trim();
    const scheduleTitle = currentSchedule[1].trim();

    if (scheduleText.length > 100) {
      if (scheduleText.length > 5000) {
        const subDocs = await splitTextWithLangChain(scheduleText, 1000, 100);
        subDocs.forEach((doc, idx) => {
          chunks.push({
            title: `${scheduleTitle} (Part ${idx + 1})`,
            text: doc,
            type: "schedule_part"
          });
        });
      } else {
        chunks.push({
          title: scheduleTitle,
          text: scheduleText,
          type: "schedule"
        });
      }
    }
  }
  return chunks;
}

async function extractAppendices(originalText) {
  const chunks = [];
  const appendixMatches = [...originalText.matchAll(APPENDIX_PATTERN)];

  for (let i = 0; i < appendixMatches.length; i++) {
    const currentAppendix = appendixMatches[i];
    const nextAppendix = appendixMatches[i + 1];
    const appendixStart = currentAppendix.index;
    const appendixEnd = nextAppendix ? nextAppendix.index : originalText.length;
    const appendixText = originalText.substring(appendixStart, appendixEnd).trim();
    const appendixTitle = currentAppendix[1].trim();

    if (appendixText.length > 100) {
      if (appendixText.length > 5000) {
        const subDocs = await splitTextWithLangChain(appendixText, 1000, 100);
        subDocs.forEach((doc, idx) => {
          chunks.push({
            title: `${appendixTitle} (Part ${idx + 1})`,
            text: doc,
            type: "appendix_part"
          });
        });
      } else {
        chunks.push({
          title: appendixTitle,
          text: appendixText,
          type: "appendix"
        });
      }
    }
  }
  return chunks;
}

// AI-based chunking with memory optimization
async function chunkTextWithCloudflareAI(text, progressTracker = null) {
  console.time("Cloudflare AI Chunking");
  
  if (progressTracker) {
    progressTracker.updateProgress(20, 'Sending to AI for intelligent chunking...');
  }

  // Split large texts into smaller chunks for AI processing
  const maxChunkSize = 100000; // 100KB limit for AI processing
  const textChunks = [];
  
  if (text.length > maxChunkSize) {
    console.log(`Text too large (${text.length} chars), splitting for AI processing...`);
    for (let i = 0; i < text.length; i += maxChunkSize) {
      textChunks.push(text.substring(i, i + maxChunkSize));
    }
  } else {
    textChunks.push(text);
  }

  const allChunks = [];

  for (let i = 0; i < textChunks.length; i++) {
    const chunk = textChunks[i];
    const prompt = `
You are an expert legal assistant specializing in constitutional law. Your task is to split the following constitutional text into logical, self-contained chunks.

Each chunk should represent a complete thought, a single legal concept, or a distinct article. Do not split in the middle of a sentence.

Return the response as a valid JSON array of objects, where each object has a "title" and a "text" property.
- The "title" should be a concise, descriptive heading for the chunk.
- The "text" should be the full text of that chunk.

Here is the text to process:
---
${chunk}
---
    `;

    try {
      const response = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            messages: [{
              role: "system",
              content: "You are a helpful assistant that splits long documents into structured JSON."
            }, {
              role: "user",
              content: prompt
            }]
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Cloudflare API error: ${response.status}`);
      }

      const result = await response.json();
      const aiResponse = result.result.response;

      try {
        const chunks = JSON.parse(aiResponse);
        const processedChunks = chunks.map(chunk => ({
          ...chunk,
          type: 'ai_generated_chunk'
        }));
        allChunks.push(...processedChunks);
      } catch (e) {
        console.error("Failed to parse JSON response from AI:", e);
        allChunks.push({
          title: `Unstructured AI Chunk ${i + 1}`,
          text: aiResponse,
          type: 'ai_fallback_unstructured'
        });
      }
    } catch (error) {
      console.error(`Error processing chunk ${i + 1}:`, error);
      throw error;
    }
  }

  if (progressTracker) {
    progressTracker.updateProgress(40, `AI chunking complete: ${allChunks.length} chunks created`);
  }

  console.timeEnd("Cloudflare AI Chunking");
  return allChunks;
}

// Main chunking orchestrator with memory optimization
async function aiBasedChunking(text, strategy = "ai", progressTracker = null) {
  console.time(`aiBasedChunking (strategy: ${strategy})`);
  
  try {
    if (progressTracker) {
      progressTracker.updateProgress(10, `Starting ${strategy} chunking...`);
    }

    let result;
    if (strategy === "ai") {
      console.log("Using Cloudflare AI for smart chunking...");
      result = await chunkTextWithCloudflareAI(text, progressTracker);
    } else if (strategy === "logical") {
      console.log("Using smart logical chunking (regex-based)...");
      result = await smartLogicalChunking(text, progressTracker);
    } else {
      console.log("Using LangChain recursive splitter...");
      if (progressTracker) {
        progressTracker.updateProgress(20, 'Processing with LangChain splitter...');
      }
      const chunks = await splitTextWithLangChain(text);
      result = chunks.map((chunk, index) => ({
        title: `General Section ${index + 1}`,
        text: chunk,
        type: "langchain_general"
      }));
      if (progressTracker) {
        progressTracker.updateProgress(40, `LangChain splitting complete: ${result.length} chunks created`);
      }
    }

    // Memory cleanup
    if (global.gc) {
      global.gc();
    }

    return result;

  } catch (error) {
    console.error(`Chunking with strategy '${strategy}' failed:`, error.message);
    if (progressTracker) {
      progressTracker.updateProgress(20, 'Chunking failed, falling back to LangChain...');
    }
    
    // Fallback to simpler method
    console.log("Falling back to LangChain recursive splitter...");
    const chunks = await splitTextWithLangChain(text);
    const result = chunks.map((chunk, index) => ({
      title: `General Section ${index + 1} (Fallback)`,
      text: chunk,
      type: "langchain_fallback"
    }));
    
    if (progressTracker) {
      progressTracker.updateProgress(40, `Fallback chunking complete: ${result.length} chunks created`);
    }
    
    return result;
  } finally {
    console.timeEnd(`aiBasedChunking (strategy: ${strategy})`);
  }
}

// CRITICAL MEMORY OPTIMIZATION: Batch processing for embeddings with cleanup
async function processEmbeddingsInBatches(chunks, progressTracker = null) {
  console.time("Batch Embedding Processing");
  
  const embeddingsArray = [];
  const failedEmbeddings = [];
  const batchSize = 5; // Process 5 embeddings at a time
  const totalBatches = Math.ceil(chunks.length / batchSize);
  
  for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
    const startIdx = batchIndex * batchSize;
    const endIdx = Math.min(startIdx + batchSize, chunks.length);
    const batch = chunks.slice(startIdx, endIdx);
    
    const progress = 45 + Math.round((batchIndex / totalBatches) * 35); // 45-80%
    if (progressTracker) {
      progressTracker.updateProgress(progress, `Processing embeddings batch ${batchIndex + 1}/${totalBatches}`);
    }
    
    console.log(`Processing embedding batch ${batchIndex + 1}/${totalBatches} (${startIdx + 1}-${endIdx})`);
    
    // Process batch in parallel but with limited concurrency
    const batchPromises = batch.map(async (chunk, localIndex) => {
      const globalIndex = startIdx + localIndex;
      try {
        const embedding = await getEmbeddings(chunk.text);
        return { index: globalIndex, embedding, success: true };
      } catch (error) {
        console.error(`Failed to embed chunk ${globalIndex + 1}:`, error.message);
        return { 
          index: globalIndex, 
          error: {
            index: globalIndex,
            title: chunk.title,
            text_snippet: chunk.text.substring(0, 100) + "...",
            error: error.message
          }, 
          success: false 
        };
      }
    });
    
    const batchResults = await Promise.all(batchPromises);
    
    // Process results
    batchResults.forEach(result => {
      if (result.success) {
        embeddingsArray[result.index] = result.embedding;
      } else {
        embeddingsArray[result.index] = null;
        failedEmbeddings.push(result.error);
      }
    });
    
    // Memory cleanup between batches
    if (global.gc) {
      global.gc();
    }
    
    // Rate limiting between batches
    if (batchIndex < totalBatches - 1) {
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }
  
  console.timeEnd("Batch Embedding Processing");
  return { embeddingsArray, failedEmbeddings };
}

// Store chunks with optimized batch processing and transactions
async function storeChunksInDatabase(structuredChunks, rawEmbeddings, progressTracker = null) {
  console.time("Database Storage");
  try {
    console.log(`Storing ${structuredChunks.length} chunks in database...`);

    const batchSize = 10;
    const totalBatches = Math.ceil(structuredChunks.length / batchSize);

    for (let i = 0; i < structuredChunks.length; i += batchSize) {
      const batchIndex = Math.floor(i / batchSize) + 1;
      const progress = 80 + Math.round((batchIndex / totalBatches) * 18); // 80-98%
      
      if (progressTracker) {
        progressTracker.updateProgress(progress, `Storing batch ${batchIndex}/${totalBatches} in database...`);
      }

      console.time(`DB Batch Insert ${batchIndex}`);
      
      // Using transactions for safety and performance
      const tx = await db.transaction("write");
      try {
        const batch = structuredChunks.slice(i, i + batchSize);
        const batchEmbeddings = rawEmbeddings.slice(i, i + batchSize);

        const promises = [];
        for (let j = 0; j < batch.length; j++) {
          const chunk = batch[j];
          const embedding = batchEmbeddings[j];
          if (embedding) {
            promises.push(
              tx.execute({
                sql: `INSERT INTO constitution_chunks (chunk_text, embedding, article_section, chunk_type) VALUES (?, ?, ?, ?)`,
                args: [chunk.text, JSON.stringify(embedding), chunk.title, chunk.type]
              })
            );
          }
        }

        await Promise.all(promises);
        await tx.commit();
        console.log(`Processed batch ${batchIndex}/${totalBatches}`);
        
      } catch (err) {
        await tx.rollback();
        console.error(`Error in batch ${batchIndex}, rolling back:`, err);
        throw err;
      }

      console.timeEnd(`DB Batch Insert ${batchIndex}`);
      
      // Memory cleanup and rate limiting
      if (global.gc && batchIndex % 3 === 0) {
        global.gc();
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    if (progressTracker) {
      progressTracker.updateProgress(98, 'Database storage complete!');
    }

    console.log(`Successfully stored ${structuredChunks.length} chunks in database`);
  } catch (error) {
    console.error("Error storing chunks in database:", error);
    throw error;
  } finally {
    console.timeEnd("Database Storage");
  }
}

// CRITICAL MEMORY FIX: Paginated vector search instead of loading all chunks
async function searchRelevantSections(queryEmbedding, limit = 5, chunkTypeFilter = null) {
  console.time("Paginated Vector Search");
  try {
    const batchSize = 1000; // Process chunks in batches of 1000
    let offset = 0;
    let allSimilarities = [];
    let hasMore = true;

    while (hasMore) {
      // Build query with pagination
      let sql = "SELECT * FROM constitution_chunks";
      let args = [];

      if (chunkTypeFilter) {
        sql += " WHERE chunk_type = ?";
        args.push(chunkTypeFilter);
      }

      sql += " LIMIT ? OFFSET ?";
      args.push(batchSize, offset);

      const result = await db.execute({ sql, args });
      const chunks = result.rows;

      if (chunks.length === 0) {
        hasMore = false;
        break;
      }

      // Process this batch of chunks
      const batchSimilarities = chunks.map(chunk => {
        try {
          const chunkEmbedding = JSON.parse(chunk.embedding);
          const similarity = cosineSimilarity(queryEmbedding, chunkEmbedding);
          return { ...chunk, similarity };
        } catch (e) {
          console.error(`Error parsing embedding JSON for chunk ID ${chunk.id}:`, e);
          return { ...chunk, similarity: -1 };
        }
      });

      // Only keep chunks above similarity threshold
      const relevantBatchChunks = batchSimilarities.filter(chunk => chunk.similarity > 0.3);
      allSimilarities.push(...relevantBatchChunks);

      // Update offset for next batch
      offset += batchSize;
      
      // If we got fewer chunks than batch size, we've reached the end
      if (chunks.length < batchSize) {
        hasMore = false;
      }

      // Memory cleanup between batches
      chunks.length = 0;
      batchSimilarities.length = 0;
      if (global.gc && offset % (batchSize * 3) === 0) {
        global.gc();
      }
    }

    // Sort all results and return top matches
    const topResults = allSimilarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
      
    console.log(`Found ${topResults.length} relevant sections from ${offset} total chunks processed`);
    return topResults;

  } catch (error) {
    console.error("Error in paginated vector search:", error);
    throw error;
  } finally {
    console.timeEnd("Paginated Vector Search");
    // Final cleanup
    if (global.gc) {
      global.gc();
    }
  }
}

// Expand user query with AI for better context
async function expandQueryWithAI(query) {
  console.time("AI Query Expansion");
  console.log(`Expanding query with AI: "${query}"`);

  const prompt = `You are a legal expert specializing in constitutional law. Your task is to expand the following user query to improve its chances of finding relevant articles in a constitutional database.

Rephrase the query into a more detailed question or a descriptive statement. Include synonyms and related legal concepts. The goal is to create a text that is ideal for a vector database search.

Original Query: "${query}"

Expanded Query:`;

  try {
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [{
            role: "system",
            content: "You are an AI assistant that expands user queries for better information retrieval from a legal database."
          }, {
            role: "user",
            content: prompt
          }],
          max_tokens: 256,
          temperature: 0.2
        }),
      }
    );

    if (!response.ok) {
      const errorDetail = await response.text();
      throw new Error(`HTTP error! status: ${response.status} - ${errorDetail}`);
    }

    const result = await response.json();
    const expandedQuery = result.result.response.trim();

    console.log(`Expanded Query: "${expandedQuery}"`);
    return expandedQuery;

  } catch (error) {
    console.error("Error expanding query with AI:", error.message);
    console.log("Falling back to the original query for embedding.");
    return query;
  } finally {
    console.timeEnd("AI Query Expansion");
  }
}

// Generate response using Cloudflare AI with Enhanced Prompt
async function generateResponse(query, relevantSections) {
  console.time("AI Response Generation");
  try {
    const context = relevantSections
      .map(section => `### ${section.article_section}\n${section.chunk_text}`)
      .join("\n\n---\n\n");

    const prompt = `
You are a legal assistant specializing in constitutional law. 
Answer the following user query using only the provided constitutional context. 
If the answer is not explicitly supported by the context, politely say so. 
Cite the relevant Article(s) or Section(s) if available.

User Query: "${query}"

Context:
${context}
    `;

    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [
            { role: "system", content: "You are an AI legal assistant." },
            { role: "user", content: prompt },
          ],
          max_tokens: 500,
          temperature: 0.3,
        }),
      }
    );

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`HTTP error! ${response.status} - ${errText}`);
    }

    const result = await response.json();
    const aiResponse = result.result.response.trim();

    console.log("AI Response Generated");
    return aiResponse;

  } catch (error) {
    console.error("Error generating AI response:", error);
    return "Sorry, I could not generate a response at this time.";
  } finally {
    console.timeEnd("AI Response Generation");
  }
}

// API Endpoints
// Process and store constitution text with memory optimization
app.post("/process", async (req, res) => {
  console.time("Total /process Request");
  
  const progressTracker = new ProgressTracker(res);
  progressTracker.startStreaming();
  
  try {
    const { constitutionText, strategy = "ai" } = req.body;
    if (!constitutionText) {
      progressTracker.error(new Error("constitutionText is required"));
      return;
    }

    console.log(`Received constitution text (${constitutionText.length} characters)`);
    console.log(`Using chunking strategy: ${strategy}`);

    progressTracker.updateProgress(5, 'Processing constitution text...');

    // Chunking with memory optimization
    const structuredChunks = await aiBasedChunking(constitutionText, strategy, progressTracker);
    console.log(`Chunking successful: ${structuredChunks.length} chunks created using ${strategy} strategy`);

    const filteredChunks = structuredChunks.filter(chunk => chunk.text && chunk.text.trim().length > 50);

    progressTracker.updateProgress(45, 'Generating embeddings for chunks...');
    console.log(`Processing embeddings for ${filteredChunks.length} filtered chunks...`);

    // MEMORY OPTIMIZATION: Use batch processing for embeddings
    const { embeddingsArray, failedEmbeddings } = await processEmbeddingsInBatches(filteredChunks, progressTracker);

    const successfulEmbeddingCount = embeddingsArray.filter(e => e !== null).length;
    const failedEmbeddingCount = failedEmbeddings.length;
    console.log(`Embedding results: ${successfulEmbeddingCount} successful, ${failedEmbeddingCount} failed`);

    if (successfulEmbeddingCount === 0) {
      throw new Error("All chunks failed to generate embeddings. No data stored.");
    }

    const finalStructuredChunksToStore = filteredChunks.filter((_, i) => embeddingsArray[i] !== null);
    const finalEmbeddingsToStore = embeddingsArray.filter(e => e !== null);

    progressTracker.updateProgress(80, 'Storing chunks in database...');
    console.log("Storing successfully embedded chunks in database...");
    await storeChunksInDatabase(finalStructuredChunksToStore, finalEmbeddingsToStore, progressTracker);

    const response = {
      success: true,
      strategy: strategy,
      totalChunksAttempted: structuredChunks.length,
      processedChunksForEmbedding: successfulEmbeddingCount,
      failedChunksForEmbedding: failedEmbeddingCount,
      storedChunks: finalStructuredChunksToStore.length,
      successRate: Math.round((successfulEmbeddingCount / structuredChunks.length) * 100),
      message: `Constitution text processed successfully using ${strategy}. ${finalStructuredChunksToStore.length} chunks stored.`
    };

    if (failedEmbeddingCount > 0) {
      response.failedChunksDetails = failedEmbeddings;
      response.warning = `${failedEmbeddingCount} chunks failed to generate embeddings. They were not stored.`;
    }

    progressTracker.updateProgress(100, 'Processing complete!');
    progressTracker.complete(response);

  } catch (error) {
    console.error("Error in /process endpoint:", error);
    progressTracker.error(error);
  } finally {
    console.timeEnd("Total /process Request");
    // Final memory cleanup
    if (global.gc) {
      global.gc();
    }
  }
});

// Query constitution for legal guidance with memory optimization
app.post("/query", async (req, res) => {
  console.time("Total /query Request");
  try {
    const { query } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Query is required" });
    }

    console.log("Processing original query:", query);

    // Expand the user's query for better context
    const expandedQuery = await expandQueryWithAI(query);

    // Use the expanded query for embeddings
    const queryEmbedding = await getEmbeddings(expandedQuery);

    // MEMORY OPTIMIZATION: Use paginated search instead of loading all chunks
    const relevantSections = await searchRelevantSections(queryEmbedding);

    if (relevantSections.length === 0) {
      const noResultsMessage = "I could not find relevant sections in the Constitution to answer your query. Please rephrase or provide more details.";
      await db.execute({
        sql: `INSERT INTO user_queries (query, response, relevant_chunks) VALUES (?, ?, ?)`,
        args: [query, noResultsMessage, "[]"]
      });
      return res.status(404).json({
        success: false,
        query,
        response: noResultsMessage,
        relevantSections: []
      });
    }

    // Pass the ORIGINAL query to the response generator
    const aiResponse = await generateResponse(query, relevantSections);

    console.time("Storing Query in History");
    await db.execute({
      sql: `INSERT INTO user_queries (query, response, relevant_chunks) VALUES (?, ?, ?)`,
      args: [query, aiResponse, JSON.stringify(relevantSections.map(s => ({
        id: s.id,
        title: s.article_section,
        type: s.chunk_type,
        similarity: s.similarity
      })))]
    });
    console.timeEnd("Storing Query in History");

    res.json({
      success: true,
      query,
      expandedQueryForSearch: expandedQuery,
      response: aiResponse,
      relevantSections: relevantSections.map(section => ({
        text: section.chunk_text,
        similarity: parseFloat(section.similarity.toFixed(4)),
        article_section: section.article_section,
        chunk_type: section.chunk_type
      }))
    });

  } catch (error) {
    console.error("Error in /query endpoint:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message
    });
  } finally {
    console.timeEnd("Total /query Request");
    // Memory cleanup
    if (global.gc) {
      global.gc();
    }
  }
});

// Get query history
app.get("/history", async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    const result = await db.execute({
      sql: "SELECT * FROM user_queries ORDER BY created_at DESC LIMIT ?",
      args: [limit]
    });
    res.json({
      success: true,
      queries: result.rows,
      count: result.rows.length
    });
  } catch (error) {
    console.error("Error in /history endpoint:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message
    });
  }
});

// Get constitution stats
app.get("/stats", async (req, res) => {
  try {
    const chunksResult = await db.execute("SELECT COUNT(*) as total_chunks FROM constitution_chunks");
    const queriesResult = await db.execute("SELECT COUNT(*) as total_queries FROM user_queries");

    res.json({
      success: true,
      totalChunks: chunksResult.rows[0].total_chunks,
      totalQueries: queriesResult.rows[0].total_queries,
      serverUptime: process.uptime(),
      memoryUsage: process.memoryUsage()
    });
  } catch (error) {
    console.error("Error in /stats endpoint:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message
    });
  }
});

// Clear all data (for testing)
app.delete("/clear", async (req, res) => {
  try {
    await db.execute("DELETE FROM constitution_chunks");
    await db.execute("DELETE FROM user_queries");

    // Force garbage collection after clearing data
    if (global.gc) {
      global.gc();
    }

    res.json({
      success: true,
      message: "All data cleared successfully"
    });
  } catch (error) {
    console.error("Error in /clear endpoint:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message
    });
  }
});

// Default route
app.get("/", (req, res) => {
  res.json({
    message: "Constitution Legal Assistant API is running!",
    endpoints: {
      "POST /process": "Process and store constitution text. Body can include 'strategy': 'ai' (default), 'logical', or 'langchain'.",
      "POST /query": "Query constitutional provisions",
      "GET /history": "Get query history",
      "GET /stats": "Get system statistics",
      "GET /health": "Health check",
      "DELETE /clear": "Clear all data"
    }
  });
});

// Health check route with memory info
app.get("/health", (req, res) => {
  const memUsage = process.memoryUsage();
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memoryUsage: {
      rss: Math.round(memUsage.rss / 1024 / 1024) + ' MB',
      heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024) + ' MB',
      heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + ' MB',
      external: Math.round(memUsage.external / 1024 / 1024) + ' MB'
    }
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error("Unhandled error:", error);

  if (error.type === 'entity.too.large') {
    return res.status(413).json({
      error: 'Payload too large',
      message: 'The request body exceeds the maximum allowed size'
    });
  }

  res.status(500).json({
    error: 'Internal server error',
    details: error.message
  });
});

// Periodic garbage collection (every 5 minutes)
if (global.gc) {
  setInterval(() => {
    console.log('Running periodic garbage collection...');
    global.gc();
    const memUsage = process.memoryUsage();
    console.log(`Memory after GC - Heap Used: ${Math.round(memUsage.heapUsed / 1024 / 1024)} MB`);
  }, 5 * 60 * 1000); // 5 minutes
}

// Initialize database and start server
async function startServer() {
  try {
    await initializeDatabase();
    app.listen(PORT, () => {
      console.log(`Server running at http://localhost:${PORT}`);
      console.log(`Constitution API ready for legal queries`);
      console.log(`Default chunking strategy: 'ai' (Cloudflare)`);
      console.log(`Max payload size: 50MB`);
      console.log(`Node.js max heap size: ${process.env.NODE_OPTIONS || 'default'}`);
      
      if (global.gc) {
        console.log('Garbage collection is available and will run periodically');
      } else {
        console.log('Warning: Garbage collection not available. Start with --expose-gc for better memory management');
      }
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

startServer().catch(console.error);