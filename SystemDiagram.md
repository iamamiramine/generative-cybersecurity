%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#F7FAFC',
      'primaryTextColor': '#2D3748',
      'primaryBorderColor': '#4A5568',
      'lineColor': '#2D3748',  
      'edgeLabelBackground': '#FFFFFF',
      'textColor': '#2D3748',  
      'secondaryColor': '#F7FAFC',
      'tertiaryColor': '#E2E8F0'
    }
  }
}%%

graph TD
    %% Styling
    classDef systemArchStyle fill:#F7FAFC,stroke:#4A5568,color:#2D3748
    classDef sharedServicesStyle fill:#EDF2F7,stroke:#4A5568,color:#2D3748
    classDef documentProcessingStyle fill:#E2E8F0,stroke:#4A5568,color:#2D3748
    classDef vectorizationStyle fill:#EDF2F7,stroke:#4A5568,color:#2D3748
    classDef memorySystemStyle fill:#E2E8F0,stroke:#4A5568,color:#2D3748
    classDef chainProcessingStyle fill:#EDF2F7,stroke:#4A5568,color:#2D3748
    classDef chainSelectionStyle fill:#E2E8F0,stroke:#4A5568,color:#2D3748
    classDef basicChainStyle fill:#EDF2F7,stroke:#4A5568,color:#2D3748
    classDef ragChainStyle fill:#E2E8F0,stroke:#4A5568,color:#2D3748
    classDef hydeChainStyle fill:#EDF2F7,stroke:#4A5568,color:#2D3748
    classDef outputProcessingStyle fill:#E2E8F0,stroke:#4A5568,color:#2D3748
    
    linkStyle default stroke:#2D3748,color:#2D3748

    subgraph System_Architecture[High-Level System Architecture]
        A[Streamlit UI] -->|User Input| B[StreamlitChat]
        B -->|Calls| C[LangchainController]
        C -->|Manages| D[LangchainService]
    end

    subgraph Shared_Services[LangchainService Shared Services]
        S1[load_model]
        S2[load_pipeline]
        S3[load_docs]
        S4[load_ensemble_retriever_from_docs]
        S5[load_chain]
        S6[generate]

        subgraph Document_Processing[Document Processing]
            LoadDocs[Load Text Files] -->|Raw Docs| CheckDataset[Check Dataset Changes]
            CheckDataset -->|If Changed| SplitDocs[Split Documents]
            CheckDataset -->|If Unchanged| LoadChunks[Load Existing Chunks]
            SplitDocs -->|Text Chunks| SaveChunks[Save Text Chunks]
            
            subgraph Vectorization[Vector DB Creation]
                CreateVectorDB[Create Vector DB]
                EmbedDocs[Embed Documents]
                CreateVectorDB -->|Texts| EmbedDocs
                EmbedDocs -->|Vectors| StoreVectors[Store Vectors]
            end

            SaveChunks --> CreateVectorDB
            LoadChunks --> CreateVectorDB
        end

        subgraph Memory_System[Memory Chain Creation]
            MemoryWrapper[Memory Chain Wrapper]
            ChatHistory[Chat History]
            ContextualizePrompt[Contextualize Question Prompt]
            
            MemoryWrapper -->|Stores| ChatHistory
            MemoryWrapper -->|Uses| ContextualizePrompt
        end

        subgraph Chain_Processing[Chain Processing Layer]
            subgraph Chain_Selection[Chain Selection]
                ChainSelector[Chain Selector]
                ChainSelector -->|basic| Basic
                ChainSelector -->|rag| RAG
                ChainSelector -->|hyde| HYDE
            end

            subgraph Basic[Basic Chain]
                B1[Basic Request] -->|prepare_basic_prompt| B2[Basic Template]
                B2 -->|format| B3[System + Question]
                B3 -->|LLM| B4[Generate]
            end

            subgraph RAG[RAG Chain]
                R1[RAG Request] -->|prepare_rag_prompt| R2[RAG Template]
                
                subgraph Retrieval[Retrieval Layer]
                    R3[Ensemble Retriever] -->|BM25| R4[Keyword Search]
                    R3 -->|Vector| R5[Semantic Search]
                    R4 -->|combine| R6[Context]
                    R5 -->|combine| R6
                end
                
                R1 --> R3
                R6 -->|format| R7[Context + Question]
                R2 --> R7
                R7 -->|LLM| R8[Generate]
            end

            subgraph HYDE[HyDE Chain]
                H1[HyDE Request] -->|prepare_hyde_prompt| H2[Two-Stage Templates]
                
                subgraph HyDE_Stage1[First Stage]
                    H3[Generation Prompt] -->|LLM| H4[Hypothetical Doc]
                    H4 -->|First Retrieval| H5[Initial Context]
                end
                
                subgraph HyDE_Stage2[Second Stage]
                    H6[Final Prompt] -->|Second Retrieval| H7[Additional Context]
                    H7 -->|combine| H8[Final Context]
                    H5 -->|combine| H8
                end
                
                H2 --> H3
                H8 -->|LLM| H9[Generate]
            end
        end

        %% Internal Shared Services connections
        D -->|Initialize| S1
        S1 -->|Model| S2
        S2 -->|Pipeline| S5
        S3 -->|Load| LoadDocs
        StoreVectors -->|Retriever| S4
        S4 -->|Retriever| S5
        S5 -->|Chain Options| ChainSelector
    end

    subgraph Output_Processing[Output Layer]
        ScriptExecTool[ScriptExecutionTool]
        O1[Generated Response] -->|Extract Scripts| ScriptExecTool
        ScriptExecTool -->|Execute Scripts| O2[Script Execution]
        O1 -->|Store| O3[Memory System]
        O2 -->|Results| O4[Final Response]
        O3 -->|History| O4
    end

    %% External connections
    Basic --> MemoryWrapper
    RAG --> MemoryWrapper
    HYDE --> MemoryWrapper
    MemoryWrapper -->|Wrapped Chain| S6
    
    S6 -->|Generated Output| O1
    O4 -->|Final Results| D
    D -->|Response| C
    C -->|Response| B
    B -->|Display| A

    ChatHistory -->|Previous Context| ContextualizePrompt
    ContextualizePrompt -->|Reformulated Question| MemoryWrapper

    %% Apply styles
    class System_Architecture systemArchStyle
    class Shared_Services sharedServicesStyle
    class Document_Processing documentProcessingStyle
    class Vectorization vectorizationStyle
    class Memory_System memorySystemStyle
    class Chain_Processing chainProcessingStyle
    class Chain_Selection chainSelectionStyle
    class Basic basicChainStyle
    class RAG ragChainStyle
    class HYDE hydeChainStyle
    class Output_Processing outputProcessingStyle