# Multi-modal Actions



## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
This document provides a comprehensive analysis of the multi-modal actions system within the RAVANA repository. The system enables the processing and analysis of non-text content such as images and audio, and supports cross-modal integration for advanced cognitive functions. The architecture leverages external AI services, particularly Google's Gemini, to extract semantic meaning from visual and auditory data. These capabilities are integrated into a broader autonomous agent framework that supports decision-making, knowledge management, and emotional intelligence. The system is designed to handle both individual media files and batch processing of directories, with robust error handling and performance optimization features.

## Project Structure
The multi-modal functionality is distributed across several key modules in the RAVANA project. The core action definitions are located in the `core/actions` directory, while the actual processing logic resides in the `services` module. The enhanced action manager orchestrates the execution flow and integrates multi-modal capabilities with the broader agent system. Additional multi-modal memory functionality is implemented in the episodic memory module, providing persistent storage and retrieval of processed media content.

```mermaid
graph TB
    subgraph "Core Actions"
        A[ProcessImageAction]
        B[ProcessAudioAction]
        C[AnalyzeDirectoryAction]
        D[CrossModalAnalysisAction]
    end
    
    subgraph "Services"
        E[MultiModalService]
        F[process_image]
        G[process_audio]
        H[cross_modal_analysis]
        I[process_directory]
    end
    
    subgraph "Action Management"
        J[EnhancedActionManager]
        K[process_image_action]
        L[process_audio_action]
        M[cross_modal_analysis_action]
    end
    
    subgraph "LLM Integration"
        N[call_gemini_image_caption]
        O[call_gemini_audio_description]
        P[safe_call_llm]
    end
    
    subgraph "Memory System"
        Q[MultiModalMemoryService]
        R[process_image_memory]
        S[search_memories]
    end
    
    A --> J
    B --> J
    C --> J
    D --> J
    J --> E
    E --> N
    E --> O
    N --> P
    O --> P
    E --> Q
    J

**Referenced Files in This Document**   
- [core/actions/multi_modal.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\multi_modal.py)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py)
- [core/enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py)
- [core/llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py)
- [core/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)
- [modules/episodic_memory/memory.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\memory.py)
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py)