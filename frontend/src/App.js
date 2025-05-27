import React, { useState, useEffect } from "react";
import {
  PlusCircle,
  FileText,
  MessageSquareText,
  ChevronLeft,
  Upload,
  Send,
  HelpCircle,
  BookOpen,
} from "lucide-react"; // Using lucide-react for icons
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css"; // Import syntax highlighting style

// Base URL for your FastAPI backend
const API_BASE_URL = "http://localhost:8000"; // Adjust if your backend is on a different port/host

function App() {
  const [selectedRole, setSelectedRole] = useState(null); // 'old_member' or 'new_member'
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null); // Stores project_id and name
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [qaType, setQaType] = useState(null); // 'project_qa' or 'document_qa'
  const [currentView, setCurrentView] = useState("role_selection"); // 'role_selection', 'project_selection', 'qa_selection', 'project_qa', 'document_qa', 'new_member_qa'
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  // State for Project-based Q&A (Old Member)
  const [projectQaSession, setProjectQaSession] = useState(null); // { session_id, project_id, question, question_entry_id, is_complete }
  const [projectQaAnswer, setProjectQaAnswer] = useState("");
  const [projectQaHistory, setProjectQaHistory] = useState([]); // [{ question, answer }]

  // State for Document-based Q&A (Old Member)
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedDocument, setUploadedDocument] = useState(null); // { document_id, file_name }
  const [currentDocQuestion, setCurrentDocQuestion] = useState(null); // { question, question_entry_id, is_complete }
  const [docQaAnswer, setDocQaAnswer] = useState("");
  const [docQaHistory, setDocQaHistory] = useState([]); // [{ question, answer, source_context }]

  // State for New Member Q&A
  const [newMemberQuery, setNewMemberQuery] = useState("");
  const [newMemberChatHistory, setNewMemberChatHistory] = useState([]); // [{ role: 'human' | 'ai', content: string, sources: [] }]

  useEffect(() => {
    if (currentView === "project_selection") {
      fetchProjects();
    }
  }, [currentView]);

  const showMessage = (msg, isError = false) => {
    setErrorMessage(msg);
    setTimeout(() => setErrorMessage(""), 5000); // Clear message after 5 seconds
  };

  const fetchProjects = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setProjects(data);
    } catch (error) {
      console.error("Error fetching projects:", error);
      showMessage(`Failed to fetch projects: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProject = async () => {
    if (!newProjectName) {
      showMessage("Project name cannot be empty.", true);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/projects/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newProjectName,
          description: newProjectDescription,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setSelectedProject({ id: data.id, name: data.name });
      setCurrentView(
        selectedRole === "old_member" ? "qa_selection" : "new_member_qa"
      );
      showMessage(`Project '${data.name}' created successfully!`);
    } catch (error) {
      console.error("Error creating project:", error);
      showMessage(`Failed to create project: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectProject = (project) => {
    setSelectedProject(project);
    setCurrentView(
      selectedRole === "old_member" ? "qa_selection" : "new_member_qa"
    );
    showMessage(`Selected project: ${project.name}`);
  };

  // --- Old Member: Project-based Q&A ---
  const startProjectQaSession = async () => {
    if (!selectedProject) {
      showMessage("Please select a project first.", true);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/transfer/project-qa/start-session/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: selectedProject.id }),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setProjectQaSession(data);
      setProjectQaHistory([{ role: "ai", content: data.question }]); // Start history with AI's first question
      setCurrentView("project_qa");
      showMessage("Project Q&A session started.");
    } catch (error) {
      console.error("Error starting project Q&A session:", error);
      showMessage(`Failed to start project Q&A: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const respondToProjectQa = async () => {
    if (!projectQaAnswer.trim()) {
      showMessage("Please provide an answer.", true);
      return;
    }
    if (!projectQaSession || projectQaSession.is_complete) {
      showMessage("Q&A session is not active or already complete.", true);
      return;
    }
    setLoading(true);
    try {
      // Add user's answer to history
      setProjectQaHistory((prev) => [
        ...prev,
        { role: "human", content: projectQaAnswer },
      ]);

      const response = await fetch(
        `${API_BASE_URL}/transfer/project-qa/respond/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: projectQaSession.session_id,
            project_id: selectedProject.id,
            answer: projectQaAnswer,
          }),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setProjectQaSession(data);
      setProjectQaAnswer(""); // Clear input

      if (data.next_question) {
        setProjectQaHistory((prev) => [
          ...prev,
          { role: "ai", content: data.next_question },
        ]);
      } else {
        showMessage(data.message);
      }
    } catch (error) {
      console.error("Error responding to project Q&A:", error);
      showMessage(`Failed to respond to project Q&A: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  // --- Old Member: Document-based Q&A ---
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUploadDocument = async () => {
    if (!selectedFile || !selectedProject) {
      showMessage("Please select a file and a project.", true);
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${API_BASE_URL}/transfer/document/?project_id=${selectedProject.id}`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setUploadedDocument({ id: data.document_id, name: data.file_name });
      showMessage(
        `Document '${data.file_name}' uploaded successfully! Now generate questions.`
      );
    } catch (error) {
      console.error("Error uploading document:", error);
      showMessage(`Failed to upload document: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateQuestions = async () => {
    if (!uploadedDocument) {
      showMessage("Please upload a document first.", true);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/transfer/document-qa/generate-questions/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: selectedProject.id,
            document_id: uploadedDocument.id,
          }),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      showMessage(data.message);
      // Automatically fetch the first question after generation
      fetchNextDocQuestion();
    } catch (error) {
      console.error("Error generating questions:", error);
      showMessage(`Failed to generate questions: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const fetchNextDocQuestion = async () => {
    if (!selectedProject || !uploadedDocument) {
      showMessage("Project or document not selected.", true);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/transfer/document-qa/get-next-question/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: selectedProject.id,
            document_id: uploadedDocument.id,
          }),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();

      if (data.question) {
        setCurrentDocQuestion({
          question: data.question,
          question_entry_id: data.question_entry_id,
          is_complete: false,
          source_context: data.source_context, // if provided by backend
        });
        showMessage("Next question loaded.");
      } else {
        // No more questions available
        setCurrentDocQuestion({ is_complete: true });
        showMessage(
          data.message || "All questions for this document have been answered."
        );
      }
    } catch (error) {
      console.error("Error fetching next document question:", error);
      showMessage(`Failed to fetch next question: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const answerDocQuestion = async () => {
    if (
      !docQaAnswer.trim() ||
      !currentDocQuestion ||
      currentDocQuestion.is_complete
    ) {
      showMessage("Please provide an answer or no active question.", true);
      return;
    }
    setLoading(true);
    try {
      // Add answered question to history
      setDocQaHistory((prev) => [
        ...prev,
        {
          question: currentDocQuestion.question,
          answer: docQaAnswer,
          source_context: currentDocQuestion.source_context,
        },
      ]);

      const response = await fetch(
        `${API_BASE_URL}/transfer/document-qa/answer-question/`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: selectedProject.id,
            question_entry_id: currentDocQuestion.question_entry_id,
            answer: docQaAnswer,
          }),
        }
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setDocQaAnswer(""); // Clear input

      showMessage(data.message || "Answer recorded successfully.");

      // Always fetch the next question after successfully answering
      await fetchNextDocQuestion();
    } catch (error) {
      console.error("Error answering document question:", error);
      showMessage(`Failed to answer document question: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  // --- New Member: Conversational Q&A ---
  const askNewMemberQuery = async () => {
    if (!newMemberQuery.trim()) {
      showMessage("Please enter a query.", true);
      return;
    }
    if (!selectedProject) {
      showMessage("Please select a project first.", true);
      return;
    }
    setLoading(true);
    try {
      // Add user's query to chat history immediately
      setNewMemberChatHistory((prev) => [
        ...prev,
        { role: "human", content: newMemberQuery },
      ]);

      const chatHistoryForAPI = newMemberChatHistory.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch(`${API_BASE_URL}/retrieve/answer/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: selectedProject.id,
          query: newMemberQuery,
          chat_history: chatHistoryForAPI, // Pass current chat history
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      console.log(data);

      setNewMemberChatHistory((prev) => [
        ...prev,
        { role: "ai", content: data.answer, sources: data.source_documents },
      ]);
      setNewMemberQuery(""); // Clear input
    } catch (error) {
      console.error("Error asking new member query:", error);
      showMessage(`Failed to get answer: ${error.message}`, true);
    } finally {
      setLoading(false);
    }
  };

  const renderRoleSelection = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-6">
          Welcome to Knowledge Relay
        </h2>
        <p className="text-gray-600 mb-8">
          Please select your role to get started:
        </p>
        <div className="flex flex-col space-y-4">
          <button
            onClick={() => {
              setSelectedRole("old_member");
              setCurrentView("project_selection");
            }}
            className="flex items-center justify-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out transform hover:scale-105"
          >
            <BookOpen className="mr-2" size={20} /> Old Member
          </button>
          <button
            onClick={() => {
              setSelectedRole("new_member");
              setCurrentView("project_selection");
            }}
            className="flex items-center justify-center px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 transition duration-300 ease-in-out transform hover:scale-105"
          >
            <HelpCircle className="mr-2" size={20} /> New Member
          </button>
        </div>
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm">{errorMessage}</p>
        )}
      </div>
    </div>
  );

  const renderProjectSelection = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
        <button
          onClick={() => setCurrentView("role_selection")}
          className="flex items-center text-gray-600 hover:text-gray-800 mb-6 transition duration-200"
        >
          <ChevronLeft size={20} className="mr-1" /> Back to Role Selection
        </button>
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Select or Create Project
        </h2>

        {/* Create Project Section */}
        <div className="mb-8 p-6 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <PlusCircle size={20} className="mr-2" /> Create New Project
          </h3>
          <input
            type="text"
            placeholder="Project Name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            className="w-full p-3 mb-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={handleCreateProject}
            className="w-full px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition duration-300 ease-in-out"
            disabled={loading}
          >
            {loading ? "Creating..." : "Create Project"}
          </button>
        </div>

        {/* Select Project Section */}
        <div className="p-6 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <FileText size={20} className="mr-2" /> Select Existing Project
          </h3>
          {loading ? (
            <p className="text-gray-500 text-center">Loading projects...</p>
          ) : projects.length === 0 ? (
            <p className="text-gray-500 text-center">
              No projects available. Create one above!
            </p>
          ) : (
            <select
              onChange={(e) => {
                const project = projects.find((p) => p.id === e.target.value);
                if (project) {
                  handleSelectProject(project);
                }
              }}
              className="w-full p-3 border border-gray-300 rounded-lg bg-white focus:ring-blue-500 focus:border-blue-500"
              defaultValue=""
            >
              <option value="" disabled>
                Select a project
              </option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          )}
        </div>
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm text-center">
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );

  const renderQaSelection = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center">
        <button
          onClick={() => setCurrentView("project_selection")}
          className="flex items-center text-gray-600 hover:text-gray-800 mb-6 transition duration-200"
        >
          <ChevronLeft size={20} className="mr-1" /> Back to Project Selection
        </button>
        <h2 className="text-3xl font-bold text-gray-800 mb-6">
          {selectedProject
            ? `Q&A for "${selectedProject.name}"`
            : "Select Q&A Type"}
        </h2>
        <p className="text-gray-600 mb-8">
          Choose how you want to contribute knowledge:
        </p>
        <div className="flex flex-col space-y-4">
          <button
            onClick={startProjectQaSession}
            className="flex items-center justify-center px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 transition duration-300 ease-in-out transform hover:scale-105"
            disabled={loading}
          >
            <MessageSquareText className="mr-2" size={20} /> Project-based Q&A
          </button>
          <button
            onClick={() => {
              setQaType("document_qa");
              setCurrentView("document_qa");
            }}
            className="flex items-center justify-center px-6 py-3 bg-orange-600 text-white font-semibold rounded-lg shadow-md hover:bg-orange-700 transition duration-300 ease-in-out transform hover:scale-105"
            disabled={loading}
          >
            <FileText className="mr-2" size={20} /> Document-based Q&A
          </button>
        </div>
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm">{errorMessage}</p>
        )}
      </div>
    </div>
  );

  const renderProjectQa = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl flex flex-col h-[80vh]">
        <button
          onClick={() => {
            setCurrentView("qa_selection");
            setProjectQaSession(null);
            setProjectQaAnswer("");
            setProjectQaHistory([]);
          }}
          className="flex items-center text-gray-600 hover:text-gray-800 mb-6 transition duration-200"
        >
          <ChevronLeft size={20} className="mr-1" /> Back to Q&A Type Selection
        </button>
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Project Q&A for "{selectedProject?.name}"
        </h2>

        <div className="flex-1 overflow-y-auto p-4 border border-gray-200 rounded-lg mb-4 bg-gray-50 flex flex-col space-y-4">
          {projectQaHistory.length === 0 && !loading && (
            <p className="text-gray-500 text-center py-8">
              Start by answering the first question...
            </p>
          )}
          {projectQaHistory.map((msg, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg shadow-sm ${
                msg.role === "human"
                  ? "bg-blue-100 self-end text-right"
                  : "bg-gray-200 self-start text-left"
              }`}
              style={{ maxWidth: "80%" }}
            >
              <p className="font-semibold text-sm mb-1">
                {msg.role === "human" ? "You" : "AI"}
              </p>
              <p className="text-gray-800">{msg.content}</p>
            </div>
          ))}
          {loading && (
            <div className="text-center text-gray-500">
              Loading next question...
            </div>
          )}
        </div>

        {projectQaSession && !projectQaSession.is_complete ? (
          <div className="flex items-center space-x-3">
            <input
              type="text"
              placeholder="Type your answer here..."
              value={projectQaAnswer}
              onChange={(e) => setProjectQaAnswer(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === "Enter") respondToProjectQa();
              }}
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
              disabled={loading}
            />
            <button
              onClick={respondToProjectQa}
              className="p-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition duration-300"
              disabled={loading}
            >
              <Send size={20} />
            </button>
          </div>
        ) : (
          <p className="text-center text-gray-600 font-semibold mt-4">
            {projectQaSession?.message ||
              "Session completed or no questions available."}
          </p>
        )}
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm text-center">
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );

  const renderDocumentQa = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl">
        <button
          onClick={() => {
            setCurrentView("qa_selection");
            setSelectedFile(null);
            setUploadedDocument(null);
            setCurrentDocQuestion(null);
            setDocQaAnswer("");
            setDocQaHistory([]);
          }}
          className="flex items-center text-gray-600 hover:text-gray-800 mb-6 transition duration-200"
        >
          <ChevronLeft size={20} className="mr-1" /> Back to Q&A Type Selection
        </button>
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Document Q&A for "{selectedProject?.name}"
        </h2>

        {/* Document Upload Section */}
        <div className="mb-8 p-6 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <Upload size={20} className="mr-2" /> Upload Document
          </h3>
          <input
            type="file"
            onChange={handleFileChange}
            className="w-full p-3 mb-4 border border-gray-300 rounded-lg bg-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          <button
            onClick={handleUploadDocument}
            className="w-full px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300"
            disabled={loading || !selectedFile}
          >
            {loading && selectedFile ? "Uploading..." : "Upload Document"}
          </button>
          {uploadedDocument && (
            <p className="text-green-600 mt-3 text-center">
              Uploaded: {uploadedDocument.name}
            </p>
          )}
        </div>

        {/* Generate Questions Section */}
        {uploadedDocument && (
          <div className="mb-8 p-6 border border-gray-200 rounded-lg bg-gray-50">
            <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
              <PlusCircle size={20} className="mr-2" /> Generate Questions
            </h3>
            <p className="text-gray-600 mb-4">
              Click below to generate questions from the uploaded document.
              These will then be available for you to answer.
            </p>
            <button
              onClick={handleGenerateQuestions}
              className="w-full px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 transition duration-300"
              disabled={loading}
            >
              {loading ? "Generating..." : "Generate Questions"}
            </button>
          </div>
        )}

        {/* Document Q&A Section */}
        {currentDocQuestion && !currentDocQuestion.is_complete && (
          <div className="p-6 border border-gray-200 rounded-lg bg-white shadow-md">
            <h3 className="text-xl font-semibold text-gray-700 mb-4">
              Question:
            </h3>
            <div className="bg-gray-100 p-4 rounded-lg mb-4 text-gray-800">
              <p>{currentDocQuestion.question}</p>
            </div>
            <textarea
              placeholder="Type your answer here..."
              value={docQaAnswer}
              onChange={(e) => setDocQaAnswer(e.target.value)}
              rows="4"
              className="w-full p-3 mb-4 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 resize-y"
              disabled={loading}
            ></textarea>
            <button
              onClick={answerDocQuestion}
              className="w-full px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 transition duration-300"
              disabled={loading}
            >
              {loading ? "Submitting..." : "Submit Answer & Get Next Question"}
            </button>
          </div>
        )}

        {currentDocQuestion && currentDocQuestion.is_complete && (
          <p className="text-center text-green-600 font-semibold mt-4">
            All questions for this document have been answered!
          </p>
        )}
        {!currentDocQuestion && uploadedDocument && !loading && (
          <button
            onClick={fetchNextDocQuestion}
            className="w-full px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition duration-300 mt-4"
            disabled={loading}
          >
            {loading ? "Loading..." : "Load First Unanswered Question"}
          </button>
        )}
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm text-center">
            {errorMessage}
          </p>
        )}

        {/* Display Document QA History */}
        {docQaHistory.length > 0 && (
          <div className="mt-8 p-6 border border-gray-200 rounded-lg bg-gray-50">
            <h3 className="text-xl font-semibold text-gray-700 mb-4">
              Answered Questions:
            </h3>
            <div className="flex flex-col space-y-4 max-h-60 overflow-y-auto">
              {docQaHistory.map((entry, index) => (
                <div key={index} className="p-3 rounded-lg bg-white shadow-sm">
                  <p className="font-semibold text-blue-700">
                    Q: {entry.question}
                  </p>
                  <p className="text-gray-800">A: {entry.answer}</p>
                  {entry.source_context && (
                    <p className="text-xs text-gray-500 mt-1">
                      Source: {entry.source_context.substring(0, 100)}...
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderNewMemberQa = () => (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl flex flex-col h-[80vh]">
        <button
          onClick={() => {
            setCurrentView("project_selection");
            setNewMemberChatHistory([]);
            setNewMemberQuery("");
          }}
          className="flex items-center text-gray-600 hover:text-gray-800 mb-6 transition duration-200"
        >
          <ChevronLeft size={20} className="mr-1" /> Back to Project Selection
        </button>
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Chat with "{selectedProject?.name}" Knowledge Base
        </h2>

        <div className="flex-1 overflow-y-auto p-4 border border-gray-200 rounded-lg mb-4 bg-gray-50 flex flex-col space-y-4">
          {newMemberChatHistory.length === 0 && !loading && (
            <p className="text-gray-500 text-center py-8">
              Ask a question to get started...
            </p>
          )}
          {newMemberChatHistory.map((msg, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg shadow-sm ${
                msg.role === "human"
                  ? "bg-blue-100 self-end text-right"
                  : "bg-gray-200 self-start text-left"
              }`}
              style={{ maxWidth: "80%" }}
            >
              <p className="font-semibold text-sm mb-1">
                {msg.role === "human" ? "You" : "AI"}
              </p>

              <div className="text-gray-800 text-left whitespace-pre-wrap">
                <ReactMarkdown
                  children={msg.content}
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                />
              </div>

              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-xs text-gray-600 border-t border-gray-300 pt-2">
                  <p className="font-semibold mb-1">Sources:</p>
                  {msg.sources.map((source, sIdx) => {
                    const filePathEncoded = encodeURIComponent(
                      source.file_path
                    );
                    const fileNameEncoded = encodeURIComponent(
                      source.file_name
                    );
                    const downloadUrl = `http://localhost:8000/download?file_path=${filePathEncoded}&filename=${fileNameEncoded}`;

                    return (
                      <div key={sIdx} className="mb-2">
                        <a
                          href={downloadUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline font-medium"
                        >
                          {source.file_name || "Unknown Source"}
                        </a>
                        {source.question && (
                          <p className="ml-2 mt-1">Q: {source.question}</p>
                        )}
                        {source.context && (
                          <p className="ml-2 italic text-gray-700">
                            Context: {source.context.substring(0, 150)}...
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="text-center text-gray-500">Thinking...</div>
          )}
        </div>

        <div className="flex items-center space-x-3">
          <input
            type="text"
            placeholder="Ask a question..."
            value={newMemberQuery}
            onChange={(e) => setNewMemberQuery(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter") askNewMemberQuery();
            }}
            className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
            disabled={loading}
          />
          <button
            onClick={askNewMemberQuery}
            className="p-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition duration-300"
            disabled={loading}
          >
            <Send size={20} />
          </button>
        </div>
        {errorMessage && (
          <p className="text-red-500 mt-4 text-sm text-center">
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );

  const renderContent = () => {
    switch (currentView) {
      case "role_selection":
        return renderRoleSelection();
      case "project_selection":
        return renderProjectSelection();
      case "qa_selection":
        return renderQaSelection();
      case "project_qa":
        return renderProjectQa();
      case "document_qa":
        return renderDocumentQa();
      case "new_member_qa":
        return renderNewMemberQa();
      default:
        return renderRoleSelection();
    }
  };

  return (
    <div className="min-h-screen font-sans antialiased text-gray-900">
      {renderContent()}
    </div>
  );
}

export default App;
