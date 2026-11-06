# cleAIra — AI-Powered Legal Document Review

cleAIra is a web-based legal document review application designed to help users analyze legal documents, identify potentially concerning clauses, and understand important contractual terms. It provides clause-by-clause feedback with priority levels to help users identify provisions that may require closer attention.

## Live Demo

**Application:** https://cleaira.vercel.app/

**GitHub Repository:** https://github.com/DhritiNarayan/cleAIra

## Features

### AI-Assisted Legal Document Review
- Analyze legal documents through a structured review workflow.
- Examine documents clause by clause.
- Identify potentially concerning contractual provisions.
- Organize findings according to priority levels.

### Priority-Based Findings
- **Critical:** Potentially serious concerns requiring careful attention.
- **High:** Important issues that may warrant further review.
- **Medium:** Provisions that may require clarification or closer examination.

### User Authentication
- User registration and login.
- Secure password hashing.
- Email verification workflow.
- Session-based authentication.

### Review History
- Access previously completed document reviews.
- View individual review details.
- Manage review history, including deleting selected entries.

### Account Management
- Dedicated account page.
- Authenticated access to account features.
- Logout confirmation to prevent accidental sign-outs.

### Web-Based Interface
- Dynamic HTML rendering using Jinja2.
- Green-themed user interface.
- Navigation between document review and review history.
- Client-side interactions using JavaScript.

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Jinja2 Templates

### Backend
- Python
- Flask

### Database
- Neon PostgreSQL
- psycopg (PostgreSQL database adapter)

### Authentication and Security
- Flask sessions
- Werkzeug password hashing
- Email verification
- Environment-based configuration

### Email Services
- SMTP for verification emails

### Deployment and Version Control
- Vercel
- Git
- GitHub

## Application Flow

### Authentication flow

1. A visitor registers with a username, email address, and password.
2. The application validates the input, hashes the password with Werkzeug, and creates
   an unverified user in PostgreSQL.
3. cleAIra sends an email verification link over SMTP.
4. The user verifies the email address and then signs in with the username or email.
5. Flask stores the authenticated user ID in the session. Protected routes redirect
   unauthenticated visitors to the login page.

### Document review flow

1. An authenticated user opens the review page and either pastes contract text or
   uploads a PDF, PNG, JPG, or JPEG file. Uploads are limited to 10 MB.
2. PDF text is extracted with PyMuPDF. If the extracted text is too short, the
   application falls back to OCR; image uploads go directly through OCR using
   Tesseract.
3. The text is cleaned, split into document units, and associated with nearby
   section headings.
4. The rule engine matches each suitable sentence against the clause definitions in
   `clauses.py`, extracts the primary provision, removes duplicate findings, and
   assigns a risk and review priority.
5. Gemini scans the document in bounded chunks for additional evidence-backed risks
   that the rule engine did not identify. Invalid or unavailable AI responses do not
   replace the rule-based findings.
6. Results are sorted by urgency, summarized by contract type, and enriched with
   obligations, financial terms, time terms, key dates, and potentially missing
   expected provisions.
7. The completed review is rendered in the browser and saved to PostgreSQL for the
   authenticated user. Uploaded files are deleted after extraction.

### History and account flow

- The History page lists the signed-in user's saved reviews in reverse chronological
  order.
- A history detail page reconstructs the saved findings and overall analysis from
  the stored JSON payload.
- Users can delete one review or selected reviews; all review queries are scoped to
  the current user.
- The Account page supports account-specific settings, including password changes,
  and logout clears the Flask session.

## Architecture

cleAIra uses a server-rendered Flask architecture with a small service layer:

```text
Browser
  │
  ├── Jinja2 pages + static JavaScript/CSS
  │       ├── login/register/email verification
  │       ├── document review form
  │       └── history/account views
  │
  ▼
Flask application (app.py)
  │
  ├── Authentication and sessions ── auth.py
  │       └── users table
  │
  ├── Input processing
  │       ├── pasted text
  │       ├── PyMuPDF PDF extraction
  │       └── Tesseract OCR for PDFs/images
  │
  ├── Rule-based review engine
  │       ├── clauses.py definitions
  │       └── sentence, heading, priority, and summary logic
  │
  ├── AI augmentation ── legal_ai_service.py
  │       └── gemini_llm_service.py ── Gemini API
  │
  ├── Review persistence ── history.py
  │       └── reviews table (JSON findings and summaries)
  │
  └── SMTP email verification

External services: Neon PostgreSQL, Gemini API, SMTP provider
```

The rule engine remains the primary source of detected clauses. Gemini is an
augmentation layer that proposes additional risks, and its output is validated
before it is shown or persisted. This keeps the application useful when the AI
service is unavailable and makes the source text for every finding visible to the
user.

## Project Structure

```text
cleAIra/
├── app.py
├── auth.py
├── clauses.py
├── gemini_llm_service.py
├── legal_ai_service.py
├── history.py
├── utils.py
├── appbackup.py
├── history_backup_v4.py
├── templates/
│   ├── index.html
│   ├── history.html
│   ├── history_detail.html
│   ├── account.html
│   ├── login.html
│   ├── register.html
│   └── verify_email.html
├── static/
│   ├── script.js
│   └── style.css
├── requirements.txt
├── test_clause_engine.py
├── test_contract_ai.py
├── .gitignore
└── README.md
```

`appbackup.py` and `history_backup_v4.py` are retained backup modules and are not
part of the primary request path. A local `.env` file is intentionally omitted
from the tree above because it is ignored by Git and must never be committed; use
a local `.env.example` or deployment secret manager for shared configuration
documentation.

## Installation and Setup

Follow these steps to run cleAIra locally.

### Prerequisites

Make sure the following are installed:

- Python 3.10 or a compatible version supported by your dependencies
- pip
- Git

You will also need a Neon PostgreSQL database and the required email service configuration if you want to use the complete authentication workflow.

### 1. Clone the Repository

```bash
git clone https://github.com/DhritiNarayan/cleAIra.git
cd cleAIra
```

### 2. Create a Virtual Environment

Create the environment:

```bash
python -m venv venv
```

Activate it on Windows using PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Alternatively, use Command Prompt:

```bat
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Configure the environment variables required by your application:

```env
CLEAIRA_SECRET_KEY=your-secret-key
CLEAIRA_BASE_URL=http://127.0.0.1:5000
DATABASE_URL=your-neon-postgresql-connection-string
GEMINI_API_KEY=your-gemini-api-key
# Optional; defaults to the model configured in gemini_llm_service.py
GEMINI_MODEL=your-gemini-model

CLEAIRA_EMAIL_HOST=smtp.gmail.com
CLEAIRA_EMAIL_PORT=465
CLEAIRA_EMAIL_USERNAME=your-smtp-username
CLEAIRA_EMAIL_PASSWORD=your-smtp-password
```

The `DATABASE_URL` should contain the connection string provided by Neon. Use the exact environment variable names expected by your application.

**Important:** Never commit real database credentials, email passwords, API keys, or production secret keys to GitHub.

### 5. Start the Application

If `app.py` contains your Flask application's entry point, run:

```bash
python app.py
```

Open the local address shown in your terminal, typically:

```text
http://127.0.0.1:5000
```

The exact startup command depends on how your Flask application is configured.

## Database and Data Persistence

cleAIra uses **Neon PostgreSQL** as its database for persistent application data, such as user accounts and review history.

Neon provides a managed PostgreSQL database that can be accessed by the Flask backend through a PostgreSQL database adapter.

The application uses a database connection configured through environment variables, allowing local development and production deployment to use different database configurations.

To run the application successfully, ensure that:

- The Neon database is available.
- The correct PostgreSQL connection string is configured.
- Required database tables are initialized.
- The application has the necessary database permissions.
- Database credentials are not exposed in source control.

## Security Considerations

Security is important when handling user accounts and legal documents.

- Passwords should be hashed before being stored.
- Authentication should protect routes that require a logged-in user.
- Secret keys and database credentials should be stored in environment variables.
- Email verification should validate the registration workflow.
- User input should be validated before processing.
- Database queries should use parameterized statements.
- Sensitive files and local configuration files should be excluded from version control.
- Production deployments should use secure database connections.

## Limitations

- AI-generated legal feedback should be treated as informational assistance rather than professional legal advice.
- The analysis may overlook important contractual details or misinterpret specific clauses.
- Results depend on the document-processing logic and the underlying analysis system.
- Users should independently verify findings before relying on them for important decisions.

## Future Improvements

Potential improvements include:

- Support for additional document formats, including PDF and DOCX.
- More detailed explanations for flagged clauses.
- Downloadable review reports.
- Advanced search and filtering for review history.
- Document comparison capabilities.
- Improved automated testing for authentication and document processing.
- Enhanced validation and security controls.
- Better handling of large documents and processing errors.
- Improved feedback on the reliability and limitations of generated findings.

## Disclaimer

cleAIra is intended to assist users in understanding legal documents. It does not replace a qualified lawyer and does not constitute professional legal advice.

AI-assisted analysis may produce incomplete or inaccurate results. Users should independently verify findings and consult a qualified legal professional before making important legal or contractual decisions.

## Author

**GitHub:** https://github.com/DhritiNarayan

## Project Links

- **Live Application:** https://cleaira.vercel.app/
- **Source Code:** https://github.com/DhritiNarayan/cleAIra
 
 
