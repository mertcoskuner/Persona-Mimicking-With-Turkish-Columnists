# Turkish Columnist Persona System

An AI-powered chat system that mimics the writing styles of Turkish columnists.

## Features

- 🤖 AI models that mimic Turkish columnists' writing styles
- 💬 Natural language processing for realistic chat experience
- 🔒 Secure user authentication
- 📱 Modern and user-friendly interface
- 📊 Detailed performance analytics

## Supported Columnists

- Barış Terkoğlu
- Ahmet Hakan
- Abdulkadir Selvi

## Technical Details

### Architecture

```
.
├── backend/           # FastAPI-based backend
├── frontend/         # Streamlit-based frontend
├── models/           # Trained model files
├── src/             # Source code
├── tests/           # Test scenarios
└── scripts/         # Helper scripts
```

### Technology Stack

- **Backend**: FastAPI, SQLAlchemy, JWT
- **Frontend**: Streamlit, CSS
- **ML/LLM**: Turkish-Llama-8b, BERT-Turkish
- **Database**: PostgreSQL
- **Cache**: Redis
- **Vector DB**: Qdrant

## Installation

For detailed installation steps, see [SETUP.md](SETUP.md).

```bash
# Install required packages
pip install -r requirements.txt

# Start backend
cd backend
python manage.py runserver

# Start frontend
cd frontend
streamlit run app.py
```

## Usage

1. Log in to the system
2. Select your desired columnist
3. Start chatting

## API Documentation

For API endpoints and usage, see [API Documentation](docs/api/README.md).

## Development

### Code Standards

- PEP 8 compliant Python code
- Type hints usage
- Docstring format
- Test coverage target: 80%

### Testing

```bash
# Run all tests
pytest

# Run specific test groups
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Deployment

For detailed deployment steps, see [Deployment Guide](docs/deployment/README.md).

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request




