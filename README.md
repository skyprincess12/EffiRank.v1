# TLS Cost Input & Ranking System

A Streamlit application for managing and ranking Transportation, Loading, and Storage (TLS) costs across different unloading points.

## Features

- **Cost Input Management**: Input and manage cost breakdowns for multiple locations
- **Efficiency Ranking**: Automatically rank locations based on KPI scores
- **Cost Analysis**: Visual analysis with charts and graphs
- **History Tracking**: Save and view historical snapshots
- **Multi-regional Support**: Separate tracking for North and South regions

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd tls-cost-ranking-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create your secrets file:
```bash
mkdir .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

4. Edit `.streamlit/secrets.toml` with your credentials:
```toml
USERNAME = "your_username"
PASSWORD = "your_password"
HISTORY_DELETE_PASSCODE = "your_delete_code"

# Optional: Supabase database (leave empty for local storage only)
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"
```

## Usage

Run the application:
```bash
streamlit run app6.py
```

Then navigate to `http://localhost:8501` in your browser.

## Database Setup (Optional)

If using Supabase, create a table with this structure:

```sql
CREATE TABLE history_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TEXT,
    date TEXT,
    week_number INTEGER,
    week_range TEXT,
    rankings_json JSONB,
    analysis_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Application Structure

- **Cost Input**: Enter cost data for each location
- **Efficiency Ranking**: View ranked locations by efficiency
- **Cost Analysis**: Visual charts and analysis
- **History**: View and manage saved snapshots
- **Calculation Tester**: Debug and verify calculations

## Key Metrics

- **Total Cost**: Sum of all operational costs
- **Cost per LKG**: Total cost divided by LKGTC
- **LKG per PHP**: Efficiency metric (LKGTC ÷ Cost per LKG)
- **KPI Score**: Normalized efficiency score for ranking

## Support

For issues or questions, please check the application logs or contact your system administrator.