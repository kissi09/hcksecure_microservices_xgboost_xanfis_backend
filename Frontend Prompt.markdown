# Frontend Development Prompt for Lovable or Bolt.new

Develop a web application for an Intrusion Detection System (IDS) with the following features:

- **Real-Time Data Handling**: Connect to a WebSocket server at `ws://localhost:8765` to send network data (e.g., feature vectors) for prediction and receive real-time results.
- **Alert Display**: Show live alerts (e.g., "Attack detected" or "Benign") with color-coded indicators (red for attacks, green for benign).
- **Historical Data**: Query a Supabase database (using provided API keys) to fetch and display past predictions in a table, with filters for time range and attack type.
- **Dashboard**: Include a chart (e.g., line or bar) showing attack trends over time based on Supabase data.
- **UI Design**: Use a clean, responsive layout with a live alert banner, a historical data section, and a dashboard.
- **Error Handling**: Handle WebSocket connection failures and invalid data gracefully, displaying user-friendly error messages.

**Technical Details**:
- Use WebSocket for bidirectional communication with the backend.
- Integrate Supabase JavaScript client for database queries (API endpoint: `https://your-supabase-url.supabase.co`, with provided API key).
- Ensure compatibility with modern browsers and mobile devices.