# Session API Documentation

The Session API provides endpoints to manage user sessions in the chatboard system. All endpoints are prefixed with `/sessions`.

## Endpoints

### List All Sessions
```http
GET /sessions/
```

Retrieves a paginated list of all sessions in the system.

**Query Parameters:**
- `limit` (optional): Number of sessions to return (default: 10)
- `offset` (optional): Number of sessions to skip (default: 0)

**Response:**
```json
[
  {
    "id": 1,
    "created_at": "2024-03-15T10:30:00Z",
    "updated_at": "2024-03-15T10:30:00Z",
    "user_id": "user123"
  },
  // ... more sessions
]
```

### Get Session by ID
```http
GET /sessions/{session_id}
```

Retrieves a specific session by its ID.

**Parameters:**
- `session_id`: The unique identifier of the session

**Response:**
```json
{
  "id": 1,
  "created_at": "2024-03-15T10:30:00Z",
  "updated_at": "2024-03-15T10:30:00Z",
  "user_id": "user123"
}
```

**Error Response (404):**
```json
{
  "detail": "Session with id {session_id} not found"
}
```

### List User Sessions
```http
GET /sessions/user/{user_id}
```

Retrieves a paginated list of sessions for a specific user.

**Parameters:**
- `user_id`: The unique identifier of the user

**Query Parameters:**
- `limit` (optional): Number of sessions to return (default: 10)
- `offset` (optional): Number of sessions to skip (default: 0)

**Response:**
```json
[
  {
    "id": 1,
    "created_at": "2024-03-15T10:30:00Z",
    "updated_at": "2024-03-15T10:30:00Z",
    "user_id": "user123"
  },
  // ... more sessions
]
```

### Get User's Last Session
```http
GET /sessions/user/{user_id}/last
```

Retrieves the most recent session for a specific user.

**Parameters:**
- `user_id`: The unique identifier of the user

**Response:**
```json
{
  "id": 1,
  "created_at": "2024-03-15T10:30:00Z",
  "updated_at": "2024-03-15T10:30:00Z",
  "user_id": "user123"
}
```

**Error Response (404):**
```json
{
  "detail": "No sessions found for user {user_id}"
}
```

### Create New Session
```http
POST /sessions/user/{user_id}
```

Creates a new session for a specific user.

**Parameters:**
- `user_id`: The unique identifier of the user

**Response:**
```json
{
  "id": 1,
  "created_at": "2024-03-15T10:30:00Z",
  "updated_at": "2024-03-15T10:30:00Z",
  "user_id": "user123"
}
```

## Example Usage

Here are some example curl commands to interact with the API:

```bash
# List all sessions (paginated)
curl "http://localhost:8000/sessions/"

# List all sessions with custom pagination
curl "http://localhost:8000/sessions/?limit=20&offset=0"

# Get a specific session
curl "http://localhost:8000/sessions/123"

# List sessions for a specific user
curl "http://localhost:8000/sessions/user/user123"

# Get the last session for a user
curl "http://localhost:8000/sessions/user/user123/last"

# Create a new session for a user
curl -X POST "http://localhost:8000/sessions/user/user123"
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Successful operation
- `404`: Resource not found
- `500`: Server error

Error responses include a detail message explaining what went wrong:
```json
{
  "detail": "Error message here"
}
``` 