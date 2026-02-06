# User API Documentation

The User API provides endpoints to manage users and their associated sessions and messages in the chatboard system. All endpoints are prefixed with `/api/users`.

## Models

### UserCreate
Dynamic model that is generated based on the provided user model class. For the default `AppAUser`, it includes:
```json
{
    "name": "string",
    "email": "string | null",
    "phone_number": "string | null",
    "type": "app_a_user"  // Set automatically based on user model
}
```

### UserResponse
Dynamic model that includes all fields from the user model plus standard fields:
```json
{
    "id": "integer",
    "created_at": "datetime",
    "updated_at": "datetime",
    "type": "string",
    // Additional fields from user model (e.g., for AppAUser):
    "name": "string | null",
    "email": "string | null",
    "phone_number": "string | null"
}
```

## Endpoints

### Create User
```http
POST /api/users/
```

Creates a new user with the provided data.

**Request Body:** UserCreate model

**Response:** UserResponse model

**Example:**
```json
// Request
{
    "name": "John Doe",
    "email": "john@example.com",
    "phone_number": "1234567890"
}

// Response
{
    "id": 1,
    "created_at": "2024-03-15T10:30:00Z",
    "updated_at": "2024-03-15T10:30:00Z",
    "type": "app_a_user",
    "name": "John Doe",
    "email": "john@example.com",
    "phone_number": "1234567890"
}
```

### Get User by ID
```http
GET /api/users/{user_id}
```

Retrieves a specific user by their ID.

**Parameters:**
- `user_id`: The unique identifier of the user

**Response:** UserResponse model

**Error Response (404):**
```json
{
    "detail": "User with id {user_id} not found"
}
```

### List Users
```http
GET /api/users/
```

Retrieves a paginated list of users.

**Query Parameters:**
- `limit` (optional): Number of users to return (default: 10)
- `offset` (optional): Number of users to skip (default: 0)

**Response:** Array of UserResponse models

### Create User Session
```http
POST /api/users/{user_id}/sessions
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
    "user_id": "1"
}
```

**Error Response (404):**
```json
{
    "detail": "User with id {user_id} not found"
}
```

### List User Sessions
```http
GET /api/users/{user_id}/sessions
```

Retrieves a paginated list of sessions for a specific user.

**Parameters:**
- `user_id`: The unique identifier of the user

**Query Parameters:**
- `limit` (optional): Number of sessions to return (default: 10)
- `offset` (optional): Number of sessions to skip (default: 0)

**Response:** Array of session objects

### Get User Messages
```http
GET /api/users/{user_id}/messages
```

Retrieves a paginated list of messages from all sessions for a specific user.

**Parameters:**
- `user_id`: The unique identifier of the user

**Query Parameters:**
- `limit` (optional): Number of messages to return (default: 10)
- `offset` (optional): Number of messages to skip (default: 0)

**Response:** Array of message objects
```json
[
    {
        "id": 1,
        "created_at": "2024-03-15T10:30:00Z",
        "role": "user",
        "name": "John",
        "content": "Hello!",
        "branch_order": 0,
        "branch_id": 1,
        "turn_id": 1
    }
]
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Successful operation
- 404: Resource not found
- 500: Internal server error

Error responses include a detail message:
```json
{
    "detail": "Error message describing what went wrong"
}
```

## Notes

1. The API is designed to be generic and work with any user model that extends BaseUserModel
2. The actual fields available in UserCreate and UserResponse models depend on the user model class provided to the router factory
3. All datetime fields are in ISO 8601 format
4. All list endpoints support pagination through limit/offset parameters 