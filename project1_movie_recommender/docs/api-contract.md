# Stable API contract

GET /v1/recommendations/content/{item_id}?limit=10
POST /v1/recommendations/content/query?limit=10   body: {"title", "genres", "description"}
GET /v1/recommendations/user/{user_id}?limit=10
POST /v1/events

The browser calls your application API. It does not hold AWS credentials and does not invoke SageMaker directly.
