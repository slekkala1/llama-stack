# Shared Types

```python
from llama_stack_cli.types import Order
```

# Pet

Types:

```python
from llama_stack_cli.types import (
    Category,
    Pet,
    PetFindByStatusResponse,
    PetFindByTagsResponse,
    PetUploadImageResponse,
)
```

Methods:

- <code title="post /pet">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">create</a>(\*\*<a href="src/llama_stack_cli/types/pet_create_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/pet.py">Pet</a></code>
- <code title="get /pet/{petId}">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">retrieve</a>(pet_id) -> <a href="./src/llama_stack_cli/types/pet.py">Pet</a></code>
- <code title="put /pet">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">update</a>(\*\*<a href="src/llama_stack_cli/types/pet_update_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/pet.py">Pet</a></code>
- <code title="delete /pet/{petId}">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">delete</a>(pet_id) -> None</code>
- <code title="get /pet/findByStatus">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">find_by_status</a>(\*\*<a href="src/llama_stack_cli/types/pet_find_by_status_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/pet_find_by_status_response.py">PetFindByStatusResponse</a></code>
- <code title="get /pet/findByTags">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">find_by_tags</a>(\*\*<a href="src/llama_stack_cli/types/pet_find_by_tags_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/pet_find_by_tags_response.py">PetFindByTagsResponse</a></code>
- <code title="post /pet/{petId}">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">update_by_id</a>(pet_id, \*\*<a href="src/llama_stack_cli/types/pet_update_by_id_params.py">params</a>) -> None</code>
- <code title="post /pet/{petId}/uploadImage">client.pet.<a href="./src/llama_stack_cli/resources/pet.py">upload_image</a>(pet_id, \*\*<a href="src/llama_stack_cli/types/pet_upload_image_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/pet_upload_image_response.py">PetUploadImageResponse</a></code>

# Store

Types:

```python
from llama_stack_cli.types import StoreListInventoryResponse
```

Methods:

- <code title="get /store/inventory">client.store.<a href="./src/llama_stack_cli/resources/store/store.py">list_inventory</a>() -> <a href="./src/llama_stack_cli/types/store_list_inventory_response.py">StoreListInventoryResponse</a></code>

## Order

Methods:

- <code title="post /store/order">client.store.order.<a href="./src/llama_stack_cli/resources/store/order.py">create</a>(\*\*<a href="src/llama_stack_cli/types/store/order_create_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/shared/order.py">Order</a></code>
- <code title="get /store/order/{orderId}">client.store.order.<a href="./src/llama_stack_cli/resources/store/order.py">retrieve</a>(order_id) -> <a href="./src/llama_stack_cli/types/shared/order.py">Order</a></code>
- <code title="delete /store/order/{orderId}">client.store.order.<a href="./src/llama_stack_cli/resources/store/order.py">delete</a>(order_id) -> None</code>

# User

Types:

```python
from llama_stack_cli.types import User, UserLoginResponse
```

Methods:

- <code title="post /user">client.user.<a href="./src/llama_stack_cli/resources/user.py">create</a>(\*\*<a href="src/llama_stack_cli/types/user_create_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/user.py">User</a></code>
- <code title="get /user/{username}">client.user.<a href="./src/llama_stack_cli/resources/user.py">retrieve</a>(username) -> <a href="./src/llama_stack_cli/types/user.py">User</a></code>
- <code title="put /user/{username}">client.user.<a href="./src/llama_stack_cli/resources/user.py">update</a>(existing_username, \*\*<a href="src/llama_stack_cli/types/user_update_params.py">params</a>) -> None</code>
- <code title="delete /user/{username}">client.user.<a href="./src/llama_stack_cli/resources/user.py">delete</a>(username) -> None</code>
- <code title="post /user/createWithList">client.user.<a href="./src/llama_stack_cli/resources/user.py">create_with_list</a>(\*\*<a href="src/llama_stack_cli/types/user_create_with_list_params.py">params</a>) -> <a href="./src/llama_stack_cli/types/user.py">User</a></code>
- <code title="get /user/login">client.user.<a href="./src/llama_stack_cli/resources/user.py">login</a>(\*\*<a href="src/llama_stack_cli/types/user_login_params.py">params</a>) -> str</code>
- <code title="get /user/logout">client.user.<a href="./src/llama_stack_cli/resources/user.py">logout</a>() -> None</code>
