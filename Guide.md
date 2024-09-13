1. Prerequisites - set up application and install packages

    I had a full-stack template project with a next.js frontend connected to a python fast api backend, containerised in Docker.
    You can access the template here, but I would encourage you to set it up from scratch for you're own learning. 
    The requirements file for all the python packages you need to install will be included in the repo.


2. Endpoint that can receive, loading and split PDF into chunks.

3. Upload chunks to vector store in batches incase of any failure in any one upload

4. Endpoint that receives a query, 