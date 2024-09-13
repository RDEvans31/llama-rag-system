# Base image with Node.js
FROM node:20-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy project code
COPY . .

# Set environment for production
ENV NODE_ENV production

# Optional: Create non-root user for security
# ... (Uncomment these lines if desired)

# Expose port
EXPOSE 3000

# Start the application with npm
CMD [ "npm", "run", "dev" ]