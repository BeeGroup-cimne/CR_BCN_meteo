# Etapa de construcción
FROM golang:1.23.1-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./

# Etapa de ejecución
FROM alpine:latest
WORKDIR /
COPY --from=builder /go-api /go-api
EXPOSE 8080
