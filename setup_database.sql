-- Create database
CREATE DATABASE IF NOT EXISTS product_catalogue;
USE product_catalogue;

-- Create all_products table
CREATE TABLE IF NOT EXISTS all_products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_full VARCHAR(255) NOT NULL,
    marque VARCHAR(100),
    category VARCHAR(100),
    available BOOLEAN DEFAULT TRUE,
    store VARCHAR(100),
    prix_2023_winter DECIMAL(10,2),
    prix_2023_spring DECIMAL(10,2),
    prix_2023_summer DECIMAL(10,2),
    prix_2023_fall DECIMAL(10,2),
    prix_2024_winter DECIMAL(10,2),
    prix_2024_spring DECIMAL(10,2),
    prix_2024_summer DECIMAL(10,2),
    prix_2024_fall DECIMAL(10,2),
    prix_2025_winter DECIMAL(10,2),
    INDEX idx_product_full (product_full)
);

-- Create hotels table
CREATE TABLE IF NOT EXISTS hotels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    formule VARCHAR(50),
    nb_etoiles INT,
    prix_2023_winter DECIMAL(10,2),
    prix_2023_spring DECIMAL(10,2),
    prix_2023_summer DECIMAL(10,2),
    prix_2023_fall DECIMAL(10,2),
    prix_2024_winter DECIMAL(10,2),
    prix_2024_spring DECIMAL(10,2),
    prix_2024_summer DECIMAL(10,2),
    prix_2024_fall DECIMAL(10,2),
    prix_2025_winter DECIMAL(10,2),
    INDEX idx_city (city),
    INDEX idx_name (name)
);

-- Create table for user-product data
CREATE TABLE IF NOT EXISTS user_products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id INT,
    product_id INT,
    categorie_enc INT,
    brand VARCHAR(100),
    price DECIMAL(10,2),
    FOREIGN KEY (product_id) REFERENCES all_products(id),
    INDEX idx_client (client_id),
    INDEX idx_product (product_id)
);

-- Grant permissions to Flask app user

