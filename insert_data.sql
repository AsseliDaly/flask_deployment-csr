USE product_catalogue;

-- Insert sample products
INSERT INTO all_products (product_full, marque, category, available, store, prix_2023_winter, prix_2023_spring, prix_2023_summer, prix_2023_fall, prix_2024_winter) VALUES
('iPhone 14 Pro 256GB', 'Apple', 'Smartphones', TRUE, 'MyTek', 4999.00, 4899.00, 4799.00, 4699.00, 4599.00),
('Samsung Galaxy S23 Ultra', 'Samsung', 'Smartphones', TRUE, 'Tunisianet', 4599.00, 4499.00, 4399.00, 4299.00, 4199.00),
('MacBook Pro M2 13"', 'Apple', 'Laptops', TRUE, 'MyTek', 6999.00, 6899.00, 6799.00, 6699.00, 6599.00),
('PS5 Digital Edition', 'Sony', 'Gaming', TRUE, 'Spacenet', 2999.00, 2899.00, 2799.00, 2699.00, 2599.00),
('LG OLED C2 65"', 'LG', 'TVs', TRUE, 'MyTek', 7999.00, 7899.00, 7799.00, 7699.00, 7599.00);

-- Insert sample hotels
INSERT INTO hotels (name, city, formule, nb_etoiles, prix_2023_winter, prix_2023_spring, prix_2023_summer, prix_2023_fall) VALUES
('Movenpick Gammarth', 'Tunis', 'All Inclusive', 5, 450.00, 500.00, 650.00, 480.00),
('Royal Thalassa', 'Monastir', 'Half Board', 5, 380.00, 420.00, 580.00, 400.00),
('Medina Belisaire', 'Hammamet', 'All Inclusive', 4, 280.00, 320.00, 450.00, 300.00),
('Iberostar Selection', 'Djerba', 'All Inclusive', 5, 420.00, 460.00, 620.00, 440.00),
('Magic Life Africana', 'Hammamet', 'All Inclusive', 4, 300.00, 340.00, 480.00, 320.00);

-- Insert sample user-product relationships
INSERT INTO user_products (client_id, product_id, categorie_enc, brand, price) VALUES
(1, 1, 1, 'Apple', 4999.00),
(1, 3, 2, 'Apple', 6999.00),
(2, 2, 1, 'Samsung', 4599.00),
(2, 4, 3, 'Sony', 2999.00),
(3, 5, 4, 'LG', 7999.00);

-- Add sample data verification
SELECT 'Products count: ' as '', COUNT(*) FROM all_products;
SELECT 'Hotels count: ' as '', COUNT(*) FROM hotels;
SELECT 'User-Products count: ' as '', COUNT(*) FROM user_products;
