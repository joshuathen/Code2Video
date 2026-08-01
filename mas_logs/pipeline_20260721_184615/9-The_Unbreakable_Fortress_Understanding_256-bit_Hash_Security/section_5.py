from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the section title and lecture lines
        self.setup_layout(
            "Real-World Guardian: Why It Matters",
            [
                "This massive security scale powers Bitcoin and global banking.",
                "It keeps your web browsing and private data safe.",
                "256-bit hashing is the unbreakable foundation of digital trust."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show 'Bitcoin' and 'Bank' logos in green #00FF00 circles.
        self.lecture[0].set_color("#00FF00")
        
        bitcoin_circle = Circle(radius=0.6, color="#00FF00")
        bitcoin_text = Text("BTC", font_size=24, color="#00FF00")
        bitcoin_logo = VGroup(bitcoin_circle, bitcoin_text)
        
        bank_circle = Circle(radius=0.6, color="#00FF00")
        bank_text = Text("BANK", font_size=24, color="#00FF00")
        bank_logo = VGroup(bank_circle, bank_text)
        
        self.place_in_area(bitcoin_logo, "B2", "C3")
        self.place_in_area(bank_logo, "B5", "C6")
        
        self.play(Create(bitcoin_logo), Create(bank_logo))
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # Highlight 'HTTPS' in a browser bar with a green rectangle.
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#00FF00")
        
        # Browser bar construction
        bar_outline = RoundedRectangle(height=1, width=4.5, corner_radius=0.1, color="#FFFFFF")
        browser_text = Text("https://www.secure-site.com", font_size=18, color="#FFFFFF")
        # Indices 0-5 for "https" in the text
        https_highlight_box = SurroundingRectangle(browser_text[0:5], color="#00FF00", buff=0.1)
        browser_bar = VGroup(bar_outline, browser_text, https_highlight_box)
        
        # Fix for Issue 28: Adjust browser_bar position to avoid jarring vertical shift
        self.place_in_area(browser_bar, "C2", "D5")
        
        self.play(
            FadeOut(bitcoin_logo),
            FadeOut(bank_logo),
            Create(bar_outline),
            Write(browser_text)
        )
        self.play(Create(https_highlight_box))
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        # Morph a vault door into a single white star in space.
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#00FF00")
        
        # Vault Door construction
        vault_outer = Square(side_length=1.5, color="#FFFFFF")
        vault_inner = Circle(radius=0.5, color="#FFFFFF")
        # Creating wheel handles for the vault
        vault_spokes = VGroup(*[
            Line(start=ORIGIN, end=0.4 * RIGHT, color="#FFFFFF").rotate(a * DEGREES, about_point=ORIGIN) 
            for a in range(0, 360, 45)
        ])
        vault_door = VGroup(vault_outer, vault_inner, vault_spokes)
        
        # Fix for Issue 29: Adjust vault_door position and size (C2-E5)
        self.place_in_area(vault_door, "C2", "E5")
        
        # Star construction
        star = Star(n=5, outer_radius=0.2, inner_radius=0.1, color="#FFFFFF", fill_opacity=1)
        # Fix for Issue 30: Adjust star position to center within the door's frame (D3-D4)
        self.place_in_area(star, "D3", "D4")
        
        # Background "space" effect
        # Using a fixed seed for reproducibility of random dots
        np.random.seed(42)
        bg_stars = VGroup(*[
            Dot(
                point=[np.random.uniform(0.5, 5.5), np.random.uniform(-2.8, 2.2), 0], 
                radius=0.01, 
                color="#FFFFFF"
            ) for _ in range(40)
        ])
        
        self.play(FadeOut(browser_bar))
        self.play(Create(vault_door))
        self.wait(1.0)
        
        # Morphing vault to star while space appears
        self.play(
            ReplacementTransform(vault_door, star),
            FadeIn(bg_stars)
        )
        
        # Final absorption time
        self.wait(2.0)
