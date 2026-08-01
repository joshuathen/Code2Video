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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "The Search Problem: Classical vs. Quantum", 
            [
                "Imagine finding one specific item in an unsorted list.", 
                "Classically, you must check items one by one.", 
                "This takes N over two tries on average."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line (matching grey squares)
        self.lecture[0].set_color("#808080")
        
        # Create a 4x4 grid of grey (#808080) squares
        # Coordinates mapping for a 4x4 grid within the B2-E5 area
        grid_coords = [
            "B2", "B3", "B4", "B5",
            "C2", "C3", "C4", "C5",
            "D2", "D3", "D4", "D5",
            "E2", "E3", "E4", "E5"
        ]
        
        squares = VGroup()
        for coord in grid_coords:
            sq = Square(side_length=0.7, color="#808080", fill_opacity=0.3)
            self.place_at_grid(sq, coord)
            squares.add(sq)
            
        # Asset integration (Issue 26): Item icon at index 11 (D5)
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/item.svg]
        item_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/item.svg")
        self.place_at_grid(item_icon, "D5", scale_factor=0.4)
        item_icon.set_opacity(0) # Secretly present
            
        self.play(FadeIn(squares))
        self.add(item_icon)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line (matching light blue classical circle)
        self.lecture[1].set_color("#ADD8E6")
        
        # Fix: classical_label position and scale to avoid overlap (Issue 28)
        classical_label = Text("Classical Search", font_size=18, color="#ADD8E6")
        self.place_at_grid(classical_label, "A2", scale_factor=0.8)
        
        search_circle = Circle(radius=0.25, color="#ADD8E6", stroke_width=4)
        self.place_at_grid(search_circle, "B2")
        
        self.play(FadeIn(search_circle), FadeIn(classical_label))
        
        # Sequential movement through first 5 squares (indices 0 to 4)
        for i in range(1, 5):
            target_coord = grid_coords[i]
            self.play(
                search_circle.animate.move_to(self.grid[target_coord]),
                run_time=0.4
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line (matching gold quantum elements)
        self.lecture[2].set_color("#FFD700")
        
        # All squares flash light blue (#ADD8E6)
        flash_highlights = VGroup(*[
            Square(side_length=0.7, color="#ADD8E6", fill_opacity=0.6).move_to(sq.get_center())
            for sq in squares
        ])
        
        self.play(FadeIn(flash_highlights), run_time=0.3)
        self.play(FadeOut(flash_highlights), run_time=0.3)
        
        # Reveal gold square (#FFD700) at index 11 (D5) with item and label
        gold_square_index = 11
        target_square = squares[gold_square_index]
        
        # Fix: quantum_label position and scale (Issue 29)
        quantum_label = Text("Quantum Speedup", font_size=18, color="#FFD700")
        self.place_at_grid(quantum_label, "D6", scale_factor=0.8)
        
        self.play(
            target_square.animate.set_color("#FFD700").set_fill("#FFD700", opacity=0.8),
            item_icon.animate.set_opacity(1),
            FadeOut(search_circle),
            FadeOut(classical_label),
            Write(quantum_label)
        )
        self.wait(2)
