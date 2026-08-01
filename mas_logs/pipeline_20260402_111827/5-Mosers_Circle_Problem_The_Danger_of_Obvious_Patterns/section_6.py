from manim import *

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define content
        title = "Conclusion: The Lesson of Moser's Problem"
        lines = [
            "Never trust a pattern without a rigorous proof.",
            "Moser’s problem warns us against jumping to conclusions.",
            "True rules often hide behind misleadingly simple starts."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors for highlighting
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_3 = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        pattern_text = Text("Pattern", color=RED_A)
        vs_text = Text("vs.", color=WHITE)
        proof_text = Text("Rigorous Proof", color=GREEN_A)
        
        comparison = VGroup(pattern_text, vs_text, proof_text).arrange(RIGHT, buff=0.3)
        # Position at the top to avoid collision with subsequent triangle
        self.place_in_area(comparison, "A1", "A6", scale_factor=0.8)
        
        self.play(Write(comparison))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        
        # Generate Pascal's Triangle
        def get_pascal_row(n):
            from math import comb
            return [comb(n, k) for k in range(n + 1)]

        triangle_rows = []
        for i in range(6): # Rows 0 to 5
            row_vals = get_pascal_row(i)
            row_tex = VGroup(*[Text(str(val), font_size=24) for val in row_vals]).arrange(RIGHT, buff=0.5)
            triangle_rows.append(row_tex)
        
        pascal_triangle = VGroup(*triangle_rows).arrange(DOWN, buff=0.3)
        # Resized and moved to center-lower area to prevent overlap with header
        self.place_in_area(pascal_triangle, "B2", "E5", scale_factor=0.6)
        
        # Highlight first 5 elements of 6th row (index 5)
        last_row = triangle_rows[5]
        highlight_rects = VGroup()
        for i in range(5): # First 5 elements: 1, 5, 10, 10, 5
            last_row[i].set_color(COLOR_2)
            rect = SurroundingRectangle(last_row[i], color=COLOR_2, buff=0.1)
            highlight_rects.add(rect)
            
        sum_label = Text("Sum = 31", font_size=24, color=COLOR_2)
        # Placed near the base of the triangle
        self.place_at_grid(sum_label, "F4", scale_factor=0.9)

        self.play(
            FadeOut(comparison),
            FadeIn(pascal_triangle)
        )
        self.play(
            Create(highlight_rects),
            Write(sum_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_3)
        
        final_message = Text(
            "Never trust a pattern until\nyou have the proof.", 
            font_size=32, 
            color=WHITE,
            line_spacing=1
        )
        self.place_in_area(final_message, "B1", "E6", scale_factor=1.0)

        self.play(
            FadeOut(pascal_triangle),
            FadeOut(highlight_rects),
            FadeOut(sum_label),
            FadeIn(final_message)
        )
        self.wait(3)
