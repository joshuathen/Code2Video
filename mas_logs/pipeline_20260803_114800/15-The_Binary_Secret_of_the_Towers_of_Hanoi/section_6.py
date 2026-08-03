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
        # Initialize Scene
        title_str = "Conclusion: Mathematical Elegance"
        lines = [
            "Binary logic transforms a complex puzzle into simple counting.",
            "Computers use this efficiency to solve recursive problems.",
            "The Tower of Hanoi hides a binary heartbeat."
        ]
        self.setup_layout(title_str, lines)

        # Colors for matching lecture lines with animation elements
        bridge_color = "#B0C4DE"
        tower_highlight = "#FFA500"
        final_white = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Binary logic transforms a complex puzzle into simple counting.
        self.lecture[0].set_color(bridge_color)
        
        # Create 'Binary Bridge' visual
        # Two horizontal lines with binary digits (0, 1) floating between them.
        top_line = Line(start=self.grid["B1"], end=self.grid["B6"], color=bridge_color, stroke_width=4)
        bottom_line = Line(start=self.grid["D1"], end=self.grid["D6"], color=bridge_color, stroke_width=4)
        
        digits = VGroup()
        binary_pattern = ["1", "0", "1", "1", "0", "1"]
        for idx, bit in enumerate(binary_pattern):
            d_tex = Text(bit, font_size=32, color=WHITE)
            # Position digits along row C
            self.place_at_grid(d_tex, f"C{idx+1}")
            digits.add(d_tex)
        
        bridge_group = VGroup(top_line, bottom_line, digits)
        
        self.play(Create(top_line), Create(bottom_line), run_time=1.5)
        self.play(FadeIn(digits, lag_ratio=0.15, shift=UP*0.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Computers use this efficiency to solve recursive problems.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(tower_highlight)
        
        # Formula '2^n - 1'
        # Fix from VideoCritic: Line 92: self.place_in_area(formula, 'B1', 'C6', scale_factor=0.9)
        formula = MathTex("2^n - 1", color=WHITE, font_size=48)
        self.place_in_area(formula, 'B1', 'C6', scale_factor=0.9)
        
        # 3-disk Tower of Hanoi stack
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg
        # Fix from VideoCritic: Line 103: self.place_in_area(tower, 'D4', 'E6', scale_factor=0.7)
        tower = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        # Ensure visible if the SVG defaults to black
        tower.set_color(WHITE)
        self.place_in_area(tower, 'D4', 'E6', scale_factor=0.7)
        
        self.play(
            Write(formula),
            FadeIn(tower, shift=RIGHT),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Tower of Hanoi hides a binary heartbeat.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(final_white)
        
        # Group everything and scale down as per storyboard
        all_graphics = VGroup(bridge_group, formula, tower)
        
        # Final conclusion title
        # Fix from VideoCritic: Line 122: self.place_in_area(final_conclusion_title, 'F1', 'F6', scale_factor=0.8)
        final_conclusion_title = Text("Mathematical Elegance", color=final_white, font_size=44)
        self.place_in_area(final_conclusion_title, 'F1', 'F6', scale_factor=0.8)
        
        self.play(
            all_graphics.animate.scale(0.5).set_opacity(0.3).shift(UP*0.5),
            FadeIn(final_conclusion_title, scale=1.1),
            run_time=2.5
        )
        self.wait(5)
