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
        title = "The Conventional Ruler vs. The New Perspective"
        lecture_lines = [
            "In standard geometry, distance grows as numbers move from zero.",
            "But what if divisibility by two determined closeness?",
            "Imagine Binary Bob, living where powers of two shrink.",
            "On a standard ruler, 1024 is very far away.",
            "Through our 2-adic lens, 1024 is actually tiny."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        standard_color = WHITE
        metric_color = "#00FF00"  # Green as requested
        
        # === Animation for Lecture Line 1 ===
        # Draw standard ruler [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg] with numbers 0 to 10
        self.lecture[0].set_color(standard_color)
        
        # Load ruler asset
        ruler_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg").set_color(WHITE)
        self.place_in_area(ruler_asset, 'B1', 'B6', scale_factor=1.2)
        
        # Numbers for the ruler
        numbers = VGroup(*[
            MathTex(str(i), font_size=20, color=WHITE)
            for i in range(0, 11, 2)
        ])
        for i, num in enumerate(numbers):
            grid_col = str(i + 1)
            self.place_at_grid(num, f"C{grid_col}", scale_factor=1.0)
            
        std_label = Text("Standard Distance", font_size=20, color=metric_color)
        # Resolved Issue 18: Fix label positioning
        self.place_in_area(std_label, 'A2', 'A5', scale_factor=0.8)
        
        self.play(FadeIn(ruler_asset), Write(numbers), Write(std_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight even numbers to hint at divisibility
        self.lecture[1].set_color(metric_color)
        highlight_rects = VGroup(*[
            SurroundingRectangle(num, color=metric_color, buff=0.1)
            for num in numbers
        ])
        self.play(Create(highlight_rects))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to 2-adic world
        self.lecture[2].set_color(metric_color)
        self.play(FadeOut(highlight_rects), FadeOut(ruler_asset), FadeOut(numbers))
        
        # Create a new 2-adic axis
        adic_axis = Line(self.grid['D1'], self.grid['D6'], color=metric_color)
        origin_label = MathTex("0", color=WHITE)
        self.place_at_grid(origin_label, 'D1', scale_factor=0.8)
        
        self.play(Create(adic_axis), Write(origin_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # On a standard ruler, 1024 is very far away
        self.lecture[3].set_color(standard_color)
        standard_1024 = MathTex("1024", color=standard_color)
        self.place_at_grid(standard_1024, 'B6', scale_factor=1.0)
        
        break_symbol = MathTex("//", color=standard_color)
        self.place_at_grid(break_symbol, 'B4', scale_factor=1.0)
        
        ruler_start = Line(self.grid['B1'], self.grid['B3'], color=standard_color)
        ruler_end = Line(self.grid['B5'], self.grid['B6'], color=standard_color)
        
        self.play(Create(ruler_start), Create(break_symbol), Create(ruler_end), Write(standard_1024))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Through our 2-adic lens, 1024 is tiny
        self.lecture[4].set_color(metric_color)
        
        adic_1024 = MathTex("1024", color=WHITE)
        # 1024 = 2^10, very close to zero
        self.place_at_grid(adic_1024, 'D2', scale_factor=0.4) 
        
        adic_2 = MathTex("2", color=WHITE)
        # Resolved Issue 20: Change scaling and position of '2'
        self.place_at_grid(adic_2, 'D6', scale_factor=1.0) 
        
        adic_dist_label = Text("2-adic Distance", font_size=20, color=metric_color)
        # Resolved Issue 19: Fix label positioning for '2-adic Distance'
        self.place_in_area(adic_dist_label, 'E2', 'E5', scale_factor=0.8)
        
        self.play(
            FadeOut(ruler_start), FadeOut(break_symbol), FadeOut(ruler_end), FadeOut(standard_1024),
            Transform(standard_1024.copy(), adic_1024),
            Write(adic_2),
            Write(adic_dist_label)
        )
        self.wait(2)
