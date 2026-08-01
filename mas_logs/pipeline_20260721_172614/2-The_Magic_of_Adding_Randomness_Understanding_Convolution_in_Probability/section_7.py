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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("Summary and Real-World Sync", [
            "Convolution blends independent uncertainties into a single result.",
            "Remember the three steps: Flip, Slide, and Measure.",
            "From radio noise to robotics, convolution is everywhere."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color("#FFFF00")

        # Cat Icon - Use Asset from Issue 26
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png]
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        # Bell Curve
        bell_curve = FunctionGraph(
            lambda x: np.exp(-x**2),
            x_range=[-2, 2],
            color="#00FF00"
        )
        
        # Formula (Symbolic)
        formula = MathTex(
            r"(f * g)(t)",
            color="#FFFF00"
        )
        
        # Positioning using the 6x6 grid
        # Apply Issue 38: Move to row A to avoid vertical crowding
        self.place_at_grid(cat_icon, "A2", scale_factor=0.8)
        self.place_at_grid(bell_curve, "A4", scale_factor=0.6)
        
        # Apply Issue 39: Move formula to area A5-A6 and scale to 0.9
        self.place_in_area(formula, 'A5', 'A6', scale_factor=0.9)
        
        self.play(FadeIn(cat_icon), Create(bell_curve), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update line colors: shift highlight to second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        # Pulse all three icons simultaneously to show conceptual connection
        self.play(
            cat_icon.animate.scale(1.2),
            bell_curve.animate.scale(1.2),
            formula.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update line colors: shift highlight to third line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        # Thank You message replacing icons
        thanks = Text("Thank You!", font_size=42, color="#FFFFFF")
        
        # Apply Issue 37: Use area C2-E5 to avoid overlap with previous icon positions
        self.place_in_area(thanks, 'C2', 'E5', scale_factor=1.0)
        
        self.play(
            FadeOut(cat_icon),
            Uncreate(bell_curve),
            FadeOut(formula),
            FadeIn(thanks)
        )
        self.wait(3)
