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
        # Initial Setup
        title = "The Hook: The Morphing Canvas"
        lines = [
            "Imagine Pixel the Cat sitting on a unit grid.",
            "A matrix transformation tilts and stretches the entire space.",
            "Pixel’s shape changes, becoming taller, thinner, or slanted.",
            "The determinant measures the change in Pixel’s total area.",
            "It is the scaling factor for every shape in space."
        ]
        self.setup_layout(title, lines)

        # Colors for highlights
        color_cat = "#FFFFFF"
        color_grid = "#444444"
        color_area = "#FFFF00"
        color_line_active = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_line_active))
        
        # Create a simple "Pixel the Cat" character
        cat_head = Circle(radius=0.3, color=WHITE, fill_opacity=1)
        cat_ear_l = Triangle(color=WHITE, fill_opacity=1).scale(0.1).rotate(30*DEGREES).move_to(cat_head.get_top() + LEFT*0.15 + UP*0.05)
        cat_ear_r = Triangle(color=WHITE, fill_opacity=1).scale(0.1).rotate(-30*DEGREES).move_to(cat_head.get_top() + RIGHT*0.15 + UP*0.05)
        cat_eye_l = Dot(radius=0.04, color=BLACK).move_to(cat_head.get_center() + LEFT*0.1 + UP*0.05)
        cat_eye_r = Dot(radius=0.04, color=BLACK).move_to(cat_head.get_center() + RIGHT*0.1 + UP*0.05)
        cat_smile = Arc(radius=0.1, start_angle=200*DEGREES, angle=140*DEGREES, color=BLACK).move_to(cat_head.get_center() + DOWN*0.1)
        pixel_cat = VGroup(cat_head, cat_ear_l, cat_ear_r, cat_eye_l, cat_eye_r, cat_smile)
        
        # Unit square grid representation
        unit_square = Rectangle(height=1.0, width=1.0, stroke_color=WHITE, stroke_width=2).move_to([0.5, 0.5, 0])
        
        # Background grid (visual only)
        bg_grid = VGroup()
        for i in range(-2, 4):
            bg_grid.add(Line([i, -2, 0], [i, 3, 0], color=color_grid, stroke_width=1))
            bg_grid.add(Line([-2, i, 0], [3, i, 0], color=color_grid, stroke_width=1))

        workspace = VGroup(bg_grid, unit_square, pixel_cat)
        
        # Resolve Issue 32: Shift workspace right and down and reduce scale to avoid lecture overlap
        self.place_in_area(workspace, 'B3', 'F6', scale_factor=0.6)
        
        # Adjust cat initial position relative to the local square
        pixel_cat.move_to(unit_square.get_center())

        self.play(FadeIn(bg_grid), Create(unit_square))
        self.play(FadeIn(pixel_cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_line_active)
        )
        
        # Matrix [[2, 1], [0.5, 1.5]]
        # Apply transformation to the workspace
        self.play(
            workspace.animate.apply_matrix(np.array([[2, 1], [0.5, 1.5]]).T),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_line_active)
        )
        # Emphasis on the stretched Pixel
        self.play(pixel_cat.animate.set_stroke(width=2, color=YELLOW), run_time=0.5)
        self.play(pixel_cat.animate.set_stroke(width=0, color=WHITE), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(color_line_active)
        )
        
        # Highlight transformed square area (now a parallelogram)
        filled_area = unit_square.copy().set_fill(color_area, opacity=0.3).set_stroke(width=0)
        
        question_mark = Text("?", font_size=48, color=color_area)
        # Resolve Issue 33: Move question_mark to D4 and adjust scale
        self.place_at_grid(question_mark, 'D4', scale_factor=0.8)
        
        self.play(FadeIn(filled_area))
        self.play(Write(question_mark))
        
        # Pulse effect
        self.play(question_mark.animate.scale(1.3), run_time=0.5, rate_func=there_and_back)
        self.play(question_mark.animate.scale(1.3), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(color_line_active)
        )
        
        # Generalizing: add a second shape (small circle) to show it scales too
        extra_shape = Circle(radius=0.2, color=BLUE, fill_opacity=0.5)
        # Resolve Issue 34: Move extra_shape to D5 to avoid transformation overlap
        self.place_at_grid(extra_shape, 'D5', scale_factor=0.8)
        self.play(FadeIn(extra_shape))
        self.play(extra_shape.animate.apply_matrix(np.array([[2, 1], [0.5, 1.5]]).T))
        
        self.wait(2)
        self.play(FadeOut(question_mark), FadeOut(filled_area), FadeOut(extra_shape))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
