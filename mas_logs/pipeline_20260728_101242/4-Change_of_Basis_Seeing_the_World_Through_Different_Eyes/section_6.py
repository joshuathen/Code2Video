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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Utility", [
            "Changing the basis doesn't move the physical point.",
            "It simply changes the language used to describe location.",
            "Essential for JPEG compression and computer graphics."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Color change for line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Point labeled 'Fixed Location'
        point = Dot(color=YELLOW)
        # Fix Issue #35: Scale factor changed from 1.5 to 1.0
        self.place_at_grid(point, "C3", scale_factor=1.0)
        label_fixed = Text("Fixed Location", font_size=18, color=YELLOW)
        label_fixed.next_to(point, UP, buff=0.2)
        
        # Two sets of coordinates in standard and custom basis
        val_ae_x = DecimalNumber(0.0, color=BLUE, num_decimal_places=1)
        val_ae_y = DecimalNumber(0.0, color=BLUE, num_decimal_places=1)
        bracket_l_a = MathTex(r"\left[", color=BLUE)
        bracket_r_a = MathTex(r"\right]_E", color=BLUE)
        matrix_a = VGroup(
            bracket_l_a, 
            VGroup(val_ae_x, val_ae_y).arrange(DOWN, buff=0.2), 
            bracket_r_a
        ).arrange(RIGHT, buff=0.1)
        
        val_bb_x = DecimalNumber(0.0, color=GREEN, num_decimal_places=1)
        val_bb_y = DecimalNumber(0.0, color=GREEN, num_decimal_places=1)
        bracket_l_b = MathTex(r"\left[", color=GREEN)
        bracket_r_b = MathTex(r"\right]_B", color=GREEN)
        matrix_b = VGroup(
            bracket_l_b, 
            VGroup(val_bb_x, val_bb_y).arrange(DOWN, buff=0.2), 
            bracket_r_b
        ).arrange(RIGHT, buff=0.1)
        
        self.place_at_grid(matrix_a, "C2", scale_factor=0.8)
        self.place_at_grid(matrix_b, "C4", scale_factor=0.8)
        
        self.play(Create(point), Write(label_fixed))
        self.play(FadeIn(matrix_a), FadeIn(matrix_b))
        
        # Animate coordinates updating to final values
        self.play(
            ChangeDecimalToValue(val_ae_x, 3.0),
            ChangeDecimalToValue(val_ae_y, 2.0),
            ChangeDecimalToValue(val_bb_x, 1.5),
            ChangeDecimalToValue(val_bb_y, 2.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition color to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )
        
        # Emphasis on "language" change
        self.play(
            matrix_a.animate.scale(1.2),
            matrix_b.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition color to line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Real-world utility descriptions
        img_comp_text = Text("Image Compression", font_size=16, color="#00FFFF")
        comp_graph_text = Text("Computer Graphics", font_size=16, color="#00FFFF")
        
        # Icons
        image_icon = Square(side_length=0.5, color="#00FFFF", fill_opacity=0.3)
        graphics_icon = Triangle(color="#00FFFF", fill_opacity=0.3).scale(0.3)
        
        icon_group_1 = VGroup(image_icon, img_comp_text).arrange(DOWN, buff=0.1)
        icon_group_2 = VGroup(graphics_icon, comp_graph_text).arrange(DOWN, buff=0.1)
        
        # Fix Issue #34: Using place_in_area for better text spacing
        self.place_in_area(icon_group_1, "E1", "E3")
        self.place_in_area(icon_group_2, "E4", "E6")
        
        self.play(
            FadeIn(icon_group_1, shift=UP),
            FadeIn(icon_group_2, shift=UP)
        )
        self.wait(2)
        
        # === Final Summary Transition ===
        # Fade out and show final message
        self.play(
            FadeOut(point, label_fixed, matrix_a, matrix_b, icon_group_1, icon_group_2),
            self.lecture.animate.set_color(WHITE)
        )
        
        final_text = Text("Same Vector, Different Basis", font_size=28, color=WHITE)
        # Fix Issue #36: Using refined area C2-D5
        self.place_in_area(final_text, "C2", "D5")
        
        self.play(Write(final_text))
        self.wait(3)
