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
        # Setup Title and Lecture Lines
        title = "The Digital Fingerprint: What is a Hash?"
        lines = [
            "A hash function is like a digital blender.",
            "It transforms any input into a unique string.",
            "One small change creates a completely different hash."
        ]
        self.setup_layout(title, lines)

        # Colors for consistency
        COLOR_BLENDER = "#FFFFFF"
        COLOR_CAT = "#ADD8E6"
        COLOR_HASH1 = "#FFFFE0"
        COLOR_HASH2 = "#FFD700"
        COLOR_WHISKER = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Create a stylized blender
        blender_jar = Polygon(
            [-0.6, -0.8, 0], [0.6, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0],
            color=COLOR_BLENDER, fill_opacity=0.2, stroke_width=4
        )
        blender_lid = Line([-0.8, 0.8, 0], [0.8, 0.8, 0], color=COLOR_BLENDER, stroke_width=8)
        blender_base = Rectangle(height=0.4, width=1.0, color=COLOR_BLENDER, fill_opacity=0.4).next_to(blender_jar, DOWN, buff=0)
        blender_main = VGroup(blender_jar, blender_lid, blender_base)
        
        blender_label = Text("Hash Function", font_size=24, color=COLOR_BLENDER)
        
        # Place blender components using grid
        self.place_in_area(blender_main, "B2", "D3", scale_factor=0.7)
        self.place_in_area(blender_label, "E2", "E3", scale_factor=0.8)
        
        blender_center = blender_main.get_center()

        # Action: Highlight line and fade in blender
        self.play(self.lecture[0].animate.set_color(COLOR_BLENDER))
        self.play(FadeIn(blender_main), FadeIn(blender_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a stylized blue cat input
        cat_body = Circle(radius=0.4, color=COLOR_CAT, fill_opacity=0.8)
        cat_head = Circle(radius=0.25, color=COLOR_CAT, fill_opacity=0.8).next_to(cat_body, UP, buff=-0.1)
        cat_ear_l = Triangle(color=COLOR_CAT, fill_opacity=0.8).scale(0.1).rotate(-30*DEGREES).move_to(cat_head.get_top() + LEFT*0.1)
        cat_ear_r = Triangle(color=COLOR_CAT, fill_opacity=0.8).scale(0.1).rotate(30*DEGREES).move_to(cat_head.get_top() + RIGHT*0.1)
        cat_input = VGroup(cat_body, cat_head, cat_ear_l, cat_ear_r)
        
        # Place cat input at C1
        self.place_at_grid(cat_input, "C1", scale_factor=0.6)

        # Create the resulting hex string hash
        hex_output_1 = Text("a591...", font_size=32, color=COLOR_HASH1)
        self.place_at_grid(hex_output_1, "C5")

        # Action: Highlight line, feed cat into blender, show output hash
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HASH1)
        )
        self.play(FadeIn(cat_input))
        self.play(
            cat_input.animate.move_to(blender_center).scale(0.3).set_opacity(0),
            run_time=2,
            rate_func=slow_into
        )
        self.play(Write(hex_output_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create the modified cat (with a red whisker)
        cat_body_2 = Circle(radius=0.4, color=COLOR_CAT, fill_opacity=0.8)
        cat_head_2 = Circle(radius=0.25, color=COLOR_CAT, fill_opacity=0.8).next_to(cat_body_2, UP, buff=-0.1)
        cat_ear_l_2 = Triangle(color=COLOR_CAT, fill_opacity=0.8).scale(0.1).rotate(-30*DEGREES).move_to(cat_head_2.get_top() + LEFT*0.1)
        cat_ear_r_2 = Triangle(color=COLOR_CAT, fill_opacity=0.8).scale(0.1).rotate(30*DEGREES).move_to(cat_head_2.get_top() + RIGHT*0.1)
        whisker = Line(LEFT*0.3, RIGHT*0.3, color=COLOR_WHISKER, stroke_width=6).move_to(cat_head_2.get_center())
        cat_input_2 = VGroup(cat_body_2, cat_head_2, cat_ear_l_2, cat_ear_r_2, whisker)
        
        # Place modified cat at C1
        self.place_at_grid(cat_input_2, "C1", scale_factor=0.6)

        # Create the new completely different hash
        hex_output_2 = Text("3b2e...", font_size=32, color=COLOR_HASH2)
        self.place_at_grid(hex_output_2, "C5")

        # Action: Highlight line, feed modified cat, transform to new hash
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HASH2)
        )
        self.play(FadeIn(cat_input_2))
        self.play(
            cat_input_2.animate.move_to(blender_center).scale(0.3).set_opacity(0),
            run_time=2,
            rate_func=slow_into
        )
        self.play(ReplacementTransform(hex_output_1, hex_output_2))
        self.wait(2)

        # Final reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
