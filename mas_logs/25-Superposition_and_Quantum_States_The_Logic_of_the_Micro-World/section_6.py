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
        # Setup title and lecture lines
        lecture_lines = [
            "Classical bits are limited, but qubits exploit superposition.",
            "This allows quantum computers to process information simultaneously.",
            "Superposition is the foundation of future computing power."
        ]
        self.setup_layout("Summary and Real-World Application", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1 (Color: #FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)

        # Classical Bit Graphic
        bit_box = Rectangle(width=1.5, height=1.5, color=WHITE)
        bit_label = Text("Classical Bit", font_size=18, color=WHITE)
        bit_value = Text("0 or 1", font_size=24, color=WHITE)
        bit_group = VGroup(bit_box, bit_value, bit_label).arrange(DOWN, buff=0.2)
        # Resolved Issue 46: Repositioned to B2-D3
        self.place_in_area(bit_group, "B2", "D3", scale_factor=0.8)

        # Qubit Sphere Graphic
        qubit_label = Text("Qubit", font_size=18, color=WHITE)
        sphere_outline = Circle(radius=1.0, color=WHITE)
        equator = Ellipse(width=2.0, height=0.5, color=GRAY, stroke_width=2).set_stroke(opacity=0.5)
        meridian = Ellipse(width=0.5, height=2.0, color=GRAY, stroke_width=2).set_stroke(opacity=0.5)
        
        # Qubit Vector (Rotating Arrow)
        qubit_arrow = Arrow(start=ORIGIN, end=[0.7, 0.7, 0], buff=0, color=WHITE)
        
        sphere_group = VGroup(sphere_outline, equator, meridian, qubit_arrow)
        qubit_container = VGroup(sphere_group, qubit_label).arrange(DOWN, buff=0.4)
        # Resolved Issue 47: Repositioned to B4-D5
        self.place_in_area(qubit_container, "B4", "D5", scale_factor=0.8)

        self.play(
            FadeIn(bit_group),
            FadeIn(qubit_container)
        )
        # Rotation of the vector to simulate superposition state representation
        self.play(
            Rotate(qubit_arrow, angle=PI*2, about_point=sphere_group.get_center(), run_time=3),
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2 (Color: #00FFFF)
        self.play(self.lecture[1].animate.set_color("#00FFFF"), run_time=0.5)

        # Glowing cyan aura around Qubit
        aura = Circle(radius=1.2, color="#00FFFF", stroke_width=15).set_opacity(0.3)
        aura.move_to(sphere_group.get_center())
        
        self.play(
            FadeIn(aura, scale=1.2),
            qubit_arrow.animate.set_color("#00FFFF"),
            sphere_outline.animate.set_color("#00FFFF")
        )
        self.play(
            aura.animate.scale(1.1).set_opacity(0.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 (Color: #FFFF00)
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)

        # Concluding Text
        conclusion_text = Text("The Foundation of Quantum Computing", font_size=24, color="#FFFF00")
        # Resolved Issue 48: Repositioned to E2-F5
        self.place_in_area(conclusion_text, "E2", "F5", scale_factor=1.0)

        self.play(Write(conclusion_text))
        
        # Final subtle glow pulse for the whole quantum system
        self.play(
            aura.animate.scale(1.2).set_opacity(0.4),
            conclusion_text.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
