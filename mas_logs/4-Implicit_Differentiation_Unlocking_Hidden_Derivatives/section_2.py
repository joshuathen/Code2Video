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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initialize the layout
        lecture_lines = [
            "Solving for y creates messy, multiple equations.",
            "A circle splits into complex top and bottom halves.",
            "We must derive without untangling the equation first."
        ]
        self.setup_layout("Why Not Just Solve for Y?", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display 'x^3 + y^3 = 6xy' in white #FFFFFF with a red #FF5555 question mark.
        eq1 = Text(r"x^3 + y^3 = 6xy", color="#FFFFFF")
        self.place_in_area(eq1, "A2", "B5", scale_factor=1.2)
        
        q_mark = Text("?", color="#FF5555", font_size=60)
        self.place_at_grid(q_mark, "B6")

        self.play(Write(eq1), Write(q_mark))
        # Highlight lecture line 1 in Red to match the "problem" elements
        self.play(self.lecture[0].animate.set_color("#FF5555"))
        self.wait(1)

        # Morph equation into a mess of square roots to show difficulty.
        # Fixed: Changed MathTex to Text to avoid FileNotFoundError for latex binary.
        messy_eq = Text(
            r"y = \sqrt[3]{-\frac{x^3}{2} + \sqrt{\frac{x^6}{4} - 8x^3}} + \dots",
            color="#FF5555"
        )
        self.place_in_area(messy_eq, "A1", "B6", scale_factor=0.7)
        
        self.play(
            Transform(eq1, messy_eq), 
            FadeOut(q_mark)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transition to circle to show another "Pain Point"
        self.play(FadeOut(eq1))
        
        # Draw a circle #FFFFFF and split it into two halves.
        circle_full = Circle(radius=1.0, color="#FFFFFF")
        self.place_in_area(circle_full, "C3", "D4")
        
        self.play(Create(circle_full))
        # Color matching for lecture line 2 (Blue for the top branch)
        self.play(self.lecture[1].animate.set_color("#00AAFF"))
        self.wait(1)

        # Split circle into two halves.
        top_half = Arc(radius=1.0, start_angle=0, angle=PI, color="#00AAFF")
        bottom_half = Arc(radius=1.0, start_angle=PI, angle=PI, color="#FFAA00")
        
        # Position them initially on top of the original circle
        self.place_in_area(top_half, "C3", "D4")
        self.place_in_area(bottom_half, "C3", "D4")

        self.play(
            FadeOut(circle_full),
            FadeIn(top_half),
            FadeIn(bottom_half)
        )
        
        # Color top half blue #00AAFF and bottom half orange #FFAA00 and split them.
        self.play(
            top_half.animate.move_to(self.grid["C4"]),
            bottom_half.animate.move_to(self.grid["D4"])
        )
        
        # Branch equations to show complexity
        # Fixed: Changed MathTex to Text to avoid FileNotFoundError for latex binary.
        y_top = Text(r"y = \sqrt{r^2 - x^2}", color="#00AAFF", font_size=24)
        y_bottom = Text(r"y = -\sqrt{r^2 - x^2}", color="#FFAA00", font_size=24)
        self.place_at_grid(y_top, "C5")
        self.place_at_grid(y_bottom, "D5")
        
        self.play(Write(y_top), Write(y_bottom))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Flash 'Messy Branches' text in red #FF5555.
        messy_branches = Text("Messy Branches!", color="#FF5555", font_size=36)
        self.place_in_area(messy_branches, "F2", "F5")
        
        self.play(self.lecture[2].animate.set_color("#FF5555"))
        self.play(Flash(messy_branches, color="#FF5555"))
        self.play(Write(messy_branches))
        self.wait(2)
