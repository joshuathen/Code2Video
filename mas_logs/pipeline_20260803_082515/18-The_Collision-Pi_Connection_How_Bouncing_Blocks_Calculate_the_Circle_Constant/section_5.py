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

class Section5Scene(TeachingScene):
    def construct(self):
        # Define CYAN as it is not a standard top-level constant in Manim CE
        CYAN = "#00FFFF"
        
        self.setup_layout("The 'Circle' Trick: Scaling the Axes", [
            "We can scale the axes to simplify the math.",
            "This transformation turns the ellipse into a perfect circle.",
            "Now, physics behaves like simple, beautiful circular geometry."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Corresponding visual: White energy ellipse on velocity axes
        self.lecture[0].set_color(WHITE)
        
        # Create axes in the right-side area
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": GREY_B},
            tips=True
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        # Labels for velocity axes
        v1_label = MathTex("v_1", font_size=28, color=WHITE)
        v2_label = MathTex("v_2", font_size=28, color=WHITE)
        # Position labels at the ends of axes
        self.place_at_grid(v1_label, 'D6', scale_factor=0.8)
        self.place_at_grid(v2_label, 'A4', scale_factor=0.8)

        # Energy ellipse (m1*v1^2 + m2*v2^2 = E)
        ellipse = Ellipse(width=1.8, height=3.6, color=WHITE, stroke_width=4)
        self.place_in_area(ellipse, 'B2', 'E5')
        
        self.play(Create(axes), Write(v1_label), Write(v2_label))
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Corresponding visual: Squash the horizontal axis to form a circle
        self.lecture[1].set_color(YELLOW)
        
        # Transformation formula shown visually
        transform_tex = MathTex("v_1 \\rightarrow \\sqrt{m_1} v_1", font_size=28, color=YELLOW)
        self.place_at_grid(transform_tex, 'B6', scale_factor=0.7)

        # Perfect circle resulting from scaling transformation
        circle = Circle(radius=1.8, color=CYAN, stroke_width=4)
        self.place_in_area(circle, 'B2', 'E5')
        
        self.play(
            Transform(ellipse, circle),
            Write(transform_tex),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Corresponding visual: Highlight the new cyan circle
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CYAN)
        
        # Pulse the resulting circle to draw attention to its symmetry
        self.play(
            ellipse.animate.set_stroke(width=10),
            run_time=0.6
        )
        self.play(
            ellipse.animate.set_stroke(width=4),
            run_time=0.6
        )
        
        self.wait(2)
