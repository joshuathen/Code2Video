from manim import *
import scipy.special
import pathlib

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
        # Initialization
        title = "The Big Picture: The Gamma Function"
        lines = [
            "The Gamma Function maps factorials onto a continuous curve.",
            "This graph proves zero factorial fits a natural progression.",
            "Visually, the curve hits one at the zero position."
        ]
        self.setup_layout(title, lines)

        # Define axes and plot parameters
        # Use label_constructor=Text to avoid dependency on 'latex'
        axes = Axes(
            x_range=[0, 3.5, 1],
            y_range=[0, 7, 1],
            x_length=4.5,
            y_length=5,
            axis_config={
                "color": WHITE, 
                "include_numbers": True, 
                "font_size": 20,
                "label_constructor": Text
            },
            tips=False
        )
        self.place_in_area(axes, "A1", "F6")

        # === Animation for Lecture Line 1 ===
        # Show discrete blue points for 1!, 2!, 3!
        self.play(self.lecture[0].animate.set_color("#0000FF"))
        
        # Points: (1, 1), (2, 2), (3, 6)
        p1 = Dot(axes.c2p(1, 1), color="#0000FF")
        p2 = Dot(axes.c2p(2, 2), color="#0000FF")
        p3 = Dot(axes.c2p(3, 6), color="#0000FF")
        
        # Using Text instead of MathTex to avoid 'latex' dependency
        l1 = Text("1!", font_size=20, color="#0000FF").next_to(p1, UR, buff=0.1)
        l2 = Text("2!", font_size=20, color="#0000FF").next_to(p2, UR, buff=0.1)
        l3 = Text("3!", font_size=20, color="#0000FF").next_to(p3, UL, buff=0.1)

        self.play(Create(axes))
        self.play(
            FadeIn(p1, l1),
            FadeIn(p2, l2),
            FadeIn(p3, l3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw the Gamma Function curve
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # Gamma curve: y = Gamma(x + 1)
        gamma_curve = axes.plot(
            lambda x: scipy.special.gamma(x + 1),
            x_range=[0, 3.1],
            color="#FF00FF"
        )
        
        gamma_label = Text("Γ(x+1)", font_size=24, color="#FF00FF")
        self.place_at_grid(gamma_label, "B5")

        self.play(Create(gamma_curve), FadeIn(gamma_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pulsing dot at (0, 1) for 0! = 1
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        zero_factorial_dot = Dot(axes.c2p(0, 1), color="#FFFFFF")
        zero_label = Text("0! = 1", font_size=24, color="#FFFFFF").next_to(zero_factorial_dot, LEFT, buff=0.2)
        
        self.play(FadeIn(zero_factorial_dot), Write(zero_label))
        
        # Pulsing animation
        self.play(
            zero_factorial_dot.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=0.5
        )
        self.play(
            zero_factorial_dot.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=0.5
        )
        
        self.wait(2)
