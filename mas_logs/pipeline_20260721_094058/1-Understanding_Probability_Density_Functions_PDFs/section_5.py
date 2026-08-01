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
        # Section data
        title_text = "Practical Application: The Waiting Game"
        lecture_lines = [
            "A uniform distribution has a flat, rectangular PDF.",
            "Consider a robot arriving within ten minutes.",
            "The height remains constant across the whole interval.",
            "Probability is simply base times height.",
            "A three-minute window gives a thirty percent chance."
        ]
        
        # Colors
        BLUE_COLOR = "#58C4DD"
        YELLOW_COLOR = "#FFFF00"
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(BLUE_COLOR)
        
        # Create Axes for the PDF
        # Adjusted area and scale factor per Issue 32
        axes = Axes(
            x_range=[0, 11, 1],
            y_range=[0, 0.2, 0.1],
            x_length=4.5,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE},
            tips=False
        )
        self.place_in_area(axes, "B1", "F5", scale_factor=0.9)
        
        # Rectangle for PDF
        # Rectangle from x=0 to x=10 with height y=0.1
        rect_width = axes.c2p(10, 0)[0] - axes.c2p(0, 0)[0]
        rect_height = axes.c2p(0, 0.1)[1] - axes.c2p(0, 0)[1]
        pdf_rect = Rectangle(
            width=rect_width,
            height=rect_height,
            stroke_color=BLUE_COLOR,
            fill_color=BLUE_COLOR,
            fill_opacity=0.2
        )
        pdf_rect.move_to(axes.c2p(5, 0.1/2))
        
        self.play(Create(axes), Create(pdf_rect), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_COLOR)
        
        # Labels for x=0 and x=10
        label_0 = Text("0", font_size=18).next_to(axes.c2p(0, 0), DOWN)
        label_10 = Text("10", font_size=18).next_to(axes.c2p(10, 0), DOWN)
        label_min = Text("minutes", font_size=14).next_to(axes.x_axis, RIGHT)
        
        self.play(Write(label_0), Write(label_10), FadeIn(label_min))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE_COLOR)
        
        # Label height 0.1
        label_h = Text("0.1", font_size=18).next_to(axes.c2p(0, 0.1), LEFT)
        h_line = DashedLine(axes.c2p(0, 0.1), axes.c2p(10, 0.1), color=BLUE_COLOR, stroke_width=2)
        
        self.play(Write(label_h), Create(h_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW_COLOR)
        
        # Shade portion x=0 to x=3
        shade_width = axes.c2p(3, 0)[0] - axes.c2p(0, 0)[0]
        shade_rect = Rectangle(
            width=shade_width,
            height=rect_height,
            fill_color=YELLOW_COLOR,
            fill_opacity=0.5,
            stroke_width=0
        )
        shade_rect.move_to(axes.c2p(1.5, 0.1/2))
        
        # Calculation text above - Updated per Issue 31
        calc_text = MathTex("3 \\times 0.1 = 0.3", color=YELLOW_COLOR, font_size=32)
        self.place_in_area(calc_text, 'A1', 'A3', scale_factor=0.8)
        
        self.play(FadeIn(shade_rect), Write(calc_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW_COLOR)
        
        # 30% Probability text - Updated per Issue 30
        prob_label = Text("30% Probability", color=YELLOW_COLOR, font_size=32)
        self.place_at_grid(prob_label, 'A5', scale_factor=0.8)
        
        self.play(Write(prob_label))
        # Pulse effect
        self.play(prob_label.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.play(prob_label.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.wait(2)
