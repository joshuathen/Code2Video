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

class Section3Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout
        # Fetching title and lecture lines from shared state
        self.setup_layout("Visualizing Vectors as Arrows", [
            "We represent vectors as arrows starting from the origin.",
            "The arrow's length shows the vector's magnitude.",
            "The tip points in the vector's specific direction."
        ])
        
        # Colors
        COLOR_BLUE = "#0000FF"
        COLOR_LIGHT_BLUE = "#ADD8E6"
        COLOR_GREEN = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE))
        
        # Background Axes for context
        # Fix: Issue 28: Scale factor from 0.9 to 0.8
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": True, "color": GREY_B},
            tips=False
        )
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.8)
        
        origin_label = Text("(0,0)", font_size=16, color=WHITE).next_to(axes.c2p(0,0), DL, buff=0.1)
        
        # The vector itself
        vector = Arrow(
            start=axes.c2p(0,0),
            end=axes.c2p(4,3),
            buff=0,
            color=COLOR_BLUE,
            stroke_width=6
        )
        
        self.play(Create(axes), Write(origin_label))
        self.play(GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight next lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_BLUE)
        )
        
        # Mathematical notation v = [4, 3]
        notation = MathTex(r"\vec{v} = [4, 3]", color=COLOR_BLUE, font_size=32)
        # Fix Issue 27: Use place_in_area for notation centering
        self.place_in_area(notation, 'A2', 'A4', scale_factor=1.0)
        
        # Bracket for magnitude (length)
        bracket = BraceBetweenPoints(axes.c2p(0,0), axes.c2p(4,3), color=COLOR_BLUE, buff=0.2)
        mag_label = Text("Magnitude", font_size=18, color=COLOR_BLUE)
        # Fix Issue 29: Use place_at_grid for magnitude label positioning
        self.place_at_grid(mag_label, 'F5', scale_factor=0.7)
        
        self.play(FadeIn(bracket), Write(mag_label))
        self.play(Write(notation))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight last lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LIGHT_BLUE)
        )
        
        # Arrowhead flash in light blue
        tip_highlight = Dot(axes.c2p(4,3), color=COLOR_LIGHT_BLUE, radius=0.15)
        
        # Visual components (4 right, 3 up) in green as per storyboard
        h_line = Line(axes.c2p(0,0), axes.c2p(4,0), color=COLOR_GREEN, stroke_width=4)
        v_line = Line(axes.c2p(4,0), axes.c2p(4,3), color=COLOR_GREEN, stroke_width=4)
        h_label = Text("4 right", font_size=16, color=COLOR_GREEN).next_to(h_line, DOWN, buff=0.1)
        v_label = Text("3 up", font_size=16, color=COLOR_GREEN).next_to(v_line, RIGHT, buff=0.1)
        
        self.play(FadeIn(tip_highlight))
        self.play(Indicate(tip_highlight, color=COLOR_LIGHT_BLUE, scale_factor=1.5))
        self.play(FadeOut(tip_highlight))
        
        self.play(Create(h_line), Write(h_label))
        self.play(Create(v_line), Write(v_label))
        self.wait(3)

        # Cleanup: reset last line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
