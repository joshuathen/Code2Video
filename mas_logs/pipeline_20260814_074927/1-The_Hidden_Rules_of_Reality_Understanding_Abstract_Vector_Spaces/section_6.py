from manim import *
import numpy as np

# Base class provided in instructions
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
        # Setup layout
        self.setup_layout(
            "The Power of Generality: Conclusion",
            [
                "Abstracting vectors allows us to solve diverse problems once.",
                "One theorem applies to arrows, functions, and data.",
                "This generality is the true power of linear algebra."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # Remote Icon (#ADD8E6)
        remote_body = RoundedRectangle(height=1.8, width=1.0, corner_radius=0.1, color="#ADD8E6", stroke_width=4)
        remote_btns = VGroup(*[Circle(radius=0.1, color="#ADD8E6", fill_opacity=0.8) for _ in range(4)])
        remote_btns.arrange_in_grid(rows=2, cols=2, buff=0.2).move_to(remote_body.get_center() + UP * 0.2)
        remote_label = Text("REMOTE", font_size=16, color="#ADD8E6").move_to(remote_body.get_center() + DOWN * 0.5)
        remote = VGroup(remote_body, remote_btns, remote_label)
        
        # Place at center of animation area
        self.place_in_area(remote, "C3", "D4", scale_factor=1.0)
        
        self.play(FadeIn(remote))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Bridge Icon (#FFFFFF) at A1
        bridge_base = Line(LEFT, RIGHT, color=WHITE)
        bridge_arch = Arc(radius=0.5, start_angle=0, angle=PI, color=WHITE)
        bridge = VGroup(bridge_base, bridge_arch).scale(0.5)
        self.place_at_grid(bridge, "A1", scale_factor=0.8)
        bridge_label = Text("Bridge", font_size=14, color=WHITE).next_to(bridge, DOWN, buff=0.1)
        
        # Speaker Icon (#FFFF00) at F1
        speaker_rect = Rectangle(height=0.6, width=0.4, color="#FFFF00")
        speaker_cone = Polygon(
            np.array([0, 0.2, 0]), 
            np.array([0.3, 0.4, 0]), 
            np.array([0.3, -0.4, 0]), 
            np.array([0, -0.2, 0]), 
            color="#FFFF00"
        ).next_to(speaker_rect, RIGHT, buff=0)
        speaker = VGroup(speaker_rect, speaker_cone).scale(0.5)
        self.place_at_grid(speaker, "F1", scale_factor=0.8)
        speaker_label = Text("Audio", font_size=14, color="#FFFF00").next_to(speaker, DOWN, buff=0.1)
        
        # Arrow/Data Icon at C6
        data_arrow = Arrow(LEFT, RIGHT, color=BLUE).scale(0.5)
        self.place_at_grid(data_arrow, "C6", scale_factor=0.8)
        data_label = Text("Arrows", font_size=14, color=BLUE).next_to(data_arrow, DOWN, buff=0.1)
        
        # Connections
        conn1 = Line(remote.get_top(), bridge.get_bottom(), color=GRAY_A, stroke_width=2)
        conn2 = Line(remote.get_bottom(), speaker.get_top(), color=GRAY_A, stroke_width=2)
        conn3 = Line(remote.get_right(), data_arrow.get_left(), color=GRAY_A, stroke_width=2)
        
        self.play(
            Create(conn1), Create(conn2), Create(conn3),
            FadeIn(bridge, bridge_label),
            FadeIn(speaker, speaker_label),
            FadeIn(data_arrow, data_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pulse Remote with #00FF00 glow
        pulse_circle = Circle(radius=0.1, color="#00FF00", stroke_width=10).move_to(remote.get_center())
        
        self.play(
            Indicate(remote, color="#00FF00", scale_factor=1.2),
            pulse_circle.animate.scale(15).set_opacity(0),
            rate_func=linear,
            run_time=2
        )
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
