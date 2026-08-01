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
        # 1. Setup layout with Section 6 content
        title = "The Million Dollar Mystery: The Riemann Hypothesis"
        lines = [
            "Zeros are points where the function's output is zero.", 
            "Non-trivial zeros seem to align on one critical line.", 
            "Riemann hypothesized they all share a real part of half.", 
            "Proving this would solve mathematics' greatest unsolved mystery.", 
            "Every zero found so far sits exactly on this line."
        ]
        self.setup_layout(title, lines)

        # Pre-define Coordinate System in the right half grid area
        axes = NumberPlane(
            x_range=[-8, 2, 2],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "stroke_opacity": 0.5},
            background_line_style={"stroke_opacity": 0.1}
        )
        # Issue 53: Scaled axes to avoid clipping
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.85)

        # === Animation for Lecture Line 1 ===
        # "Zeros are points where the function's output is zero."
        self.lecture[0].set_color("#0000FF")
        
        blue_dots = VGroup(*[
            Dot(axes.c2p(x, 0), color="#0000FF", radius=0.08)
            for x in [-2, -4, -6]
        ])
        
        self.play(Create(axes), run_time=1.5)
        self.play(FadeIn(blue_dots, shift=UP), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Non-trivial zeros seem to align on one critical line."
        self.lecture[1].set_color("#555555")
        
        strip_width = axes.x_axis.get_unit_size() * 1
        strip_height = axes.y_axis.get_unit_size() * 5.5
        critical_strip = Rectangle(
            width=strip_width,
            height=strip_height,
            fill_color="#555555",
            fill_opacity=0.4,
            stroke_width=0
        ).move_to(axes.c2p(0.5, 0))
        
        self.play(FadeIn(critical_strip), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Riemann hypothesized they all share a real part of half."
        self.lecture[2].set_color("#FFD700")
        
        critical_line = Line(
            axes.c2p(0.5, -2.75),
            axes.c2p(0.5, 2.75),
            color="#FFD700",
            stroke_width=6
        )
        
        self.play(Create(critical_line), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Proving this would solve mathematics' greatest unsolved mystery."
        self.lecture[3].set_color("#FFFFFF")
        
        # Issue 36: mystery.svg asset integration
        mystery_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/mystery.svg", color="#FFFFFF")
        # Issue 52: Positioning at A6 to avoid visual clutter
        self.place_at_grid(mystery_icon, 'A6', scale_factor=0.7)
        
        self.play(FadeIn(mystery_icon))
        self.play(
            mystery_icon.animate.scale(1.2), 
            rate_func=there_and_back, 
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Every zero found so far sits exactly on this line."
        self.lecture[4].set_color("#FF0000")
        
        # Representative coordinates for non-trivial zeros
        zero_y_positions = [1.5, 2.4, -0.5, -2.0]
        
        for y_pos in zero_y_positions:
            target_pt = axes.c2p(0.5, y_pos)
            red_dot = Dot(target_pt, color="#FF0000", radius=0.07)
            
            # Issue 36: radar.svg asset integration for radar pings
            ping = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/radar.svg", color="#FF0000")
            ping.scale(0.1).move_to(target_pt)
            
            self.play(
                FadeIn(red_dot, scale=0.5),
                ping.animate.scale(10).set_opacity(0),
                run_time=0.8,
                rate_func=linear
            )
            self.remove(ping)
            
        self.wait(2)
