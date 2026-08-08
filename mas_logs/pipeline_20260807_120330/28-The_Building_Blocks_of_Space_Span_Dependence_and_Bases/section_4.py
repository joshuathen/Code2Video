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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "Linear Dependence: The Redundant Arrow"
        lecture_lines = [
            "Linear dependence means one vector is actually redundant.",
            "It can be built using the other available arrows.",
            "Adding a dependent vector doesn't expand the reachable Span.",
            "The third vector falls onto the existing 2D plane.",
            "It adds \"backup\" but no new directions for movement."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        v_color = WHITE
        w_color = WHITE
        u_color = "#FF0000"  # Red
        highlight_color = YELLOW
        plane_color = BLUE_B

        # === Animation for Lecture Line 1 ===
        # "Linear dependence means one vector is actually redundant."
        self.lecture[0].set_color(highlight_color)
        
        # Coordinate System setup
        # Optimized area per VideoCritic Issue 27
        axes = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"include_tip": True, "color": BLUE_E}
        )
        self.place_in_area(axes, "C3", "E5")
        self.add(axes)

        # Vectors: v=(1,0), w=(0,1), u=(1,1)
        v_arrow = Arrow(axes.c2p(0, 0), axes.c2p(1, 0), buff=0, color=v_color)
        w_arrow = Arrow(axes.c2p(0, 0), axes.c2p(0, 1), buff=0, color=w_color)
        u_arrow = Arrow(axes.c2p(0, 0), axes.c2p(1, 1), buff=0, color=u_color)

        v_label = MathTex("\\vec{v}", color=v_color)
        self.place_at_grid(v_label, "E5", scale_factor=0.8) # Issue 28
        
        w_label = MathTex("\\vec{w}", color=w_color)
        self.place_at_grid(w_label, "C2", scale_factor=0.8) # Issue 28

        u_label = MathTex("\\vec{u}", color=u_color)
        self.place_at_grid(u_label, "A5", scale_factor=0.8) # Issue 29

        self.play(GrowArrow(v_arrow), Write(v_label))
        self.play(GrowArrow(w_arrow), Write(w_label))
        self.play(GrowArrow(u_arrow), Write(u_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It can be built using the other available arrows."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color)
        )
        
        # Copy for tip-to-tail demonstration
        w_copy = w_arrow.copy().set_stroke(opacity=0.6)
        
        # Move w_copy to tip of v to demonstrate u = v + w
        shift_vec = axes.c2p(1, 0) - axes.c2p(0, 0)
        self.play(w_copy.animate.shift(shift_vec), run_time=2)
        
        # Flash u to show redundancy
        self.play(Flash(u_arrow, color=u_color, flash_radius=0.5))
        self.play(u_arrow.animate.set_stroke(width=8), run_time=0.5)
        self.play(u_arrow.animate.set_stroke(width=4), run_time=0.5)
        
        self.play(FadeOut(w_copy))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Adding a dependent vector doesn't expand the reachable Span."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )
        
        # Show span area (square [0,1]x[0,1])
        span_rect = Rectangle(
            width=axes.x_axis.get_unit_size(), 
            height=axes.y_axis.get_unit_size(),
            stroke_width=0,
            fill_color=plane_color,
            fill_opacity=0.3
        )
        span_rect.move_to(axes.c2p(0.5, 0.5))
        
        # Label redundant vector as DEPENDENT with asset (Issue 17)
        dep_text = Text("DEPENDENT", color=u_color, font_size=24)
        # Using SVGMobject from the provided asset path
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg").set_color(u_color)
        dep_group = VGroup(dep_text, based_icon).arrange(RIGHT, buff=0.2)
        # Place it above the u_label/vector
        self.place_in_area(dep_group, "A3", "B5", scale_factor=0.7)
        
        self.play(FadeIn(span_rect), Write(dep_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The third vector falls onto the existing 2D plane."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(highlight_color)
        )
        
        # Show the 'plane' as a larger rectangle to represent 2D space
        full_plane = Rectangle(
            width=axes.x_axis.get_unit_size() * 3, 
            height=axes.y_axis.get_unit_size() * 3,
            stroke_width=0,
            fill_color=plane_color,
            fill_opacity=0.15
        )
        full_plane.move_to(axes.c2p(1, 1))
        
        self.play(Transform(span_rect, full_plane))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "It adds \"backup\" but no new directions for movement."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(highlight_color)
        )
        
        # Pulse animation for u_arrow to emphasize redundancy/backup
        self.play(
            u_arrow.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        self.play(
            u_arrow.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(2)
