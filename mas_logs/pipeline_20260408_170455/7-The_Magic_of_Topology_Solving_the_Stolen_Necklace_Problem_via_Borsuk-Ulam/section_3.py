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
        # Initial Setup
        title_str = "The Core: The Borsuk-Ulam Theorem"
        lines = [
            "The Borsuk-Ulam theorem relates spheres to continuous functions.",
            "It states pairs of antipodal points map to same values.",
            "Imagine measuring temperature and pressure anywhere on Earth.",
            "Two opposite spots always share the exact same conditions.",
            "This holds true for any continuous mapping on spheres."
        ]
        self.setup_layout(title_str, lines)
        
        # Hide title initially for the "fade in" instruction later
        self.title.set_opacity(0)

        # === Pre-creation of Objects ===
        # 1. Earth Representation
        earth_circle = Circle(radius=1.2, color=BLUE_D, fill_opacity=0.2)
        # Add some "meridians" and "parallels" for a 3D sphere look
        meridian = Ellipse(width=0.6, height=2.4, color=BLUE_E, stroke_width=1)
        parallel = Ellipse(width=2.4, height=0.6, color=BLUE_E, stroke_width=1)
        earth = VGroup(earth_circle, meridian, parallel)
        # Fixed Issue 33: Adjusted scale_factor to 0.8 to avoid visual obstruction
        self.place_in_area(earth, 'B1', 'D3', scale_factor=0.8)

        # 2. Antipodal Points
        # Using left/right extremes for clarity
        p1_dot = Dot(color=YELLOW).move_to(earth_circle.get_left())
        p2_dot = Dot(color=YELLOW).move_to(earth_circle.get_right())
        points_group = VGroup(p1_dot, p2_dot)

        # 3. Temperature/Pressure Labels
        # Stage 1 labels - using Text for symbols
        t1_label = Text("T1", font_size=24, color=WHITE).next_to(p1_dot, UP, buff=0.1)
        t2_label = Text("T2", font_size=24, color=WHITE).next_to(p2_dot, UP, buff=0.1)
        p1_label = Text("P1", font_size=24, color=WHITE).next_to(p1_dot, DOWN, buff=0.1)
        p2_label = Text("P2", font_size=24, color=WHITE).next_to(p2_dot, DOWN, buff=0.1)
        
        # Stage 2 equality labels
        t_eq_label = Text("T1 = T2", font_size=24, color="#00FF00")
        p_eq_label = Text("P1 = P2", font_size=24, color="#00FFFF")
        # Fixed Issue 34: Scaled t_eq_label to 0.8
        self.place_at_grid(t_eq_label, 'E2', scale_factor=0.8)
        # Fixed Issue 32: Moved p_eq_label to 'E5' and scaled to 0.8 to prevent overlap
        self.place_at_grid(p_eq_label, 'E5', scale_factor=0.8)

        # 4. Mapping Arrow and Plane
        mapping_arrow = Arrow(
            start=earth.get_right() + RIGHT*0.2,
            end=self.grid['C4'] + RIGHT*1.0,
            buff=0.1,
            color=GRAY
        )
        # Axes labels added manually
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"include_tip": True}
        )
        x_lab = Text("Temp", font_size=14).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_lab = Text("Pres", font_size=14).next_to(axes.y_axis, UP, buff=0.1)
        plane_group = VGroup(axes, x_lab, y_lab)
        self.place_in_area(plane_group, 'B4', 'D6')
        
        target_point = Dot(axes.c2p(2.5, 3.5), color=YELLOW)
        target_coords = Text("(T, P)", font_size=20, color=YELLOW).next_to(target_point, UR, buff=0.1)

        # === Animation Sequence ===

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(earth), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(points_group))
        self.play(
            p1_dot.animate.scale(1.5),
            p2_dot.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Write(t1_label), Write(t2_label), Write(p1_label), Write(p2_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(
            ReplacementTransform(VGroup(t1_label, t2_label).copy(), t_eq_label),
            ReplacementTransform(VGroup(p1_label, p2_label).copy(), p_eq_label),
            run_time=2
        )
        
        # Fade in Title explicitly as per instructions
        self.play(self.title.animate.set_opacity(1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Show mapping arrow and 2D plane
        self.play(GrowArrow(mapping_arrow))
        self.play(Create(plane_group))
        self.play(FadeIn(target_point), Write(target_coords))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
