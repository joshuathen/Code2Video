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
        # Define lecture lines
        lecture_lines = [
            'Map the sphere to the space of bead amounts.',
            'Borsuk-Ulam identifies a point where both shares match.',
            'This guarantees a perfect 50/50 split for every type.'
        ]
        
        self.setup_layout("The Elegant Solution", lecture_lines)
        
        # Color palette
        L1_COLOR = "#FFFF00"  # Yellow
        L2_COLOR = "#87CEEB"  # SkyBlue
        L3_COLOR = "#90EE90"  # LightGreen

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(L1_COLOR)
        
        # Coordinate System
        axes = Axes(
            x_range=[0, 1.2, 0.5],
            y_range=[0, 1.2, 0.5],
            x_length=3,
            y_length=3,
            axis_config={"color": L1_COLOR, "include_tip": True}
        )
        x_label = Text("Red Share", font_size=14, color=RED)
        y_label = Text("Green Share", font_size=14, color=GREEN)
        axes_labels = VGroup(
            x_label.next_to(axes.x_axis, RIGHT, buff=0.1),
            y_label.next_to(axes.y_axis, UP, buff=0.1)
        )
        axes_group = VGroup(axes, axes_labels)
        # Resolved Issue 38: Move axes_group to E2-F4
        self.place_in_area(axes_group, "E2", "F4", scale_factor=0.7)
        
        # Sphere from Asset
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/earth.svg]
        sphere = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/earth.svg")
        sphere.set_color(L1_COLOR)
        sphere_label = Text("Sphere of Cuts (S²)", font_size=16, color=L1_COLOR)
        sphere_group = VGroup(sphere, sphere_label.next_to(sphere, UP, buff=0.1))
        # Resolved Issue 37: Move sphere_group to C2-D4
        self.place_in_area(sphere_group, "C2", "D4", scale_factor=0.6)
        
        mapping_arrow = Arrow(sphere.get_bottom(), axes.get_top(), color=L1_COLOR, buff=0.2)
        mapping_text = Text("Mapping f", font_size=14, color=L1_COLOR).next_to(mapping_arrow, LEFT, buff=0.1)

        self.play(Create(sphere_group), run_time=1)
        self.play(Create(mapping_arrow), Write(mapping_text), run_time=1)
        self.play(Create(axes_group), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(L2_COLOR)
        
        # Antipodal points
        p1 = Dot(sphere.get_top(), color=L2_COLOR)
        p1_opp = Dot(sphere.get_bottom(), color=L2_COLOR)
        p1_label = Text("P", font_size=18, color=L2_COLOR).next_to(p1, UR, buff=0.05)
        p1_opp_label = Text("-P", font_size=18, color=L2_COLOR).next_to(p1_opp, DL, buff=0.05)
        
        # Target point in R2
        target_point = axes.c2p(0.5, 0.5)
        v_dot = Dot(target_point, color=L2_COLOR)
        v_label = Text("f(P) = f(-P)", font_size=18, color=L2_COLOR).next_to(v_dot, UR, buff=0.1)
        
        h_line = DashedLine(axes.c2p(0, 0.5), axes.c2p(0.5, 0.5), color=L2_COLOR)
        v_line = DashedLine(axes.c2p(0.5, 0), axes.c2p(0.5, 0.5), color=L2_COLOR)
        val_label = Text("0.5", font_size=14, color=L2_COLOR).next_to(target_point, DOWN, buff=0.1)
        
        self.play(FadeIn(p1, p1_opp, p1_label, p1_opp_label))
        self.play(TransformFromCopy(VGroup(p1, p1_opp), v_dot))
        self.play(Write(v_label), Create(h_line), Create(v_line), Write(val_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(L3_COLOR)
        
        # Necklace Visualization
        bead_colors = [RED, RED, GREEN, RED, GREEN, GREEN, RED, GREEN]
        necklace_line = Line(LEFT*2.5, RIGHT*2.5, color=GREY_A)
        beads = VGroup(*[Dot(radius=0.1, color=c) for c in bead_colors]).arrange(RIGHT, buff=0.3)
        
        cut1 = Line(UP*0.4, DOWN*0.4, color=L3_COLOR, stroke_width=4)
        cut2 = Line(UP*0.4, DOWN*0.4, color=L3_COLOR, stroke_width=4)
        
        necklace_group = VGroup(necklace_line, beads, cut1, cut2)
        self.place_in_area(necklace_group, "A1", "A6", scale_factor=0.8)
        
        # Initial cut positions
        cut1.move_to(beads[1].get_center() + RIGHT*0.1)
        cut2.move_to(beads[5].get_center() + RIGHT*0.1)
        
        self.play(FadeIn(necklace_line, beads))
        self.play(Create(cut1), Create(cut2))
        
        # Slide cuts to fair positions
        self.play(
            cut1.animate.move_to(beads[3].get_center() + LEFT*0.15),
            cut2.animate.move_to(beads[6].get_center() + RIGHT*0.15),
            run_time=2
        )
        
        # Resolved Issue 39: Move final_text to C5
        final_text = Text("Perfect Split!", font_size=20, color=L3_COLOR)
        self.place_at_grid(final_text, 'C5', scale_factor=1.0)
        
        success_rect = SurroundingRectangle(necklace_group, color=L3_COLOR, buff=0.1)
        self.play(Create(success_rect), Write(final_text))
        self.play(Indicate(v_dot, color=L3_COLOR))
        
        self.wait(3)
