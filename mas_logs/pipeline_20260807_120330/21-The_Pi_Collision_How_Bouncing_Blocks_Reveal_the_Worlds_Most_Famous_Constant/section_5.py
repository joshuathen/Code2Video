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
        # Setup lines and title
        title = "Mapping Collisions to Arcs"
        lines = [
            "- Each collision corresponds to a point on the circle.",
            "- Bouncing between wall and block creates fixed arcs.",
            "- The arc length depends on the mass ratio.",
            "- We count how many arcs fit in the circle.",
            "- This counting process mirrors the definition of Pi."
        ]
        self.setup_layout(title, lines)

        # Common Colors
        CIRCLE_COLOR = BLUE_D
        ARC_COLOR = YELLOW
        POINT_COLOR = RED
        TEXT_HIGHLIGHT = YELLOW
        CHORD_COLOR = WHITE

        # Assets
        block_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(TEXT_HIGHLIGHT)
        
        # Phase space circle
        circle = Circle(radius=1.5, color=CIRCLE_COLOR, stroke_width=2)
        self.place_in_area(circle, "B2", "E5")
        circle_center = circle.get_center()
        
        # Axes for context
        axes = VGroup(
            Line(circle_center + LEFT * 1.8, circle_center + RIGHT * 1.8, color=GRAY, stroke_opacity=0.3),
            Line(circle_center + DOWN * 1.8, circle_center + UP * 1.8, color=GRAY, stroke_opacity=0.3)
        )
        
        # Physical assets (symbolic)
        wall = SVGMobject(wall_asset_path, height=0.6, color=GRAY)
        block = SVGMobject(block_asset_path, height=0.5, color=WHITE)
        self.place_at_grid(wall, "A1", scale_factor=1.0)
        self.place_at_grid(block, "A2", scale_factor=1.0)
        
        # Collision point P0
        start_angle = PI / 6
        p0_pos = circle.point_at_angle(start_angle)
        p0 = Dot(p0_pos, color=POINT_COLOR, radius=0.08)
        p0_label = MathTex("P_0", font_size=20, color=POINT_COLOR)
        p0_label.next_to(p0, UR, buff=0.1)
        
        self.play(Create(circle), Create(axes), FadeIn(wall), FadeIn(block), run_time=1.2)
        self.play(FadeIn(p0), FadeIn(p0_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(TEXT_HIGHLIGHT)
        
        # Define theta and the next point
        theta_val = 40 * DEGREES
        p1_angle = start_angle + theta_val
        p1_pos = circle.point_at_angle(p1_angle)
        p1 = Dot(p1_pos, color=POINT_COLOR, radius=0.08)
        p1_label = MathTex("P_1", font_size=20, color=POINT_COLOR)
        p1_label.next_to(p1, UP, buff=0.1)
        
        # Arc and Chord
        jump_arc = Arc(radius=1.5, start_angle=start_angle, angle=theta_val, color=ARC_COLOR).move_to(circle_center)
        chord = Line(p0_pos, p1_pos, color=CHORD_COLOR, stroke_width=2)
        
        self.play(
            Create(chord),
            Create(jump_arc),
            TransformFromCopy(p0, p1),
            FadeIn(p1_label),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TEXT_HIGHLIGHT)
        
        # Show theta angle visuals
        theta_line1 = Line(circle_center, p0_pos, color=WHITE, stroke_width=1)
        theta_line2 = Line(circle_center, p1_pos, color=WHITE, stroke_width=1)
        theta_arc = Arc(radius=0.4, start_angle=start_angle, angle=theta_val).move_to(circle_center)
        
        label_pos = circle_center + 0.6 * np.array([np.cos(start_angle + theta_val/2), np.sin(start_angle + theta_val/2), 0])
        theta_label = MathTex(r"\theta", font_size=24, color=WHITE).move_to(label_pos)
        
        # Formula for theta - Fix for Issue 32
        formula = MathTex(r"\theta = 2 \arctan\left(\sqrt{\frac{m}{M}}\right)", font_size=24, color=ARC_COLOR)
        self.place_at_grid(formula, "A5", scale_factor=0.8)
        
        self.play(
            Create(theta_line1),
            Create(theta_line2),
            Create(theta_arc),
            Write(theta_label),
            Write(formula)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(TEXT_HIGHLIGHT)
        
        # Fade out clutter
        self.play(
            FadeOut(p0_label), 
            FadeOut(p1_label), 
            FadeOut(theta_line1), 
            FadeOut(theta_line2), 
            FadeOut(theta_label), 
            FadeOut(theta_arc),
            FadeOut(chord)
        )
        
        # Loop to create more arcs
        num_additional = 4
        current_angle = p1_angle
        for i in range(num_additional):
            next_angle = current_angle + theta_val
            new_arc = Arc(radius=1.5, start_angle=current_angle, angle=theta_val, color=ARC_COLOR).move_to(circle_center)
            new_point = Dot(circle.point_at_angle(next_angle), color=POINT_COLOR, radius=0.08)
            self.play(Create(new_arc), FadeIn(new_point), run_time=0.4)
            current_angle = next_angle
            
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(TEXT_HIGHLIGHT)
        
        # Summary Pi note - Fix for Issue 33
        total_angle = (num_additional + 1) * theta_val
        full_path_arc = Arc(radius=1.6, start_angle=start_angle, angle=total_angle, color=WHITE, stroke_width=3).move_to(circle_center)
        pi_note = MathTex(r"N \cdot \theta \approx \pi", font_size=28, color=WHITE)
        self.place_at_grid(pi_note, "F5", scale_factor=0.8)
        
        self.play(Create(full_path_arc))
        self.play(Write(pi_note))
        self.wait(3)

        # Global Cleanup
        self.play(*[FadeOut(m) for m in self.mobjects])
