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
        # Setup layout
        self.setup_layout("Solving the Problem: The k-Color Solution", [
            "For two bead types, two cuts are always enough.",
            "The theorem finds where bead differences are zero.",
            "This logic extends to any number of bead varieties.",
            "For k types, k cuts guarantee a fair split.",
            "Topology proves a perfect solution always exists."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color(YELLOW)
        
        # Create necklace and beads as a group at origin first
        necklace_circle = Circle(radius=1.0, color=GRAY_A)
        bead_colors = [RED, GREEN, RED, GREEN, GREEN, RED, GREEN, RED]
        beads = VGroup()
        for i, color in enumerate(bead_colors):
            angle = i * (TAU / 8)
            dot = Dot(color=color, radius=0.1)
            dot.shift(RIGHT * np.cos(angle) + UP * np.sin(angle))
            beads.add(dot)
        
        necklace_group = VGroup(necklace_circle, beads)
        # Place the necklace group in the left-ish area of the right side (B2 area)
        self.place_in_area(necklace_group, "B1", "D3", scale_factor=1.0)
        necklace_center = necklace_group.get_center()
        
        # Diameter representing two cuts
        diameter = Line(
            necklace_center + UP * 1.2, 
            necklace_center + DOWN * 1.2, 
            color=WHITE, 
            stroke_width=5
        )
        
        self.play(Create(necklace_circle), Create(beads))
        self.play(Create(diameter))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Difference space axes
        axes = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=2.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": BLUE_C}
        )
        self.place_in_area(axes, "B4", "D6", scale_factor=0.8)
        axes_origin = axes.get_center() # This matches a grid center
        
        # Vector labels - placed near grid points
        ruby_label = Text("Ruby Diff", font_size=14).move_to(self.grid["D6"] + DOWN * 0.3)
        emerald_label = Text("Emerald Diff", font_size=14).move_to(self.grid["B4"] + LEFT * 0.5).rotate(PI/2)
        
        # Initial difference vector (not zero)
        vector_end = axes_origin + RIGHT * 0.7 + UP * 0.7
        diff_vector = Arrow(axes_origin, vector_end, buff=0, color=PINK, stroke_width=4)
        vector_coords = Text("(x, y)", font_size=18, color=PINK).move_to(vector_end + UR * 0.2)

        self.play(Create(axes), Write(ruby_label), Write(emerald_label))
        self.play(GrowArrow(diff_vector), FadeIn(vector_coords))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Moving towards zero - diameter rotates, vector moves
        mid_vector_end = axes_origin + RIGHT * 0.3 + DOWN * 0.4
        self.play(
            Rotate(diameter, angle=PI/3, about_point=necklace_center),
            diff_vector.animate.put_start_and_end_on(axes_origin, mid_vector_end),
            vector_coords.animate.move_to(mid_vector_end + DR * 0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Reach (0,0) - perfect split
        self.play(
            Rotate(diameter, angle=PI/4, about_point=necklace_center),
            diff_vector.animate.put_start_and_end_on(axes_origin, axes_origin),
            FadeOut(vector_coords),
            run_time=2
        )
        
        # Highlight final cuts in gold
        diameter.set_color("#FFD700")
        cut1 = Star(n=5, color="#FFD700", fill_opacity=1).scale(0.12).move_to(diameter.get_start())
        cut2 = Star(n=5, color="#FFD700", fill_opacity=1).scale(0.12).move_to(diameter.get_end())
        
        self.play(Indicate(diameter), FadeIn(cut1), FadeIn(cut2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Solution proof
        proof_text = Text("Solution Exists", font_size=20, color="#FFD700").move_to(self.grid["E5"])
        self.play(Write(proof_text))
        self.play(Circumscribe(necklace_group))
        self.wait(2)
