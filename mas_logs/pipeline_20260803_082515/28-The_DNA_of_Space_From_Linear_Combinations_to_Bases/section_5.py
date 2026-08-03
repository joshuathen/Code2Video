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
        lecture_lines = [
            "A basis is a minimalist toolkit for a space.",
            "It must span the space without any redundant vectors.",
            "North and East form a perfect basis for 2D.",
            "This efficient set defines our entire coordinate system."
        ]
        self.setup_layout("The Basis: The Efficient Toolkit", lecture_lines)

        # Colors for lines
        colors = [YELLOW, RED, BLUE, GREEN]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Grid Background
        grid_back = NumberPlane(
            x_range=[0, 4, 1], y_range=[0, 4, 1], 
            background_line_style={"stroke_color": GRAY, "stroke_width": 1}
        ).scale(0.6)
        self.place_in_area(grid_back, "B2", "E5")
        
        # Compass Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg]
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, "B1", scale_factor=0.5)
        
        # Vectors: North and East
        v_north = Arrow(start=grid_back.c2p(0,0), end=grid_back.c2p(0,2), buff=0, color=YELLOW)
        v_east = Arrow(start=grid_back.c2p(0,0), end=grid_back.c2p(2,0), buff=0, color=BLUE)
        
        label_north = Text("North", font_size=16, color=YELLOW)
        label_east = Text("East", font_size=16, color=BLUE)
        
        # Positioning fixes from Issues 29 and 30
        self.place_at_grid(label_north, 'A2', scale_factor=1.0)
        self.place_at_grid(label_east, 'E6', scale_factor=1.0)
        
        labels_basis_1 = VGroup(
            Text("Linearly Independent", font_size=18, color=WHITE),
            Text("Spanning 2D", font_size=18, color=WHITE)
        ).arrange(DOWN)
        # Positioning fix from Issue 28
        self.place_in_area(labels_basis_1, 'A2', 'A5', scale_factor=0.8)

        self.play(Create(grid_back), FadeIn(compass))
        self.play(GrowArrow(v_north), GrowArrow(v_east))
        self.play(Write(label_north), Write(label_east))
        self.play(FadeIn(labels_basis_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(colors[1])
        )
        
        # Add North-East (Redundant)
        v_ne = Arrow(start=grid_back.c2p(0,0), end=grid_back.c2p(2,2), buff=0, color=RED)
        label_redundant = Text("Redundant!", font_size=24, color=RED)
        self.place_at_grid(label_redundant, "C5", scale_factor=1.0)
        
        self.play(FadeOut(labels_basis_1))
        self.play(GrowArrow(v_ne), Write(label_redundant))
        self.wait(1)
        
        # Show Insufficient Span (Only North)
        label_insufficient = Text("Insufficient Span", font_size=20, color=RED)
        self.place_at_grid(label_insufficient, "B5", scale_factor=1.0)
        
        self.play(
            FadeOut(v_ne), FadeOut(label_redundant),
            FadeOut(v_east), FadeOut(label_east)
        )
        self.play(Write(label_insufficient))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(colors[2])
        )
        
        # Restore North and East
        self.play(
            FadeOut(label_insufficient),
            FadeIn(v_east), FadeIn(label_east)
        )
        
        perfect_label = Text("Perfect Toolkit", font_size=20, color=BLUE)
        self.place_at_grid(perfect_label, "A4", scale_factor=1.0)
        self.play(Write(perfect_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(colors[3])
        )
        
        # Highlight Basis
        self.play(
            v_north.animate.set_color(WHITE),
            v_east.animate.set_color(WHITE),
            FadeOut(perfect_label)
        )
        
        final_label = Text("THE BASIS", font_size=32, color=WHITE).set_stroke(WHITE, width=1)
        self.place_at_grid(final_label, "B4", scale_factor=1.0)
        
        # Toolkit Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/toolkit.svg]
        toolkit_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/toolkit.svg")
        self.place_at_grid(toolkit_icon, "D4", scale_factor=0.6)
        
        self.play(Write(final_label), FadeIn(toolkit_icon))
        self.play(Indicate(v_north), Indicate(v_east))
        self.wait(2)
