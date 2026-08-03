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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        self.setup_layout("The Hook: The Tale of Two Navigators", [
            "Meet Alice and Bob exploring the same terrain.",
            "Alice uses a standard North-East grid map.",
            "Bob's map aligns with the tilted mountain range.",
            "They see the same treasure at different coordinates.",
            "Both are correct, using different basis languages."
        ])

        # === Animation for Lecture Line 1 ===
        # Meet Alice and Bob exploring the same terrain.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Terrain box as a visual container for the grids
        terrain_box = Rectangle(width=6, height=6, color=GREY_E, fill_opacity=0.05)
        self.place_in_area(terrain_box, 'A1', 'F6')
        
        # Load Alice and Bob icons
        alice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg").set_color(WHITE)
        bob_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bob.svg").set_color(BLUE)
        
        self.place_at_grid(alice_icon, 'C2', scale_factor=0.5)
        self.place_at_grid(bob_icon, 'C5', scale_factor=0.5)
        
        self.play(FadeIn(terrain_box))
        self.play(FadeIn(alice_icon), FadeIn(bob_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Alice uses a standard North-East grid map.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Fixed scale based on Issue 31
        alice_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": WHITE, "stroke_width": 1, "stroke_opacity": 0.2},
            axis_config={"stroke_color": WHITE, "stroke_width": 2},
            tips=False
        )
        # Apply Issue 31 fix: scale_factor=0.8
        self.place_in_area(alice_grid, 'A1', 'F6', scale_factor=0.8)
        
        alice_label = Text("Alice's Grid", font_size=18, color=WHITE)
        self.place_at_grid(alice_label, 'A2', scale_factor=1.0)
        
        self.play(
            Create(alice_grid),
            Write(alice_label),
            alice_icon.animate.scale(0.5).next_to(alice_label, LEFT, buff=0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Bob's map aligns with the tilted mountain range.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )
        
        # Bob's grid: Basis vectors [2, 1] and [1, 1]
        bob_matrix = [[2, 1], [1, 1]]
        bob_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE, "stroke_width": 1, "stroke_opacity": 0.3},
            axis_config={"stroke_color": BLUE, "stroke_width": 2},
            tips=False
        ).apply_matrix(bob_matrix)
        
        # Apply Issue 29 fix: scale_factor=0.7
        self.place_in_area(bob_grid, 'A1', 'F6', scale_factor=0.7)
        
        bob_label = Text("Bob's Grid", font_size=18, color=BLUE)
        # Apply Issue 30 fix: 'A5', scale_factor=0.8
        self.place_at_grid(bob_label, 'A5', scale_factor=0.8)
        
        self.play(
            Create(bob_grid),
            Write(bob_label),
            bob_icon.animate.scale(0.5).next_to(bob_label, RIGHT, buff=0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # They see the same treasure at different coordinates.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(RED)
        )
        
        # Point P at Alice's (3, 2). 
        # Coordinates in Alice's grid space are (3, 2).
        # Alice's grid was placed at area A1-F6 and scaled by 0.8.
        # Manim's NumberPlane units are 1 scene unit per grid unit by default.
        # We need the point relative to the grid's center.
        p_pos = alice_grid.c2p(3, 2)
        point_p = Dot(point=p_pos, color=RED, radius=0.1)
        p_label = Text("P", font_size=24, color=RED).next_to(point_p, UR, buff=0.1)
        
        alice_coords = Text("(3, 2)", font_size=20, color=WHITE).next_to(point_p, UP, buff=0.3)
        bob_coords = Text("(1, 1)", font_size=20, color=BLUE).next_to(point_p, DOWN, buff=0.3)
        
        self.play(FadeIn(point_p, scale=0.5), Write(p_label))
        self.play(Write(alice_coords))
        self.play(Write(bob_coords))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Both are correct, using different basis languages.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(GREEN)
        )
        
        # Highlight both coordinate labels
        highlight_box = SurroundingRectangle(VGroup(alice_coords, bob_coords), color=YELLOW, buff=0.1)
        
        self.play(
            Create(highlight_box),
            alice_coords.animate.set_color(YELLOW),
            bob_coords.animate.set_color(YELLOW)
        )
        self.wait(2)
        self.play(FadeOut(highlight_box))
        self.wait(2)
