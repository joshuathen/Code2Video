from manim import *

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
        # Setup the basic layout
        self.setup_layout("The Solution: Connecting Topology to Beads", [
            "We map necklace cuts to points on a sphere.",
            "A function measures the bead difference for one thief.",
            "For k bead types, k cuts ensure a perfect split.",
            "Two cuts always divide two bead types fairly.",
            "Topology guarantees a solution for any number of types."
        ])

        # Colors for lecture lines matching animations
        colors = [BLUE_A, GREEN_A, YELLOW_A, RED_A, PURPLE_A]

        # === Animation for Lecture Line 1: Mapping necklace cuts to sphere ===
        self.lecture[0].set_color(colors[0])
        
        # Represent the necklace as a unit interval [0, 1]
        necklace_line = Line(LEFT, RIGHT, color=WHITE, stroke_width=4)
        # Resolved Issue 34: Adjusted position and scale
        self.place_in_area(necklace_line, 'C2', 'C6', scale_factor=2.5)
        
        # Add bead densities (dots)
        beads = VGroup()
        for i in range(10):
            color = RED_A if i % 2 == 0 else BLUE_A
            dot = Dot(radius=0.08, color=color)
            dot.move_to(necklace_line.point_from_proportion(0.1 + i * 0.08))
            beads.add(dot)
        
        self.play(Create(necklace_line), FadeIn(beads))
        self.wait(1)

        # Transformation into a sphere
        sphere_base = Circle(radius=1.5, color=colors[0])
        sphere_lat = Ellipse(width=3, height=1, color=colors[0], stroke_opacity=0.5)
        sphere_long = Ellipse(width=1, height=3, color=colors[0], stroke_opacity=0.5)
        sphere_group = VGroup(sphere_base, sphere_lat, sphere_long)
        # Resolved Issue 35: Adjusted position and scale
        self.place_in_area(sphere_group, 'B3', 'E6', scale_factor=0.9)

        self.play(
            ReplacementTransform(necklace_line, sphere_base),
            FadeOut(beads),
            Create(sphere_lat),
            Create(sphere_long)
        )
        self.wait(1)

        # === Animation for Lecture Line 2: Function measures bead difference ===
        self.lecture[1].set_color(colors[1])
        
        # Defining a mapping for surplus
        func_label = Text("f(P) = (Diff1, Diff2)", color=colors[1], font_size=24)
        self.place_at_grid(func_label, "A4")
        
        # Highlight a point on the sphere representation
        point_p = Dot(color=colors[1])
        # Manually offset from center of sphere group
        point_p.move_to(sphere_group.get_center() + UP*0.8 + RIGHT*0.5)
        label_p = Text("P", color=colors[1], font_size=20).next_to(point_p, UR, buff=0.1)
        
        self.play(Write(func_label), FadeIn(point_p), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 3: For k bead types, k cuts ensure a perfect split ===
        self.lecture[2].set_color(colors[2])
        
        # Text to represent the generalization
        k_logic = Text("k types = k cuts", color=colors[2], font_size=24)
        self.place_at_grid(k_logic, "B5")
        
        self.play(Write(k_logic))
        self.wait(1)

        # === Animation for Lecture Line 4: Two cuts always divide two bead types fairly ===
        self.lecture[3].set_color(colors[3])
        
        # Indicator of a zero crossing (perfect split)
        zero_res = Text("f(P*) = (0, 0)", color=colors[3], font_size=24)
        self.place_at_grid(zero_res, "E4")
        
        # Solution point on sphere
        sol_point = Dot(color=colors[3])
        sol_point.move_to(sphere_group.get_center() + DOWN*0.8 + LEFT*0.5)
        sol_label = Text("Solution", color=colors[3], font_size=18).next_to(sol_point, DL, buff=0.1)
        
        self.play(
            ReplacementTransform(point_p, sol_point),
            ReplacementTransform(label_p, sol_label),
            Write(zero_res)
        )
        self.wait(1)

        # === Animation for Lecture Line 5: Topology guarantees a solution ===
        self.lecture[4].set_color(colors[4])
        
        # Show the necklace being cut at those two specific spots
        final_necklace = Line(LEFT, RIGHT, color=WHITE, stroke_width=4)
        self.place_in_area(final_necklace, "F2", "F6", scale_factor=2.5)
        
        final_beads = VGroup()
        for i in range(10):
            color = RED_A if i % 2 == 0 else BLUE_A
            dot = Dot(radius=0.08, color=color)
            dot.move_to(final_necklace.point_from_proportion(0.1 + i * 0.08))
            final_beads.add(dot)
            
        # Two cuts represented by vertical segments
        cut1 = Line(UP*0.3, DOWN*0.3, color=YELLOW_A).move_to(final_necklace.point_from_proportion(0.35))
        cut2 = Line(UP*0.3, DOWN*0.3, color=YELLOW_A).move_to(final_necklace.point_from_proportion(0.65))
        
        # Perfect equity indicator
        equity_box = SurroundingRectangle(VGroup(final_necklace, final_beads), color=colors[4], buff=0.2)
        
        self.play(
            FadeIn(final_necklace),
            FadeIn(final_beads),
            Create(cut1),
            Create(cut2)
        )
        self.play(Create(equity_box))
        self.wait(2)
