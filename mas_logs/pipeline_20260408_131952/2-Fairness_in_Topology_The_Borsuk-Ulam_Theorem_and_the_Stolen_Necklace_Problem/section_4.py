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

class Section4Scene(TeachingScene):
    def construct(self):
        # Define line colors
        line_colors = ["#FFD700", "#87CEEB", "#FFD700", "#87CEEB", "#FFD700"]
        alice_color = "#FFD700"
        bob_color = "#87CEEB"
        
        self.setup_layout(
            "Mapping the Necklace to a Sphere", 
            [
                "We can represent necklace partitions using a sphere's surface.", 
                "Each point on the sphere defines specific cut locations.", 
                "Moving to the opposite point swaps Alice and Bob's portions.", 
                "This maps the physical problem into a topological space.", 
                "The number of bead types determines the sphere's dimension."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(line_colors[0])
        
        # Issue 27 & 36: Asset integration and positioning
        necklace_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg")
        self.place_in_area(necklace_asset, "A2", "A5", scale_factor=0.9)
        
        self.play(DrawBorderThenFill(necklace_asset))
        self.wait(1)
        
        # Transform necklace into a circular loop (S1)
        necklace_loop = Circle(radius=1.2, color=WHITE).set_stroke(opacity=0.3)
        self.place_in_area(necklace_loop, "C2", "E5")
        
        # Visual representation of beads on the circle
        bead_colors = [RED, BLUE, GREEN, RED, BLUE, GREEN, RED, BLUE]
        beads_on_circle = VGroup()
        for i, color in enumerate(bead_colors):
            angle = i * (TAU / len(bead_colors))
            dot = Dot(radius=0.08, color=color)
            dot.move_to(necklace_loop.point_at_angle(angle))
            beads_on_circle.add(dot)
            
        self.play(
            ReplacementTransform(necklace_asset, beads_on_circle),
            Create(necklace_loop)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(line_colors[1])
        
        # Sphere depth representation
        sphere_depth = Ellipse(width=2.4, height=0.8, color=WHITE).set_stroke(opacity=0.2)
        sphere_depth.move_to(necklace_loop.get_center())
        
        # Highlighting a point on the "sphere"
        cut_point = Dot(color=YELLOW).move_to(necklace_loop.point_at_angle(PI/4))
        cut_label = Text("Cut Positions", font_size=16, color=YELLOW)
        
        # Issue 35: Reposition cut_label to C6
        self.place_at_grid(cut_label, "C6", scale_factor=0.8)

        self.play(Create(sphere_depth))
        self.play(FadeIn(cut_point), Write(cut_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(line_colors[2])
        
        # Visual segments for Alice and Bob
        arc_alice = Arc(radius=1.2, start_angle=PI/4, angle=PI, color=alice_color, stroke_width=6)
        arc_bob = Arc(radius=1.2, start_angle=PI/4 + PI, angle=PI, color=bob_color, stroke_width=6)
        arc_alice.move_to(necklace_loop.get_center())
        arc_bob.move_to(necklace_loop.get_center())
        
        # Mini asset representation during swap (Issue 27)
        necklace_mini = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg")
        self.place_at_grid(necklace_mini, "A6", scale_factor=0.4)
        
        self.play(Create(arc_alice), Create(arc_bob), FadeIn(necklace_mini))
        self.wait(1)
        
        # Antipodal point position
        antipodal_pos = necklace_loop.point_at_angle(PI/4 + PI)
        
        # Swapped state
        new_arc_alice = Arc(radius=1.2, start_angle=PI/4 + PI, angle=PI, color=alice_color, stroke_width=6)
        new_arc_bob = Arc(radius=1.2, start_angle=PI/4, angle=PI, color=bob_color, stroke_width=6)
        new_arc_alice.move_to(necklace_loop.get_center())
        new_arc_bob.move_to(necklace_loop.get_center())

        self.play(
            cut_point.animate.move_to(antipodal_pos),
            Transform(arc_alice, new_arc_alice),
            Transform(arc_bob, new_arc_bob),
            necklace_mini.animate.set_color(WHITE).scale(1.1).set_color(bob_color) # Indicating Bob's gain
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(line_colors[3])
        
        pulse_sphere = necklace_loop.copy().set_color(BLUE).set_stroke(width=10, opacity=0.5)
        self.play(pulse_sphere.animate.scale(1.2).set_opacity(0), run_time=1.5)
        self.remove(pulse_sphere, necklace_mini)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(line_colors[4])
        
        # Issue 34: Reposition dimension text to F3-F5
        dimension_text = Text("S^{k-1}", color=WHITE)
        desc_text = Text("Sphere Dimension", font_size=18, color=WHITE)
        dim_group = VGroup(dimension_text, desc_text).arrange(DOWN, buff=0.2)
        self.place_in_area(dim_group, "F3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(dim_group))
        self.wait(3)
